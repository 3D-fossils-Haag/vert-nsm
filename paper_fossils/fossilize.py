#!/usr/bin/env python3
"""
fossilize.py: turn modern vertebra meshes into fossil-like meshes.

    python paper_fossils/fossilize.py modern.ply fossil.ply --seed 7
    python paper_fossils/fossilize.py vertebrae_meshes/ fossilized/ -n 20 --variants 3 --workers 2
    python paper_fossils/fossilize.py vertebrae_meshes/ fossilized/ --set p_major=0.8 --set rough=0.5
    python paper_fossils/fossilize.py --settings                 # list every setting

Run from the repo root. Folder mode writes fossilized/fossilize_log.csv, which
align_fossil_morphoweave.py reads for each fossil's completeness.

Each bone gets a random mix of effects (breaks, wear, texture, deformation,
surface cracks); every tunable lives in SETTINGS below.
Lengths are fractions of the bone's size, so the same settings work in mm or m.

Requires: trimesh numpy scipy manifold3d pymeshfix (and vtk for .vtk files)
"""
from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import trimesh
from scipy.spatial import cKDTree

# ============================================================================
# SETTINGS: every tunable value, in one place
# ============================================================================
SETTINGS = dict(
    # --- how often each effect happens (chance per bone, 0-1) ---
    p_chip=0.5,          # small chips off process tips
    p_major=0.3,         # a major break
    p_lost=0.15,         # a whole process broken off at its base
    p_ground=0.1,        # a whole process worn away to a smooth nub
    p_smooth=0.5,        # water rounding
    p_erode=0.5,         # rock-tumbler abrasion
    p_roughen=0.7,       # a strong weathering-texture episode
    p_mineralize=0.4,    # a thick mineral-crust episode
    p_deform=0.5,        # shear / twist / flattening
    p_crack=0.5,         # surface cracks (never split the bone)
    p_spongy=0.7,        # thick breaks show porous spongy bone
    # --- fragmentation: how much of each bone survives (fraction of its volume) ---
    fragment=True,       # on/off
    completeness=(       # (share of outputs, min left, max left)
        (0.05, 0.2, 0.4),    # highly fragmented
        (0.10, 0.4, 0.6),    # very fragmented
        (0.35, 0.6, 0.8),    # lightly fragmented
        (0.50, 0.8, 1.0),    # mostly intact
    ),
    # --- breakage ---
    chips=(1, 3),                  # number of chips when a bone gets chipped
    majors=(1, 1),                 # number of major breaks
    chip_size=(0.005, 0.05),       # surface area removed per chip (fraction)
    major_size=(0.05, 0.15),       # surface area removed per major break
    break_relief=0.012,            # relief of fracture surfaces
    break_tilt=30,                 # max obliqueness of fractures (degrees)
    protect_breaks=0.95,           # 0-1: fresh breaks resist later wear/texture
    cortex=0.015,                  # dense-bone rim around spongy breaks
    pore=0.035,                    # spongy-bone pore size
    # --- wear ---
    smooth_iters=10,               # water rounding strength
    tumble_radius=(0.02, 0.06),    # rock-tumbler rounding radius, drawn per bone
    wear_elsewhere=0.1,            # wear away from break sites (0.1 = 10x gentler)
    wear_reach=0.15,               # how far down each process strong wear extends
    # --- texture ---
    rough=1.0,                     # weathering-texture strength
    rough_where="patchy",          # global | patchy | local | gradient
    mineral=1.0,                   # mineral-crust thickness
    mineral_where="global",
    light_texture=(0.08, 0.2),     # light texture after every step (0, 0 = off)
    strong_texture=(0.6, 1.4),     # strength of a roughen/mineralize episode
    # --- cracks (surface grooves only; the bone stays one piece) ---
    cracks=(1, 4),                 # number of cracks when a bone gets them
    crack_length=(0.3, 0.9),       # length of each crack
    crack_width=0.004,             # width of the groove
    crack_depth=0.004,             # depth (capped at 30 % of local thickness)
    crack_branching=0.25,          # how readily cracks branch
    # --- deformation ---
    shear=0.15,                    # parallelogram shear (0.2 tilts ~11 degrees)
    twist=0.0,                     # twist end to end, degrees
    flatten=0.0,                   # squash along compaction_axis (0.2 = 20 %)
    compaction_axis=(0, 0, 1),
    # --- mesh handling ---
    resolution=0.006,              # max edge length before texturing
    repair=True,                   # repair meshes with holes (needed for breaks)
    fix_flat_cuts=True,            # re-break perfectly flat crop planes in the input
)

STEP_ORDER = ("break", "smooth", "erode", "roughen", "deform", "crack", "mineralize")
TEXTURE_STEPS = ("roughen", "mineralize")


# ============================================================================
# Small helpers
# ============================================================================
def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def _smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def _rebuild(mesh, vertices):
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)


def _laplacian(mesh):
    return trimesh.smoothing.laplacian_calculation(mesh, equal_weight=True)


def _vertex_areas(mesh):
    a = np.zeros(len(mesh.vertices))
    np.add.at(a, mesh.faces.ravel(), np.repeat(mesh.area_faces / 3.0, 3))
    return a


def _cleanup(mesh):
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    return mesh


def _trim_memory():
    """Free unreachable objects and return freed memory to the OS (Linux)."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass


class FourierNoise:
    """Smooth 3D fractal noise from random plane waves; output roughly in [-1, 1]."""

    def __init__(self, scale, octaves=4, persistence=0.5, n_waves=48, rng=None):
        rng = np.random.default_rng(rng)
        K, phi, amp = [], [], []
        for o in range(octaves):
            k = _unit(rng.normal(size=(n_waves, 3)))
            K.append(k * (2 * np.pi / scale) * 2 ** o * rng.uniform(0.7, 1.3, (n_waves, 1)))
            phi.append(rng.uniform(0, 2 * np.pi, n_waves))
            amp.append(np.full(n_waves, persistence ** o / np.sqrt(n_waves)))
        self.K, self.phi, self.amp = np.vstack(K), np.concatenate(phi), np.concatenate(amp)
        self.norm = 2.0 * np.sqrt(np.sum(self.amp ** 2) / 2.0)

    def __call__(self, pts, chunk=40000):
        pts = np.asarray(pts, float)
        out = np.empty(len(pts))
        for i in range(0, len(pts), chunk):
            out[i:i + chunk] = np.sin(pts[i:i + chunk] @ self.K.T + self.phi) @ self.amp
        return out / self.norm


# ============================================================================
# Mesh utilities
# ============================================================================
def ensure_resolution(mesh, max_edge_rel=0.006):
    """Subdivide until no edge is longer than max_edge_rel x bone size."""
    max_edge = max_edge_rel * mesh.scale
    if mesh.edges_unique_length.max() <= max_edge:
        return mesh
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge, max_iter=8)
    return trimesh.Trimesh(v, f, process=True)


def refine_region(mesh, faces_mask, max_edge, max_rounds=3):
    """Split the masked triangles until edges <= max_edge; neighbours are split to
    match, so the mesh stays watertight (red-green refinement)."""
    m, sel = mesh, np.asarray(faces_mask, bool)
    for _ in range(max_rounds):
        f, v = np.array(m.faces), np.array(m.vertices)
        e = np.sort(np.c_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]].reshape(-1, 2), axis=1)
        elen = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1).reshape(-1, 3)
        sel = sel & (elen.max(1) > max_edge)
        if not sel.any():
            break
        keys = np.unique(e.reshape(-1, 3, 2)[sel].reshape(-1, 2), axis=0)
        mid = 0.5 * (v[keys[:, 0]] + v[keys[:, 1]])
        key_id = {tuple(k): len(v) + i for i, k in enumerate(keys)}
        faces, flags = [], []
        for fi, (a, b, c) in enumerate(f):
            ab = key_id.get((min(a, b), max(a, b)))
            bc = key_id.get((min(b, c), max(b, c)))
            ca = key_id.get((min(c, a), max(c, a)))
            split = (ab is not None, bc is not None, ca is not None)
            if not any(split):
                tris = [(a, b, c)]
            elif all(split):
                tris = [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
            elif split == (True, False, False):
                tris = [(a, ab, c), (ab, b, c)]
            elif split == (False, True, False):
                tris = [(a, b, bc), (a, bc, c)]
            elif split == (False, False, True):
                tris = [(a, b, ca), (ca, b, c)]
            elif ab is None:
                tris = [(a, b, bc), (a, bc, ca), (ca, bc, c)]
            elif bc is None:
                tris = [(a, ab, ca), (ab, b, c), (ab, c, ca)]
            else:
                tris = [(a, ab, bc), (a, bc, c), (ab, b, bc)]
            faces += tris
            flags += [sel[fi]] * len(tris)
        m = trimesh.Trimesh(np.vstack([v, mid]), np.array(faces), process=False)
        sel = np.array(flags)
    return m


def _refine_masked(mesh, vmask, max_edge):
    """refine_region on the faces touching vmask; returns (mesh, vmask on new mesh)."""
    n_old = len(mesh.vertices)
    mesh = refine_region(mesh, vmask[mesh.faces].any(1), max_edge)
    if len(mesh.vertices) > n_old:              # a midpoint is masked if both parents are
        _, near = cKDTree(mesh.vertices[:n_old]).query(mesh.vertices[n_old:], k=2)
        vmask = np.r_[vmask, vmask[near[:, 0]] & vmask[near[:, 1]]]
    return mesh, vmask


def transfer(src, dst, values):
    """Carry per-vertex values from src to dst by nearest vertex."""
    if values is None or (len(src.vertices) == len(dst.vertices)
                          and np.allclose(src.vertices, dst.vertices)):
        return values
    return np.asarray(values)[cKDTree(src.vertices).query(dst.vertices)[1]]


def fracture_mask(before, after, grow=3):
    """0-1 mask of vertices on `after` that weren't on `before` (new fracture
    surface), grown `grow` rings. Tolerant of the boolean engine's float32 rounding."""
    d, _ = cKDTree(before.vertices).query(after.vertices)
    f32 = np.finfo(np.float32).eps * np.abs(before.vertices).max()
    m = (d > max(0.05 * before.edges_unique_length.mean(), 20 * f32)).astype(float)
    L = _laplacian(after)
    for _ in range(grow):
        m = np.maximum(m, L @ m)
    return np.clip(m / max(m.max(), 1e-12), 0, 1)


class _Quiet:
    """Silence C-level stdout/stderr (pymeshfix prints directly)."""

    def __enter__(self):
        self.saved = []
        try:
            sys.stdout.flush(); sys.stderr.flush()
            self.null = os.open(os.devnull, os.O_WRONLY)
            for fd in (1, 2):
                self.saved.append((fd, os.dup(fd)))
                os.dup2(self.null, fd)
        except (OSError, ValueError):
            pass
        return self

    def __exit__(self, *exc):
        sys.stdout.flush(); sys.stderr.flush()
        for fd, saved in self.saved:
            os.dup2(saved, fd)
            os.close(saved)
        if self.saved:
            os.close(self.null)


def repair_mesh(mesh):
    """Make a mesh watertight (simple hole filling, then pymeshfix). Returns (mesh, note)."""
    if mesh.is_watertight:
        return mesh, ""
    m = mesh.copy()
    trimesh.repair.fill_holes(m)
    if m.is_watertight:
        return m, "repaired (filled holes)"
    try:
        import pymeshfix
    except ImportError:
        return mesh, "not watertight and pymeshfix isn't installed: breakage skipped"
    with _Quiet():
        v, f = pymeshfix.clean_from_arrays(np.ascontiguousarray(mesh.vertices, dtype=np.float64),
                                           np.ascontiguousarray(mesh.faces, dtype=np.int32))
    r = trimesh.Trimesh(v, f, process=True)
    if len(r.faces) and r.is_watertight:
        trimesh.repair.fix_normals(r)
        return r, "repaired (pymeshfix)"
    return mesh, "not watertight and repair failed: breakage skipped"


# ============================================================================
# Where effects apply
# ============================================================================
def region_mask(mesh, mode, rng):
    """Per-vertex 0-1 weights: global (~1 with mild variation), patchy (noise
    patches), local (a few blobs) or gradient (one side, +z, more affected)."""
    v, S = mesh.vertices, mesh.scale
    if mode == "global":
        return 1 - 0.15 * (0.5 + 0.5 * FourierNoise(0.4 * S, octaves=2, rng=rng)(v))
    if mode == "patchy":
        nz = FourierNoise(0.2 * S, octaves=2, rng=rng)(v)
        thr = np.quantile(nz, 0.5)
        return _smoothstep(nz, thr - 0.2, thr + 0.2)
    if mode == "local":
        centres, _ = trimesh.sample.sample_surface(mesh, 3, seed=int(rng.integers(1 << 31)))
        jag = FourierNoise(0.09 * S, octaves=2, rng=rng)(v)
        w = np.zeros(len(v))
        for c in centres:
            r = 0.15 * S * rng.uniform(0.6, 1.4)
            w = np.maximum(w, 1 - _smoothstep(np.linalg.norm(v - c, axis=1) / (r * (1 + 0.35 * jag)), 0.5, 1))
        return w
    if mode == "gradient":
        h = v[:, 2] - mesh.centroid[2]
        return _smoothstep(h / np.abs(h).max(), -0.25, 0.25)
    raise ValueError(f"unknown region {mode!r}: use global, patchy, local or gradient")


def resolve_region(mesh, region, rng):
    """Region name, per-vertex array, or a list of these (multiplied) -> weights."""
    if region is None:
        return np.ones(len(mesh.vertices))
    if isinstance(region, str):
        return region_mask(mesh, region, rng)
    if isinstance(region, list):
        w = np.ones(len(mesh.vertices))
        for r in region:
            w = w * resolve_region(mesh, r, rng)
        return w
    return np.clip(np.asarray(region, float), 0, 1)


def find_tips(mesh, n_max=8, percentile=88, min_separation=0.22):
    """Tips of the processes: points farthest from the centroid, spread apart."""
    v = np.array(mesh.vertices)          # plain array: indexing trimesh's is slow
    r = np.linalg.norm(v - mesh.centroid, axis=1)
    cand = np.where(r >= np.percentile(r, percentile))[0]
    sep, picked = min_separation * mesh.scale, []
    for i in cand[np.argsort(-r[cand])]:
        if not picked or np.linalg.norm(v[picked] - v[i], axis=1).min() > sep:
            picked.append(i)
            if len(picked) == n_max:
                break
    return v[picked]


def site_weights(mesh, base=0.1, reach=0.15, tips=None):
    """Wear weights: 1 at the process tips (break sites), fading to `base` elsewhere."""
    tips = find_tips(mesh) if tips is None else tips
    near = np.zeros(len(mesh.vertices))
    for t in np.atleast_2d(tips):
        d = np.linalg.norm(mesh.vertices - t, axis=1) / (reach * mesh.scale)
        near = np.maximum(near, 1 - _smoothstep(d, 0.45, 1.0))
    return base + (1 - base) * near


# ============================================================================
# Deformation
# ============================================================================
def deform(mesh, shear=0.15, twist=0.0, flatten=0.0, axis=(0, 0, 1), heterogeneity=0.25, rng=None):
    """Flatten along `axis`, shear like a parallelogram, and twist about the long axis."""
    rng = np.random.default_rng(rng)
    v, c, a = mesh.vertices.copy(), mesh.centroid, _unit(axis)
    vary = 1 + heterogeneity * FourierNoise(mesh.scale * 0.8, octaves=2, rng=rng)(v)
    if flatten:
        rel = v - c
        h = rel @ a
        s = np.clip(flatten * vary, 0, 0.9)
        v = c + (rel - h[:, None] * a) / np.sqrt(1 - s)[:, None] + (h * (1 - s))[:, None] * a
    if shear:
        sd = rng.normal(size=3)
        sd = _unit(sd - np.dot(sd, a) * a)
        v = v + (shear * vary * ((v - c) @ a))[:, None] * sd
    if twist:
        t = _unit(mesh.principal_inertia_vectors[0])        # long axis
        rel = v - c
        h = rel @ t
        theta = np.deg2rad(twist) * 0.5 * (h / np.abs(h).max()) * vary
        cos, sin = np.cos(theta)[:, None], np.sin(theta)[:, None]
        v = c + rel * cos + np.cross(t, rel) * sin + np.outer(h, t) * (1 - cos)
    return _rebuild(mesh, v)


# ============================================================================
# Breakage
# ============================================================================
def _fracture_relief(P, relief, feature, rng):
    """Height of a fracture surface over its plane: fractal undulation + terraces."""
    z = relief * FourierNoise(feature, octaves=2, persistence=0.55, rng=rng)(P)
    return z + 0.6 * relief * np.tanh(4 * FourierNoise(feature * 3, octaves=1, rng=rng)(P))


def _fracture_slab(origin, normal, half_width, depth, relief, feature, rng, n=160):
    """Watertight block whose lower face is a fracture surface through `origin`;
    everything on its +normal side gets cut away. The grid is fine near the break
    and coarse far away (x = hw sinh(a t) / sinh(a))."""
    target, dt = feature / 30, 2.0 / (n - 1)
    lo, hi = 1e-3, 12.0
    if half_width * dt > target:
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if half_width * mid / np.sinh(mid) * dt > target else (lo, mid)
    xs = half_width * np.sinh(hi * np.linspace(-1, 1, n)) / np.sinh(hi)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    P = np.c_[X.ravel(), Y.ravel(), np.zeros(n * n)]
    top = P.copy()
    top[:, 2] = np.minimum(_fracture_relief(P, relief, feature, rng), 0.8 * depth)
    bot = P.copy()
    bot[:, 2] = depth
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = (idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(),
                  idx[1:, 1:].ravel(), idx[:-1, 1:].ravel())
    ring = np.r_[idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]]
    r0, r1, nn = ring, np.roll(ring, -1), n * n
    faces = np.vstack([np.c_[a, c, b], np.c_[a, d, c],                       # top (-z)
                       np.c_[a, b, c] + nn, np.c_[a, c, d] + nn,             # bottom (+z)
                       np.c_[r0, r1 + nn, r1], np.c_[r0, r0 + nn, r1 + nn]])  # sides
    slab = trimesh.Trimesh(np.vstack([top, bot]), faces, process=False)
    if slab.volume < 0:
        slab.invert()
    T = trimesh.geometry.align_vectors([0, 0, 1], normal)
    T[:3, 3] = origin
    slab.apply_transform(T)
    return slab


def _break_once(mesh, tip, size, rng, cfg, at_base=False):
    """Break the process at `tip`. The fracture position along the process is
    searched so the original surface area removed lands within `size` (fractions).
    With at_base, the whole process goes: the cut sits just before the removed
    area jumps (where it would start entering the body).
    Returns (mesh, area_fraction, cut_centre) or None."""
    S, total_area = mesh.scale, mesh.area
    vert_area = _vertex_areas(mesh)
    tol = max(0.05 * mesh.edges_unique_length.mean(),
              20 * np.finfo(np.float32).eps * np.abs(mesh.vertices).max())
    tiny = 1e-5 * abs(mesh.volume)
    r_tip = np.linalg.norm(tip - mesh.centroid)
    near = cKDTree(mesh.vertices).query_ball_point(tip, 0.5 * r_tip)
    axis = _unit(tip - (mesh.vertices[near].mean(0) if len(near) >= 10 else mesh.centroid))
    tilt = np.deg2rad(rng.uniform(0, cfg["break_tilt"]))
    normal = _unit(np.cos(tilt) * axis + np.sin(tilt) * _unit(np.cross(axis, rng.normal(size=3))))
    noise_seed = int(rng.integers(1 << 31))       # same fracture shape on every try
    goal = rng.uniform(*size)

    def attempt(depth_frac):
        origin = tip - axis * depth_frac * r_tip
        rel = mesh.vertices - origin
        h = rel @ normal
        lat = np.linalg.norm(rel - h[:, None] * normal, axis=1)
        band = (np.abs(h) < 0.015 * S) & (lat < 0.6 * r_tip)
        w = np.clip(2 * np.percentile(lat[band], 75) if band.sum() > 5 else 0.1 * S, 0.03 * S, 0.4 * S)
        slab = _fracture_slab(origin, normal, 1.05 * lat.max() + 0.02 * S,
                              max(h.max(), 0) + 0.15 * S + 2 * w, cfg["break_relief"] * S,
                              0.05 * S, np.random.default_rng(noise_seed))
        removed = trimesh.boolean.intersection([mesh, slab], engine="manifold")
        pieces = [p for p in removed.split(only_watertight=False) if abs(p.volume) > tiny]
        if not pieces:
            return slab, None, [], 0.0
        k = int(np.argmin([cKDTree(p.vertices).query(tip)[0] for p in pieces]))
        d, _ = cKDTree(pieces[k].vertices).query(mesh.vertices)
        return slab, pieces[k], pieces[:k] + pieces[k + 1:], vert_area[d < tol].sum() / total_area

    best = None
    if at_base:
        depths = np.linspace(0.08, 0.9, 12)
        a_orig, a_piece = np.full(12, np.nan), np.full(12, np.nan)
        for i, dfr in enumerate(depths):
            try:
                _, piece, _, fa = attempt(dfr)
                if piece is not None:
                    a_orig[i], a_piece[i] = fa * total_area, piece.area
            except Exception:  # noqa: BLE001
                pass
            gc.collect()
        a_frac = a_piece - a_orig
        width = 2 * np.sqrt(np.clip(a_frac, 0, None) / np.pi)
        # a real process: carries much more surface than its fracture face, isn't a
        # flat shaving, and isn't too big
        good = (np.isfinite(a_orig) & (a_orig <= 0.2 * total_area) & (a_orig >= 2 * a_frac)
                & (depths * r_tip >= 0.7 * width))
        inc = np.diff(np.where(np.isfinite(a_orig), a_orig / total_area, np.inf))
        k = len(depths) - 1
        for j in range(1, len(inc)):                 # first clear jump in removed area
            prev = inc[:j][np.isfinite(inc[:j])]
            if not np.isfinite(inc[j]) or inc[j] > max(2.5 * (np.median(prev) if len(prev) else inc[j]), 0.02):
                k = j
                break
        while k >= 0 and not good[k]:
            k -= 1
        if k < 1:
            return None
        slab, piece, others, fa = attempt(depths[k])
        if piece is None:
            return None
        best = (slab, others, fa, tip - axis * depths[k] * r_tip)
    else:
        lo, hi, depth = 0.0, 1.0, 0.3
        for _ in range(7):                            # bisection on depth
            try:
                slab, piece, others, fa = attempt(depth)
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"break failed: {e}")
                return None
            hit = piece is not None and size[0] <= fa <= size[1]
            if hit:
                best = (slab, others, fa, tip - axis * depth * r_tip)
                if abs(fa - goal) < 0.25 * (size[1] - size[0]):
                    break
            lo, hi = (depth, hi) if fa < goal else (lo, depth)
            depth = 0.5 * (lo + hi)
        if best is None:
            return None

    slab, others, fa, centre = best
    try:
        kept = trimesh.boolean.difference([mesh, slab], engine="manifold")
        if others:                                   # other structures crossed: put back
            kept = trimesh.boolean.union([kept] + others, engine="manifold")
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"break skipped: {e}")
        return None
    parts = sorted([p for p in kept.split(only_watertight=False) if abs(p.volume) > tiny],
                   key=lambda p: -abs(p.volume))
    if not parts or any(abs(p.volume) > 0.002 * abs(mesh.volume) for p in parts[1:]):
        return None                                  # would detach another structure
    res = _cleanup(parts[0])
    _trim_memory()
    return (res, fa, centre) if res.is_watertight else None


def break_tips(mesh, plan, cfg, rng, relief_scale=1.0):
    """Carry out `plan` (list of 'chip' / 'major' / 'lost' / 'ground') on randomly
    chosen process tips. Returns (mesh, report)."""
    rep = dict(chip=0, major=0, lost=0, ground=0, area=0.0)
    if not plan or not mesh.is_watertight:
        return mesh, rep
    cfg = dict(cfg, break_relief=cfg["break_relief"] * relief_scale)
    tips = find_tips(mesh)
    order = list(rng.permutation(len(tips)))
    area0 = mesh.area
    for kind in plan:
        size = cfg["chip_size"] if kind == "chip" else cfg["major_size"]
        while order:                                  # next tip that breaks
            area_before = mesh.area
            res = _break_once(mesh, tips[order.pop(0)], size, rng, cfg, at_base=kind in ("lost", "ground"))
            if res:
                mesh, fa, _ = res
                rep[kind] += 1
                rep["area"] += fa * area_before / area0
                break
    return mesh, rep


def fragment(mesh, target, vol0, cfg, rng):
    """Cut the bone down to `target` x vol0 of its original volume with 1-3 large
    fractures through the body; the biggest piece is kept. Each cut's position is
    searched so the fragment lands on its share of the remaining loss.
    Returns (mesh, number of cuts)."""
    S = mesh.scale
    n_cuts = 1 if target > 0.75 else 2 if target > 0.5 else 3
    done = 0
    for k in range(n_cuts):
        now = abs(mesh.volume) / vol0
        goal = now * (target / now) ** (1 / (n_cuts - k))       # this cut's share
        normal = _unit(rng.normal(size=3))
        seed = int(rng.integers(1 << 31))
        h_all = (mesh.vertices - mesh.centroid) @ normal
        lo, hi, best = h_all.min(), h_all.max(), None

        def cut(s):
            origin = mesh.centroid + normal * s
            rel = mesh.vertices - origin
            h = rel @ normal
            slab = _fracture_slab(origin, normal, 1.05 * np.linalg.norm(rel - h[:, None] * normal, axis=1).max()
                                  + 0.02 * S, max(h.max(), 0) + 0.15 * S, cfg["break_relief"] * S,
                                  0.05 * S, np.random.default_rng(seed))
            kept = trimesh.boolean.difference([mesh, slab], engine="manifold")
            parts = kept.split(only_watertight=False)
            return max(parts, key=lambda p: abs(p.volume)) if len(parts) else None

        for _ in range(9):                         # bisection on the cut position
            s = 0.5 * (lo + hi)
            try:
                piece = cut(s)
            except Exception:  # noqa: BLE001
                piece = None
            if piece is None:
                lo = s
                continue
            frac = abs(piece.volume) / vol0
            if best is None or abs(frac - goal) < abs(best[1] - goal):
                best = (piece, frac)
            if abs(frac - goal) < 0.02:
                break
            lo, hi = (s, hi) if frac < goal else (lo, s)
            gc.collect()
        if best is not None:
            res = _cleanup(best[0])
            if res.is_watertight:
                mesh, done = res, done + 1
        _trim_memory()
    return mesh, done


def find_flat_cuts(mesh, min_area=0.002, max_rms=0.0005):
    """Perfectly flat regions (CT crop planes, flat hole fills); real bone is never
    this flat. Returns a list of face-index arrays."""
    import scipy.sparse as sp
    import scipy.sparse.csgraph as cg
    adj, ang = mesh.face_adjacency, mesh.face_adjacency_angles
    k = ang < np.radians(1.0)
    g = sp.coo_matrix((np.ones(k.sum()), (adj[k, 0], adj[k, 1])), shape=(len(mesh.faces),) * 2)
    _, lab = cg.connected_components(g, directed=False)
    areas = np.bincount(lab, weights=mesh.area_faces)
    out = []
    for t in np.where(areas > min_area * mesh.area)[0]:
        fi = np.where(lab == t)[0]
        P = mesh.triangles_center[fi] - mesh.triangles_center[fi].mean(0)
        if np.sqrt(((P @ np.linalg.svd(P, full_matrices=False)[2][2]) ** 2).mean()) < max_rms * mesh.scale:
            out.append(fi)
    return out


def _plane(mesh, fi):
    P = mesh.triangles_center[fi]
    c = P.mean(0)
    vt = np.linalg.svd(P - c, full_matrices=False)[2]
    n = vt[2] if vt[2] @ mesh.face_normals[fi].mean(0) > 0 else -vt[2]
    return c, n, vt


def recut_flat_cuts(mesh, relief, rng):
    """Re-break flat crop planes as natural fractures: a fresh fracture is cut just
    beneath each plane. Flat patches inside a surface (filled holes) get fracture
    relief in place instead. Returns (mesh, 0-1 mask of the new surfaces)."""
    S, out, embedded = mesh.scale, mesh, []
    for fi in sorted(find_flat_cuts(mesh), key=lambda f: -mesh.area_faces[f].sum()):
        c, n, _ = _plane(mesh, fi)
        w = float(np.clip(2 * np.sqrt(mesh.area_faces[fi].sum() / np.pi), 0.03 * S, 0.4 * S))
        seed = int(rng.integers(1 << 31))
        probe = _fracture_relief(np.c_[rng.uniform(-w, w, (4000, 2)), np.zeros(4000)],
                                 relief * S, 0.05 * S, np.random.default_rng(seed))
        origin = c - n * (probe.max() + 0.01 * S)       # whole fracture below the plane
        rel = out.vertices - origin
        h = rel @ n
        tiny = 1e-5 * abs(out.volume)
        try:
            slab = _fracture_slab(origin, n, 1.05 * np.linalg.norm(rel - h[:, None] * n, axis=1).max()
                                  + 0.02 * S, max(h.max(), 0) + 0.15 * S + 2 * w, relief * S,
                                  0.05 * S, np.random.default_rng(seed))
            removed = trimesh.boolean.intersection([out, slab], engine="manifold")
            kept = trimesh.boolean.difference([out, slab], engine="manifold")
            tree = cKDTree(mesh.vertices[np.unique(mesh.faces[fi])])
            others = [p for p in removed.split(only_watertight=False)
                      if abs(p.volume) > tiny and tree.query(p.vertices)[0].min() > 0.02 * S]
            if others:
                kept = trimesh.boolean.union([kept] + others, engine="manifold")
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"re-cutting a flat crop plane failed: {e}")
            continue
        parts = sorted([p for p in kept.split(only_watertight=False) if abs(p.volume) > tiny],
                       key=lambda p: -abs(p.volume))
        if not parts or abs(parts[0].volume) < 0.85 * abs(out.volume):
            embedded.append(fi)                       # would remove too much bone
            continue
        res = _cleanup(parts[0])
        if res.is_watertight:
            out = res
    mask = fracture_mask(mesh, out, grow=0) if out is not mesh else np.zeros(len(mesh.vertices))
    for fi in embedded:                               # relief in place, removing material only
        c, n, vt = _plane(mesh, fi)
        vi = np.unique(np.concatenate(cKDTree(out.vertices).query_ball_point(
            mesh.vertices[np.unique(mesh.faces[fi])], 1e-6 * S)))
        if len(vi) < 3:
            continue
        v = out.vertices.copy()
        Q = v[vi] - c
        z = _fracture_relief(np.c_[Q @ vt[0], Q @ vt[1], np.zeros(len(Q))], relief * S, 0.05 * S, rng)
        rim, _ = cKDTree(np.delete(v, vi, axis=0)).query(v[vi])
        v[vi] += ((z - z.max()) * _smoothstep(rim, 0.0, 0.03 * S))[:, None] * n
        out = _rebuild(out, v)
        mask[vi] = 1.0
    return out, mask


def finish_fracture(mesh, fracture, relief, spongy, cortex=0.015, pore=0.035, depth=0.016, rng=None):
    """Texture fresh fracture faces: refine them locally, add fine roughness, and if
    `spongy`, carve porous spongy bone inside a rim of dense bone (`cortex` wide).
    Thin breaks narrower than ~2 x cortex stay plain."""
    rng = np.random.default_rng(rng)
    fv = np.asarray(fracture) > 0.5
    if fv.sum() < 3 or fv.all():
        return mesh
    S = mesh.scale
    mesh, fv = _refine_masked(mesh, fv, min(0.005, pore / 7) * S)
    v, n = mesh.vertices.copy(), mesh.vertex_normals
    rim, _ = cKDTree(v[~fv]).query(v[fv])            # distance in from the face's edge
    tex = 0.25 * relief * S * FourierNoise(0.05 * S / 3, octaves=2, rng=rng)(v[fv])
    v[fv] += (tex * _smoothstep(rim, 0.0, 0.01 * S))[:, None] * n[fv]
    fade = _smoothstep(rim, cortex * S, 1.6 * cortex * S)
    if not spongy or not (fade > 0).any():
        return _rebuild(mesh, v)
    # spongy bone: two sizes of bowl-shaped pores from warped cellular noise
    pts, P = v[fv], pore * S
    q = pts + 0.35 * P * np.c_[[FourierNoise(1.5 * P, octaves=2, rng=rng)(pts) for _ in range(3)]].T

    def cells(size):
        lo, hi = q.min(0) - size, q.max(0) + size
        k = int(np.clip(np.prod(hi - lo) / size ** 3 * 2.0, 50, 300000))
        (f1, f2), (i1, _) = [x.T for x in cKDTree(rng.uniform(lo, hi, (k, 3))).query(q, k=2)]
        bowl = np.clip(1 - (f1 / np.maximum(0.5 * (f1 + f2), 1e-12)) ** 2, 0, 1) ** 0.5
        wall_w = size * (0.04 + 0.06 * (0.5 + 0.5 * FourierNoise(2 * size, octaves=1, rng=rng)(pts)))
        wall = 1 - _smoothstep(f2 - f1, 0.5 * wall_w, 1.5 * wall_w)
        return bowl * (1 - wall) * rng.uniform(0.5, 1.0, k)[i1.astype(int)]

    disp = np.zeros(len(v))
    disp[fv] = depth * S * np.maximum(cells(P), 0.6 * cells(0.5 * P)) * fade
    L = _laplacian(mesh)
    for _ in range(2):                               # round off the walls between pores
        disp = 0.5 * disp + 0.5 * (L @ disp)
    disp[~fv] = 0
    return _rebuild(mesh, v - disp[:, None] * mesh.vertex_normals)


# ============================================================================
# Wear
# ============================================================================
def water_smooth(mesh, iterations=10, strength=0.5, shrink=0.3, weights=None):
    """Water rounding (Taubin smoothing). Vertices with weight 0.1 get ~10x less
    change than weight 1: fewer passes and a smaller step."""
    w0 = np.ones(len(mesh.vertices)) if weights is None else weights
    L, v = _laplacian(mesh), mesh.vertices.copy()
    mu = -(strength + 0.02) * (1 - shrink)
    for k in range(iterations):
        w = (strength * w0 ** 0.33 * np.clip(w0 * iterations - k, 0, 1))[:, None]
        if not w.any():
            break
        v = v + w * (L @ v - v)
        v = v + (mu / strength) * w * (L @ v - v)
    return _rebuild(mesh, v)


def abrade(mesh, iterations=20, rate=0.6, weights=None):
    """Fine abrasion: only removes material; bumps wear, hollows are spared."""
    w0 = np.ones(len(mesh.vertices)) if weights is None else weights
    L, v = _laplacian(mesh), mesh.vertices.copy()
    passes, w_step = np.clip(w0, 0, 1) * iterations, w0 ** 0.27
    for it in range(iterations):
        if it % 3 == 0:
            n = _rebuild(mesh, v).vertex_normals
        w = w_step * np.clip(passes - it, 0, 1)
        d = L @ v - v
        dn = np.einsum("ij,ij->i", d, n)
        v = v + (rate * w * np.minimum(dn, 0))[:, None] * n + (0.3 * rate * w)[:, None] * (d - dn[:, None] * n)
    return _rebuild(mesh, v)


def _grid_sample(field, origin, pitch, pts):
    from scipy.ndimage import map_coordinates
    return map_coordinates(field, ((pts - origin) / pitch).T, order=1, mode="nearest")


def _gauss_curvature_grid(phi, pitch):
    """Gaussian curvature of the level sets of a signed-distance grid, and |grad|."""
    gx, gy, gz = np.gradient(phi, pitch)
    gxx, gxy, gxz = np.gradient(gx, pitch)
    _, gyy, gyz = np.gradient(gy, pitch)
    gzz = np.gradient(gz, pitch, axis=2)
    num = (gx * gx * (gyy * gzz - gyz * gyz) + gy * gy * (gxx * gzz - gxz * gxz)
           + gz * gz * (gxx * gyy - gxy * gxy) + 2 * gx * gy * (gxz * gyz - gxy * gzz)
           + 2 * gy * gz * (gxy * gxz - gyz * gxx) + 2 * gx * gz * (gxy * gyz - gxz * gyy))
    g2 = gx * gx + gy * gy + gz * gz
    return num / np.maximum(g2 * g2, 1e-12), np.sqrt(g2)


def tumble(mesh, radius=0.03, weights=None, polish=6, resolution=160):
    """Rock-tumbler abrasion: points and corners (high Gaussian curvature) wear
    until rounded to ~`radius`; flat faces, rod sides and hollows barely wear.
    Simulated on a signed-distance grid, then the mesh is moved onto the result."""
    from scipy.ndimage import distance_transform_edt as edt, gaussian_filter
    w = np.ones(len(mesh.vertices)) if weights is None else weights
    S, R, m = mesh.scale, radius * mesh.scale, mesh
    if R > 0 and m.is_watertight:
        pitch = min(R / 3.0, S / resolution)
        vox = m.voxelized(pitch).fill()
        pad = int(np.ceil(1.5 * R / pitch)) + 3
        occ = np.pad(vox.matrix, pad)
        origin = vox.transform[:3, 3] - pad * pitch
        phi0 = gaussian_filter(((edt(~occ) - edt(occ)) * pitch).astype(np.float32), 0.8)
        idx = np.argwhere(np.abs(phi0) < 3 * R)        # wear rate on the grid
        wg = np.zeros(phi0.shape, np.float32)
        wg[tuple(idx.T)] = w[cKDTree(m.vertices).query(origin + idx * pitch)[1]]
        wg = gaussian_filter(wg, 1.0)
        total, step = 1.2 * R, 0.4 * pitch                # max wear depth, per-step move
        band, phi = np.abs(phi0) < total + 3 * R, phi0.copy()
        for k in range(int(np.ceil(total / step))):
            if k % 3 == 0:
                K, gnorm = _gauss_curvature_grid(gaussian_filter(phi, 1.0), pitch)
                speed = np.clip(K * R * R - 1.0, 0, 1.0) * wg * band
                gnorm = np.clip(gnorm, 0, 1.5)
            phi = phi + step * speed * gnorm
        phi = gaussian_filter(np.minimum(phi, phi0 + total), 0.8)
        # move vertices onto the worn surface, keeping triangles even
        v0 = m.vertices.copy()
        v = v0.copy()
        bias = _grid_sample(gaussian_filter(phi0, 0.8), origin, pitch, v)
        grads = np.gradient(phi, pitch)
        L, moved = _laplacian(m), np.zeros(len(v), bool)
        for it in range(6):
            d = _grid_sample(phi, origin, pitch, v) - bias
            g = _unit(np.c_[[_grid_sample(gi, origin, pitch, v) for gi in grads]].T)
            go = d > 0.02 * pitch
            moved |= go
            v[go] -= np.minimum(d[go], 0.5 * total)[:, None] * g[go]
            off = v - v0
            dist = np.linalg.norm(off, axis=1)
            over = dist > 1.3 * total
            v[over] = v0[over] + off[over] * (1.3 * total / dist[over])[:, None]
            if it < 5:
                n = _rebuild(m, v).vertex_normals
                dl = L @ v - v
                dl -= np.einsum("ij,ij->i", dl, n)[:, None] * n
                v = v + np.clip(moved + (L @ moved.astype(float)), 0, 1)[:, None] * 0.5 * dl
                bias = np.where(moved, 0.0, bias)
        disp = v - v0
        for _ in range(6):
            disp = 0.5 * disp + 0.5 * (L @ disp)
        m = _rebuild(m, v0 + disp)
    return abrade(m, iterations=polish, weights=w) if polish else m


# ============================================================================
# Texture
# ============================================================================
def roughen(mesh, strength=1.0, region="patchy", rng=None, amplitude=0.0018, feature=0.025,
            bias=-0.15, crack_depth=0.0012, n_pits=30, pit_depth=0.0018):
    """Weathering texture: fractal relief, cracks and pits, displaced along normals."""
    rng = np.random.default_rng(rng)
    S, v, n = mesh.scale, mesh.vertices, mesh.vertex_normals
    mask = resolve_region(mesh, region, rng)
    disp = strength * amplitude * S * mask * (FourierNoise(feature * S, octaves=3, rng=rng)(v) + bias)
    crack = FourierNoise(feature * S * 4, octaves=2, persistence=0.4, rng=rng)(v)
    disp -= strength * crack_depth * S * mask * np.clip(1 - np.abs(crack) / 0.06, 0, 1) ** 2
    tree = cKDTree(v)
    for c in v[rng.choice(len(v), size=n_pits, p=mask / mask.sum())] if mask.sum() > 0 else []:
        rad = rng.uniform(0.004, 0.012) * S
        idx = np.array(tree.query_ball_point(c, 2.5 * rad))
        if idx.size:
            d2 = np.sum((v[idx] - c) ** 2, axis=1)
            disp[idx] -= strength * pit_depth * S * rng.uniform(0.4, 1) * np.exp(-d2 / (2 * (rad / 1.5) ** 2))
    return _rebuild(mesh, v + disp[:, None] * n)


def mineralize(mesh, strength=1.0, region="global", rng=None, thickness=0.002, grain=0.008,
               granularity=0.3, evenness=0.85, fill_hollows=0.5):
    """Add a granular mineral crust, thicker in hollows."""
    rng = np.random.default_rng(rng)
    S, v, n = mesh.scale, mesh.vertices, mesh.vertex_normals
    bumps = np.clip(FourierNoise(grain * S, octaves=2, persistence=0.35, rng=rng)(v), 0, None) ** 0.7
    layer = (1 - granularity) + granularity * 2.0 * bumps
    layer *= evenness + (1 - evenness) * (1 + FourierNoise(0.3 * S, octaves=2, rng=rng)(v))
    concavity = np.einsum("ij,ij->i", _laplacian(mesh) @ v - v, n)
    layer *= 1 + fill_hollows * np.clip(concavity / (0.05 * mesh.edges_unique_length.mean()), 0, 3)
    disp = strength * thickness * S * resolve_region(mesh, region, rng) * np.clip(layer, 0, None)
    return _rebuild(mesh, v + disp[:, None] * n)


# ============================================================================
# Cracks
# ============================================================================
def _crack_paths(mesh, n, length_range, branching, rng, step):
    """Trace cracks across the surface. Each runs both ways from a random start:
    half run lengthwise along the centrum (between condyle and cotyle), ~30 %
    wrap around the bone, the rest go anywhere. They meander and sometimes
    branch. Returns a list of (N x 3) polylines."""
    V, tri, fn = np.array(mesh.vertices), mesh.triangles, np.array(mesh.face_normals)
    vtree, vfaces = cKDTree(V), np.array(mesh.vertex_faces)

    def snap(x):
        """Closest surface point to x, searching only triangles around nearby vertices."""
        faces = np.unique(vfaces[vtree.query(x, k=8)[1]].ravel())
        faces = faces[faces >= 0]
        cp = trimesh.triangles.closest_point(tri[faces], np.repeat(x[None], len(faces), 0))
        j = int(np.argmin(np.linalg.norm(cp - x, axis=1)))
        return cp[j], np.linalg.norm(cp[j] - x), faces[j]

    def walk(p, d, L, branches):
        pts = [p]
        for _ in range(int(L / step)):
            q, dist, fi = snap(pts[-1] + d * step)
            if dist > 2 * step:                      # walked off an edge
                break
            nrm = fn[fi]
            ang = rng.normal(0, np.radians(12))      # meander about the normal
            d = _unit(d - (d @ nrm) * nrm)
            d = _unit(d * np.cos(ang) + np.cross(nrm, d) * np.sin(ang))
            pts.append(q)
            if branches is not None and rng.random() < branching * step / (0.05 * mesh.scale):
                turn = rng.choice([-1, 1]) * np.radians(rng.uniform(25, 55))
                branches.append((q, _unit(d * np.cos(turn) + np.cross(nrm, d) * np.sin(turn)),
                                 L * rng.uniform(0.4, 1.0)))
        return np.array(pts)

    # long axis = centrum axis (condyle to cotyle); the centrum is the part of the
    # surface closest to that axis
    long_axis = _unit(mesh.principal_inertia_vectors[0])
    rel = mesh.triangles_center - mesh.centroid
    radial = np.linalg.norm(rel - np.outer(rel @ long_axis, long_axis), axis=1)
    on_centrum = radial < np.percentile(radial, 40)
    paths = []
    for _ in range(n):
        kind = rng.choice(["along", "around", "random"], p=[0.5, 0.3, 0.2])
        pool = np.where(on_centrum)[0] if kind == "along" else np.arange(len(mesh.faces))
        fi = rng.choice(pool, p=mesh.area_faces[pool] / mesh.area_faces[pool].sum())
        bary = rng.dirichlet([1, 1, 1])
        p = bary @ mesh.triangles[fi]
        nrm = mesh.face_normals[fi]
        d = {"along": long_axis, "around": np.cross(nrm, long_axis),
             "random": rng.normal(size=3)}[kind]
        d = _unit(d - (d @ nrm) * nrm)
        half = rng.uniform(*length_range) * mesh.scale / 2
        branches = []
        a, b = walk(p, -d, half, branches), walk(p, d, half, branches)
        paths.append(np.vstack([a[::-1], b[1:]]))     # one continuous crack
        paths += [walk(bp, bd, bl, None) for bp, bd, bl in branches]
    return [pth for pth in paths if len(pth) > 3]


def _thickness(v, nrm, idx, max_t):
    """Bone thickness behind vertices idx: distance to the nearest surface point on
    the opposite side (behind it, facing the other way); max_t if none within reach.
    Uses a point lookup instead of ray casting, which needs a lot of memory."""
    tree = cKDTree(v)
    out = np.full(len(idx), max_t)
    for k, i in enumerate(idx):
        j = np.asarray(tree.query_ball_point(v[i], max_t))
        rel = v[j] - v[i]
        depth = -(rel @ nrm[i])                          # how far behind the surface
        lateral = np.linalg.norm(rel + depth[:, None] * nrm[i], axis=1)
        ok = (depth > 0) & (nrm[j] @ nrm[i] < -0.3) & (lateral < 0.5 * depth + 1e-12)
        if ok.any():
            out[k] = depth[ok].min()
    return out


def crack(mesh, count=(1, 4), length=(0.3, 0.9), width=0.004, depth=0.004, branching=0.25,
          rng=None):
    """Surface cracks: narrow grooves pressed into the surface along meandering,
    branching paths. Only vertices move, so the bone stays one watertight piece;
    depth is capped at 30 % of the local bone thickness."""
    rng = np.random.default_rng(rng)
    S = mesh.scale
    n = int(rng.integers(count[0], count[1] + 1))
    paths = _crack_paths(mesh, n, length, branching, rng, step=0.004 * S)
    if not paths:
        return mesh
    # resample paths densely; each sample gets a width/depth factor that varies
    # along the crack and tapers to nothing at its ends
    pts, fac = [], []
    for path in paths:
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        t = np.r_[0, np.cumsum(seg)]
        k = max(int(t[-1] / (0.0015 * S)), 2)
        ts = np.linspace(0, t[-1], k)
        pts.append(np.c_[[np.interp(ts, t, path[:, i]) for i in range(3)]].T)
        taper = np.sin(np.pi * np.clip(ts / t[-1], 0, 1)) ** 0.4
        wobble = 0.75 + 0.25 * FourierNoise(0.05 * S, octaves=2, rng=rng)(pts[-1])
        fac.append(taper * wobble)
    pts, fac = np.vstack(pts), np.concatenate(fac)
    W = width * S
    # refine the mesh along the cracks so a hairline groove can be represented
    near = cKDTree(pts).query(mesh.vertices, distance_upper_bound=1.5 * W)[0] < np.inf
    mesh, near = _refine_masked(mesh, near, W / 2)
    v, nrm = np.array(mesh.vertices), mesh.vertex_normals
    d, i = cKDTree(pts).query(v, distance_upper_bound=1.2 * W)
    hit = np.isfinite(d)
    if not hit.any():
        return mesh
    w_local = W * (0.5 + 0.5 * fac[i[hit]])
    groove = depth * S * fac[i[hit]] * np.clip(1 - d[hit] / w_local, 0, 1) ** 1.5
    # never deeper than 30 % of the bone's thickness (measured along the crack)
    on_crack = cKDTree(v).query(pts)[1]
    thick = _thickness(v, nrm, on_crack, 0.06 * S)
    groove = np.minimum(groove, 0.3 * thick[i[hit]])
    v[hit] -= groove[:, None] * nrm[hit]
    return _rebuild(mesh, v)


# ============================================================================
# Pipeline
# ============================================================================
class _Bone:
    """The mesh being worked on, plus two 0-1 masks: `frac` (fresh fractures, shielded
    from later wear) and `broken` (everything that isn't original surface: fractures
    and worn-away stumps; saved with the output for preprocessing)."""

    def __init__(self, mesh):
        self.mesh, self.frac, self.broken = mesh, None, None

    def update(self, new):
        """Swap in a new mesh, carrying both masks across remeshing."""
        if new is not self.mesh:
            self.frac = transfer(self.mesh, new, self.frac)
            self.broken = transfer(self.mesh, new, self.broken)
        self.mesh = new

    def mark_broken(self, mask):
        self.broken = mask if self.broken is None else np.maximum(self.broken, mask)

    def refine(self, res):
        self.update(ensure_resolution(self.mesh, res))

    def apply_break(self, new):
        """Swap in a broken mesh and mark its new fracture surfaces."""
        fresh = fracture_mask(self.mesh, new)
        self.update(new)
        self.frac = fresh if self.frac is None else np.maximum(fresh, self.frac)
        self.mark_broken(fresh)

    def shield(self, amount):
        """Weights that protect fresh fractures from later wear/texture."""
        return None if self.frac is None else 1 - amount * self.frac


def plan_steps(cfg, rng, steps=None):
    """Draw which effects happen. Returns (steps, break plan, tumble radius, spongy,
    completeness target)."""
    plan = (["ground"] * (rng.random() < cfg["p_ground"]) + ["lost"] * (rng.random() < cfg["p_lost"])
            + ["major"] * (int(rng.integers(cfg["majors"][0], cfg["majors"][1] + 1))
                           if rng.random() < cfg["p_major"] else 0)
            + ["chip"] * (int(rng.integers(cfg["chips"][0], cfg["chips"][1] + 1))
                          if rng.random() < cfg["p_chip"] else 0))
    radius = float(rng.uniform(*cfg["tumble_radius"]))
    spongy = rng.random() < cfg["p_spongy"]
    target = 1.0
    if cfg["fragment"]:
        shares = np.array([c[0] for c in cfg["completeness"]], float)
        lo, hi = cfg["completeness"][rng.choice(len(shares), p=shares / shares.sum())][1:]
        target = float(rng.uniform(lo, hi))
    if steps is None:
        if target < 0.97 and not plan:     # fragmenting needs the break step
            plan = ["fragment"]
        others = [s for s in STEP_ORDER[1:] if rng.random() < cfg[f"p_{s}"]]
        if not others and not plan:            # at least one effect per bone
            others = [str(rng.choice(STEP_ORDER[1:]))]
        steps = (["break"] if plan else []) + [s for s in others if s not in TEXTURE_STEPS]
        for t in [s for s in others if s in TEXTURE_STEPS]:   # texture at a random point
            steps.insert(int(rng.integers(0, len(steps) + 1)), t)
    elif "break" in steps and not plan:
        plan = ["chip"] * cfg["chips"][0]
    return list(steps), [k for k in plan if k != "fragment"], radius, spongy, target


def fossilize(mesh, seed=None, steps=None, **settings):
    """Fossilize one mesh. `settings` override SETTINGS; `steps` (e.g.
    ["break", "roughen"]) runs exactly those effects instead of random ones.
    What was done is stored in result.metadata["fossilize"]; which surface is
    broken (1) vs original (0) is in result.vertex_attributes["broken"] and is
    saved with the file (a point array in .vtk, a vertex property in .ply)."""
    unknown = set(settings) - set(SETTINGS)
    if unknown:
        raise ValueError(f"unknown setting(s): {', '.join(sorted(unknown))}")
    cfg = {**SETTINGS, **settings}
    rng = np.random.default_rng(seed)
    steps, plan, radius, spongy, target = plan_steps(cfg, rng, steps)
    S_res, guard = cfg["resolution"], cfg["protect_breaks"]

    m = mesh.copy()
    offset = m.centroid.copy()
    m.apply_translation(-offset)          # work near the origin (boolean precision)
    log, rep = [], {}
    if cfg["repair"] and not m.is_watertight:
        m, note = repair_mesh(m)
        log.append(f"[{note}]")
    bone = _Bone(m)

    if cfg["fix_flat_cuts"] and find_flat_cuts(bone.mesh):
        new, cut = recut_flat_cuts(bone.mesh, cfg["break_relief"], rng)
        if cut.any():
            fine = ensure_resolution(new, S_res)
            cut = transfer(new, fine, cut)
            bone.mesh = finish_fracture(fine, cut, cfg["break_relief"], spongy and cfg["p_spongy"] > 0,
                                        cfg["cortex"], cfg["pore"], rng=rng)
            bone.frac = transfer(fine, bone.mesh, cut)
            bone.broken = bone.frac.copy()
            log.append("[flat crop plane(s) re-broken]")

    def texture(name, strength, region=None):
        fn = roughen if name == "roughen" else mineralize
        base = region or (cfg["rough_where"] if name == "roughen" else cfg["mineral_where"])
        knob = cfg["rough"] if name == "roughen" else cfg["mineral"]
        shield = bone.shield(guard)
        bone.update(fn(bone.mesh, knob * strength, base if shield is None else [base, shield], rng=rng))

    def wear_weights():
        w = site_weights(bone.mesh, cfg["wear_elsewhere"], cfg["wear_reach"])
        shield = bone.shield(guard)
        return w if shield is None else w * shield

    for name in steps:
        if name not in ("break", "deform"):
            bone.refine(S_res)
        if name == "break":
            rep = _break_step(bone, plan, cfg, spongy, target, rng)
            parts = [f"{rep[k]} {k}" for k in ("lost", "ground", "major", "chip", "cut") if rep[k]]
            parts += ["%.0f%% complete" % (100 * rep.get("complete", 1.0))]
            parts += ["spongy"] if rep.get("spongy") else []
            log.append("break (" + ", ".join(parts) + ")")
        elif name in TEXTURE_STEPS:
            k = rng.uniform(*cfg["strong_texture"])
            texture(name, k)
            log.append(f"{name} x{k:.2f}")
        elif name == "smooth":
            bone.update(water_smooth(bone.mesh, cfg["smooth_iters"], weights=wear_weights()))
            log.append("smooth")
        elif name == "erode":
            bone.update(tumble(bone.mesh, radius, weights=wear_weights()))
            log.append(f"erode (tumble r={radius:.3f})")
        elif name == "crack":
            bone.update(crack(bone.mesh, cfg["cracks"], cfg["crack_length"], cfg["crack_width"],
                              cfg["crack_depth"], cfg["crack_branching"], rng=rng))
            log.append("crack")
        elif name == "deform":
            bone.update(deform(bone.mesh, cfg["shear"], cfg["twist"], cfg["flatten"],
                               cfg["compaction_axis"], rng=rng))
            log.append("deform")
        else:
            raise ValueError(f"unknown step {name!r}; choose from {', '.join(STEP_ORDER)}")
        _trim_memory()
        if max(cfg["light_texture"]) > 0:              # a little texture after every step
            bone.refine(S_res)
            for t in TEXTURE_STEPS:
                if t != name:
                    texture(t, rng.uniform(*cfg["light_texture"]), region="global")

    labels = (bone.broken > 0.5) if bone.broken is not None else np.zeros(len(bone.mesh.vertices), bool)
    before = bone.mesh.copy()
    out = _cleanup(bone.mesh)
    out.vertex_attributes["broken"] = transfer(before, out, labels).astype(np.uint8)
    out.apply_translation(offset)
    out.metadata["fossilize"] = {"steps": " -> ".join(log), **{f"n_{k}": rep.get(k, 0) for k in
                                 ("chip", "major", "lost", "ground", "cut")},
                                 "complete": round(rep.get("complete", 1.0), 3)}
    return out


def _break_step(bone, plan, cfg, spongy, target, rng):
    """Breakage. All cuts are made on the coarse mesh first; then it is refined
    once, 'ground' stumps are worn to smooth nubs, and fresh breaks get fine
    texture (and spongy pores)."""
    rep = dict(chip=0, major=0, lost=0, ground=0, cut=0, area=0.0)
    start = bone.mesh
    vol0 = abs(start.volume)
    stump = None
    if "ground" in plan:                       # cut with gentle relief: it gets worn smooth
        new, r = break_tips(start, ["ground"], cfg, rng, relief_scale=0.25)
        if r["ground"]:
            stump = fracture_mask(start, new, grow=6)
            bone.update(new)
            rep["ground"], rep["area"] = r["ground"], r["area"]
    before = bone.mesh
    new, r = break_tips(before, [k for k in plan if k != "ground"], cfg, rng)
    for k in ("chip", "major", "lost"):
        rep[k] = r[k]
    rep["area"] += r["area"]
    if new.is_watertight and abs(new.volume) > (target + 0.02) * vol0:   # still too complete
        new, rep["cut"] = fragment(new, target, vol0, cfg, rng)
    rep["complete"] = abs(new.volume) / vol0 if new.is_watertight else 1.0
    raw = None
    if new is not before:
        stump = transfer(before, new, stump)
        bone.apply_break(new)
        raw = fracture_mask(before, new, grow=0)
    if stump is None and raw is None:
        return rep
    coarse = bone.mesh
    bone.refine(cfg["resolution"])
    if stump is not None:                      # wear the ground-away stump to a nub
        stump = transfer(coarse, bone.mesh, stump)
        L = _laplacian(bone.mesh)
        for _ in range(4):
            stump = L @ stump
        stump = np.clip(stump * 1.5, 0, 1)
        m = water_smooth(bone.mesh, 60, shrink=0.0, weights=stump)
        m = abrade(m, 25, weights=stump)
        bone.update(water_smooth(m, 20, shrink=0.0, weights=stump))
        bone.mark_broken((stump > 0.3).astype(float))     # worn-away process: not original surface
    if raw is not None:                        # texture the fresh breaks
        raw = transfer(coarse, bone.mesh, raw)
        fine = bone.mesh
        rep["spongy"] = spongy
        bone.mesh = finish_fracture(fine, raw, cfg["break_relief"], spongy, cfg["cortex"], cfg["pore"], rng=rng)
        bone.frac = transfer(fine, bone.mesh, bone.frac)
        bone.broken = transfer(fine, bone.mesh, bone.broken)
    return rep


# ============================================================================
# Files and folders
# ============================================================================
MESH_EXTS = (".ply", ".stl", ".obj", ".off", ".glb", ".gltf", ".vtk", ".vtp")


def load_mesh(path):
    if os.path.splitext(path)[1].lower() not in (".vtk", ".vtp"):
        return trimesh.load(path, force="mesh")
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    r = vtk.vtkPolyDataReader() if path.lower().endswith(".vtk") else vtk.vtkXMLPolyDataReader()
    r.SetFileName(path)
    r.Update()
    tf = vtk.vtkTriangleFilter()
    tf.SetInputData(r.GetOutput())
    tf.Update()
    pd = tf.GetOutput()
    labels = pd.GetPointData().GetArray("broken")
    m = trimesh.Trimesh(vtk_to_numpy(pd.GetPoints().GetData()).astype(float),
                        vtk_to_numpy(pd.GetPolys().GetConnectivityArray()).reshape(-1, 3),
                        process=labels is None)       # keep vertex order if labels are attached
    if labels is not None:
        m.vertex_attributes["broken"] = vtk_to_numpy(labels).astype(np.uint8)
    return m


def save_mesh(mesh, path):
    if os.path.splitext(path)[1].lower() not in (".vtk", ".vtp"):
        return mesh.export(path)
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray as ids
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(mesh.vertices, dtype=np.float64), deep=True))
    cells = vtk.vtkCellArray()
    cells.SetData(ids(np.arange(0, 3 * len(mesh.faces) + 1, 3, dtype=np.int64), deep=True),
                  ids(np.ascontiguousarray(mesh.faces.ravel(), dtype=np.int64), deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    if "broken" in mesh.vertex_attributes:          # per-vertex label: 1 = broken surface
        arr = numpy_to_vtk(np.ascontiguousarray(mesh.vertex_attributes["broken"], dtype=np.uint8), deep=True)
        arr.SetName("broken")
        pd.GetPointData().AddArray(arr)
    w = vtk.vtkPolyDataWriter() if path.lower().endswith(".vtk") else vtk.vtkXMLPolyDataWriter()
    w.SetFileName(path)
    w.SetInputData(pd)
    if path.lower().endswith(".vtk"):
        w.SetFileTypeToBinary()
    w.Write()


def _process_one(job):
    src, dst, seed, steps, settings = job
    t0 = time.time()
    rec = {"input": src, "output": dst, "seed": seed}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mesh = load_mesh(src)
            out = fossilize(mesh, seed=seed, steps=steps, **settings)
            save_mesh(out, dst)
        info = dict(out.metadata["fossilize"])
        rec.update(status="ok", **info, input_watertight=mesh.is_watertight, watertight=out.is_watertight,
                   n_faces=len(out.faces), notes=" | ".join(sorted({str(w.message) for w in caught}))[:300])
    except Exception as e:  # noqa: BLE001
        rec.update(status="error", error=f"{type(e).__name__}: {e}")
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


LOG_FIELDS = ["input", "output", "seed", "status", "steps", "complete", "n_chip", "n_major", "n_lost",
              "n_ground", "n_cut", "input_watertight", "watertight", "n_faces", "seconds", "notes", "error"]


def fossilize_folder(in_dir, out_dir, n=None, variants=1, seed=None, workers=1, recursive=False,
                     fmt=None, overwrite=False, steps=None, **settings):
    """Fossilize every mesh in in_dir (or n random ones) into out_dir, with a CSV log."""
    files = sorted(os.path.join(root, f) for root, _, names in os.walk(in_dir)
                   for f in names if f.lower().endswith(MESH_EXTS)
                   if recursive or root == in_dir)
    if not files:
        raise FileNotFoundError(f"no mesh files in {in_dir}")
    seed = int(np.random.SeedSequence().entropy % 2 ** 31) if seed is None else seed
    if n is not None and n < len(files):
        files = sorted(np.random.default_rng(seed).choice(files, n, replace=False).tolist())
    jobs, skipped = [], 0
    for src in files:
        rel = os.path.relpath(src, in_dir)
        stem, ext = os.path.splitext(rel)
        ext = "." + (fmt or ext).lstrip(".")
        for k in range(variants):
            dst = os.path.join(out_dir, f"{stem}_fossil{f'_{k + 1:02d}' if variants > 1 else ''}{ext}")
            if os.path.exists(dst) and not overwrite:
                skipped += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            file_seed = int(hashlib.sha256(f"{seed}|{rel}|{k}".encode()).hexdigest()[:8], 16)
            jobs.append((src, dst, file_seed, steps, settings))
    print(f"{len(files)} file(s), {len(jobs)} output(s) to make, seed {seed}, {workers} worker(s)")
    if skipped:
        print(f"skipped {skipped} existing output(s); use --overwrite to redo them")

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "fossilize_log.csv")
    if os.path.exists(log_path) and open(log_path).readline().strip().split(",") != LOG_FIELDS:
        os.replace(log_path, log_path.replace(".csv", "_old.csv"))
    new_log, records = not os.path.exists(log_path), []
    with open(log_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if new_log:
            writer.writeheader()

        def record(rec):
            writer.writerow(rec)
            fh.flush()
            records.append(rec)
            msg = rec.get("steps", "") if rec["status"] == "ok" else rec.get("error", "")
            print(f"[{len(records)}/{len(jobs)}] {rec['status']:5s} {os.path.basename(rec['output'])} "
                  f"({rec['seconds']}s) {msg}", flush=True)

        remaining = list(jobs)
        if workers > 1:
            from concurrent.futures.process import BrokenProcessPool
            try:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futs = {ex.submit(_process_one, j): j for j in jobs}
                    for f in as_completed(futs):
                        record(f.result())
                        remaining.remove(futs[f])
            except BrokenProcessPool:
                print(f"a worker died (likely out of memory); doing the last {len(remaining)} one at a time")
        for job in remaining:                        # serial mode, or after a crash
            record(_process_one(job))
    failed = sum(r["status"] != "ok" for r in records)
    print(f"done: {len(records) - failed} written, {failed} failed. log: {log_path}")
    return records


# ============================================================================
# Command line
# ============================================================================
def _parse_setting(text):
    key, _, value = text.partition("=")
    key = key.strip().replace("-", "_")
    if key not in SETTINGS:
        raise argparse.ArgumentTypeError(f"unknown setting {key!r} (see --settings)")
    try:
        value = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        value = value.strip()
    return key, value


def _print_settings():
    src = open(__file__).read()
    block = src[src.index("SETTINGS = dict(") + len("SETTINGS = dict(\n"):src.index("\n)\n", src.index("SETTINGS = dict("))]
    print("Settings (change with --set name=value, e.g. --set p_major=0.8 --set chips=1,2):\n")
    print("\n".join(line[4:] for line in block.splitlines()))


def main():
    p = argparse.ArgumentParser(description="Turn modern vertebra meshes into fossil-like meshes.",
                                epilog="Every other option is a setting: see --settings, change with --set.")
    p.add_argument("input", nargs="?", help="mesh file or folder")
    p.add_argument("output", nargs="?", help="output file or folder")
    p.add_argument("--seed", type=int, help="random seed (same seed = same result)")
    p.add_argument("--set", action="append", default=[], type=_parse_setting, metavar="NAME=VALUE",
                   help="change a setting; repeatable")
    p.add_argument("--settings", action="store_true", help="list all settings and exit")
    p.add_argument("--steps", help="run exactly these effects, e.g. break,roughen (default: random)")
    g = p.add_argument_group("folder mode")
    g.add_argument("-n", type=int, help="only process N randomly chosen files")
    g.add_argument("--variants", type=int, default=1, help="fossil versions per file")
    g.add_argument("--workers", type=int, default=1, help="files in parallel (~1-3 GB memory each)")
    g.add_argument("--format", help="output file type, e.g. ply or vtk (default: same as input)")
    g.add_argument("--recursive", action="store_true", help="include subfolders")
    g.add_argument("--overwrite", action="store_true", help="redo outputs that already exist")
    a = p.parse_args()

    if a.settings:
        return _print_settings()
    if not (a.input and a.output):
        p.error("input and output are required")
    settings = dict(a.set)
    steps = [s.strip() for s in a.steps.split(",")] if a.steps else None
    if os.path.isdir(a.input):
        recs = fossilize_folder(a.input, a.output, a.n, a.variants, a.seed, a.workers, a.recursive,
                                a.format, a.overwrite, steps, **settings)
        sys.exit(any(r["status"] != "ok" for r in recs))
    out = fossilize(load_mesh(a.input), seed=a.seed, steps=steps, **settings)
    save_mesh(out, a.output)
    print(f"{out.metadata['fossilize']['steps']}\nwrote {a.output}: {len(out.faces):,} faces, "
          f"watertight={out.is_watertight}")


if __name__ == "__main__":
    main()