#!/usr/bin/env python3
"""
prep_nsm.py: prepare fossil meshes for NSM shape completion, automating the
3D Slicer Segment Editor steps, including cutting away broken surface.

    python preprocess_fossilized.py fossil.vtk fossil_hollow_cut.vtk
    python preprocess_fossilized.py fossilized/ fossilized_preprocessed/ --workers 8           # a whole folder
    python preprocess_fossilized.py fossil.vtk out.vtk --intact modern.vtk        # fossil saved without labels

Steps, in voxels (defaults match the manual workflow):
    labelmap -> grow margin 3 -> smooth 6 -> fill holes 6 -> hollow (medial
    surface, shell 3) -> cut away the shell over broken surface (thin cracks
    are left) -> drop specks under 1000 voxels -> surface mesh

Which surface is broken comes from the 'broken' label fossilize.py saves with
each output. For fossils made before that existed, pass --intact (the modern
mesh it was made from): it is fitted onto the fossil (allowing shear and
squashing) and surface far from it is treated as broken.

Requires: numpy scipy trimesh scikit-image (vtk for .vtk), and fossilize.py
in the same folder.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

from fossilize import MESH_EXTS, load_mesh, save_mesh

# all sizes in voxels unless noted
DEFAULTS = dict(
    voxels=220,        # voxels across the bone's longest side (sets the voxel size)
    margin=3,          # grow margin
    smooth=6,          # median smoothing kernel
    fill=6,            # fill holes (closing) kernel
    shell=3,           # hollow: shell thickness around the medial surface
    cut_margin=2,      # cut this much further into intact surface around breaks
    min_width=8,       # broken patches narrower than this (cracks, seams) are not cut
    min_island=1000,   # keep every piece of shell at least this big, not just the largest
    tol=0.015,         # --intact only: surface farther than this (x bone size) is broken
)


# ----------------------------------------------------------------------------
# Broken-surface labels
# ----------------------------------------------------------------------------
def labels_from_intact(fossil, intact, tol=0.015, max_angle=45, iters=40):
    """Label fossil vertices broken where they are far from the intact mesh, or where
    they face a different way than the intact surface there (fracture faces cut
    across the original surface, even through thin parts where they stay close to
    it). The intact mesh is first fitted onto the fossil with a trimmed affine ICP,
    which absorbs shear and flattening."""
    S = fossil.scale
    fv, fn = np.array(fossil.vertices), np.array(fossil.vertex_normals)
    src, fi = trimesh.sample.sample_surface_even(intact, 200000, seed=0)
    src_n = intact.face_normals[fi]
    A, t = np.eye(3), fossil.centroid - intact.centroid      # intact -> fossil
    q = fv[np.random.default_rng(0).choice(len(fv), min(len(fv), 30000), replace=False)]
    for it in range(iters):
        d, j = cKDTree(src @ A.T + t).query(q)
        keep = d <= np.quantile(d, 0.7 if it < iters // 2 else 0.85)   # ignore breaks
        sol, *_ = np.linalg.lstsq(np.c_[src[j[keep]], np.ones(keep.sum())], q[keep], rcond=None)
        A, t = sol[:3].T, sol[3]
    d, j = cKDTree(src @ A.T + t).query(fv)
    n_int = src_n[j] @ np.linalg.inv(A)                       # normals under the affine map
    n_int /= np.linalg.norm(n_int, axis=1, keepdims=True)
    facing = np.einsum("ij,ij->i", fn, n_int)
    broken = (d > tol * S) | (facing < np.cos(np.radians(max_angle)))
    # tidy: drop specks (texture can tilt a few normals), grow a little over the rims
    L = trimesh.smoothing.laplacian_calculation(fossil, equal_weight=True)
    b = broken.astype(float)
    for _ in range(4):
        b = L @ b
    broken = b > 0.5
    for _ in range(3):
        broken = (L @ broken.astype(float)) > 0
    return broken.astype(np.uint8)


# ----------------------------------------------------------------------------
# The Segment Editor steps
# ----------------------------------------------------------------------------
def _ball_close(occ, r):
    """Morphological closing with a ball of radius r voxels (fills holes/gaps)."""
    dil = ndimage.distance_transform_edt(~occ) <= r
    return ndimage.distance_transform_edt(dil) > r


def prepare(mesh, broken=None, voxels=220, margin=3, smooth=6, fill=6, shell=3, cut_margin=2,
            min_width=8, min_island=1000, **_):
    """Run the preprocessing on one mesh. `broken` is a per-vertex 0/1 label.
    Returns (hollow_cut mesh, info dict)."""
    pitch = mesh.extents.max() / voxels
    pad = margin + smooth + fill + shell + 4
    vox = mesh.voxelized(pitch).fill()
    occ = np.pad(vox.matrix, pad)
    origin = vox.transform[:3, 3] - pad * pitch               # world position of voxel (0,0,0)

    occ = ndimage.distance_transform_edt(~occ) <= margin      # grow margin
    k = smooth + (smooth % 2 == 0)                            # median of a binary = majority
    occ = ndimage.uniform_filter(occ.astype(np.float32), size=k) > 0.5
    occ = _ball_close(occ, fill / 2)                          # fill holes
    d_in = ndimage.distance_transform_edt(occ)                # hollow around the surface
    d_out = ndimage.distance_transform_edt(~occ)
    inner = shell // 2 + shell % 2
    hollow = (occ & (d_in <= inner)) | (~occ & (d_out <= shell - inner))
    n_hollow = int(hollow.sum())

    n_cut = 0
    if broken is not None and np.any(broken):
        broken = np.asarray(broken).astype(bool)
        idx = np.argwhere(hollow)
        pts = idx * pitch + origin
        v = np.array(mesh.vertices)
        db = cKDTree(v[broken]).query(pts)[0] / pitch
        di = cKDTree(v[~broken]).query(pts)[0] / pitch if (~broken).any() else np.full(len(pts), np.inf)
        cut = np.zeros_like(hollow)
        cut[tuple(idx[db < di].T)] = True                     # nearest surface is broken
        if min_width > 0:                                     # thin lines (cracks, seams) stay: only
            r = min_width / 2                                 # cut where the broken patch is wide
            to_intact = ndimage.distance_transform_edt(~(hollow & ~cut))
            core = cut & (to_intact > r)
            cut &= ndimage.distance_transform_edt(~core) <= r + 1
        cut = hollow & (ndimage.distance_transform_edt(~cut) <= cut_margin)   # then reach past the rim
        hollow &= ~cut
        n_cut = int(cut.sum())

    lab, n = ndimage.label(hollow, structure=np.ones((3, 3, 3)))
    if n == 0:
        raise ValueError("nothing left after cutting")
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.isin(lab, 1 + np.flatnonzero(sizes >= min(min_island, sizes.max())))  # drop specks only

    from skimage.measure import marching_cubes
    field = ndimage.gaussian_filter(keep.astype(np.float32), 0.6)   # gentle, like Slicer's surface smoothing
    verts, faces, _, _ = marching_cubes(field, 0.5)
    out = trimesh.Trimesh(verts * pitch + origin, faces, process=True)
    trimesh.repair.fix_normals(out)
    info = dict(voxel_size=pitch, shell_voxels=n_hollow, cut_voxels=n_cut,
                cut_pct=round(100 * n_cut / max(n_hollow, 1), 1), islands=n,
                kept_pct=round(100 * sizes.max() / max(sizes.sum(), 1), 1))
    return out, info


# ----------------------------------------------------------------------------
# Files, folders, command line
# ----------------------------------------------------------------------------
def _one(job):
    src, dst, intact, opts = job
    t0 = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mesh = load_mesh(src)
            if intact:
                broken = labels_from_intact(mesh, load_mesh(intact), opts["tol"])
            else:
                broken = mesh.vertex_attributes.get("broken")
            out, info = prepare(mesh, broken, **opts)
            save_mesh(out, dst)
        note = "" if broken is not None else "  (no 'broken' label: nothing cut)"
        return f"ok    {os.path.basename(dst)} ({time.time() - t0:.0f}s) cut {info['cut_pct']}% of shell, " \
               f"voxel {info['voxel_size']:.4g}{note}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR {os.path.basename(src)}: {type(e).__name__}: {e}"


def main():
    p = argparse.ArgumentParser(description="Prepare fossil meshes for NSM: hollow shell with broken "
                                            "surface cut away (automates the Slicer steps).")
    p.add_argument("input", help="fossil mesh, or folder of them")
    p.add_argument("output", help="output mesh, or output folder")
    p.add_argument("--intact", help="intact mesh (or folder of them, matched by name) for fossils "
                                    "saved without a 'broken' label")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v, help=f"default {v}")
    p.add_argument("--workers", type=int, default=1, help="files in parallel")
    p.add_argument("--suffix", default="_hollow_cut", help="added to output names in folder mode")
    a = p.parse_args()
    opts = {k: getattr(a, k) for k in DEFAULTS}

    if not os.path.isdir(a.input):
        print(_one((a.input, a.output, a.intact, opts)))
        return
    os.makedirs(a.output, exist_ok=True)
    jobs = []
    for f in sorted(os.listdir(a.input)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in MESH_EXTS:
            continue
        intact = a.intact
        if intact and os.path.isdir(intact):            # match fossil "name_fossil..." to intact "name.*"
            base = stem.split("_fossil")[0]
            hits = [g for g in os.listdir(intact) if os.path.splitext(g)[0] == base]
            if not hits:
                print(f"skip  {f}: no intact mesh named {base}.* in {intact}")
                continue
            intact = os.path.join(intact, hits[0])
        jobs.append((os.path.join(a.input, f), os.path.join(a.output, stem + a.suffix + ext), intact, opts))
    print(f"{len(jobs)} file(s), {a.workers} worker(s)")
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            for f in as_completed([ex.submit(_one, j) for j in jobs]):
                print(f.result(), flush=True)
    else:
        for j in jobs:
            print(_one(j), flush=True)


if __name__ == "__main__":
    main()