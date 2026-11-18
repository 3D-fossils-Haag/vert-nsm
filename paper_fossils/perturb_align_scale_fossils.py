#!/usr/bin/env python3
"""
perturb_align_scale_fossils.py: test landmark alignment of fossils before NSM.

python perturb_align_scale_fossils.py perturb fossilized/ perturbed/
python perturb_align_scale_fossils.py perturb fossilized_preprocessed/ perturbed_preprocessed/

python perturb_align_scale_fossils.py sufficiency sufficiency.parquet

perturb      Keep only landmarks on surviving surface, add placement error, then move each
             fossil (mesh + landmarks) by a random rotation, scale and shift, like a new scan.
             Broken labels are read from BROKEN/<fossil name>.seg.nrrd and stored on the mesh.
             Writes orig/models/, orig/lms/ and perturb_log.csv; run align_model_to_ssm.py next.
sufficiency  Alignment error vs number and position of landmarks, on the intact meshes.

Paths and settings are below. Errors are relative to where the mesh already is (the
training meshes are in the atlas frame): RMS displacement and scale error as % of bone
size, rotation error in degrees.
"""
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from NSM.alignment import load_mrk_json, save_mrk_json, align_to_template, rms_err_pct
from NSM.fossils import load_broken_seg

# ----------------------------------------------------------------------------
# Paths and settings
# ----------------------------------------------------------------------------
LANDMARKS = "alignedLMs"                          # TO DO: '<name>.mrk.json' on the intact meshes
INTACT = "vertebrae_meshes"                       # TO DO: intact meshes (landmarks follow fossil deformation)
ATLAS = "atlas_db/sparse_landmarks.mrk.json"      # TO DO: atlas landmarks
BROKEN = "fossilized_broken"                      # TO DO: '<fossil name>.seg.nrrd' broken segmentations

KEEP = (3, 12)          # landmarks used per fossil: random number in this range (None = all usable)
ROTATE = 180            # max rotation, degrees
SCALE = (0.5, 2.0)      # scale range
TRANSLATE = 1.0         # max shift, x bone size
JITTER = 0.01           # landmark placement error, x bone size
TOL = 0.01              # a landmark must be this close to the fossil surface (x bone size)
GUARD = 0.02            # ... and this far from broken surface (x bone size)
SUBSETS = 50            # sufficiency: random subsets per landmark count above 3
GOOD = 2.0              # sufficiency: alignment error counted as good, % of bone size
SEED = 0

# For sufficiency experiment
TRIPLETS = 300          # sufficiency: random triplets per specimen (None = all)

# Helper funcs
def random_rotation(rng, max_deg):
    """Random rotation matrix about a random axis, and its angle in degrees."""
    axis = rng.normal(size=3)
    a = np.radians(rng.uniform(0, max_deg))
    return Rotation.from_rotvec(axis / np.linalg.norm(axis) * a).as_matrix(), float(np.degrees(a))

def alignment_error(verts, moved, size, s, R):
    """How far moved (verts after a similarity with scale s and rotation R) is from verts."""
    return dict(rms_err_pct=rms_err_pct(moved, verts, size),
                rot_err_deg=float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),   # rotation angle
                scale_err_pct=100 * (s - 1))

def fit_affine(src, dst, iters=30):
    """Trimmed affine ICP src -> dst, absorbing shear/flattening; missing parts are ignored."""
    tree = cKDTree(dst)
    A, t = np.eye(3), dst.mean(0) - src.mean(0)
    for _ in range(iters):
        d, j = tree.query(src @ A.T + t)
        keep = d <= np.quantile(d, 0.6)
        sol = np.linalg.lstsq(np.c_[src[keep], np.ones(keep.sum())], dst[j[keep]], rcond=None)[0]
        A, t = sol[:3].T, sol[3]
    return A, t

def usable_landmarks(L, fv, broken, size, intact_pts, tol, guard):
    """Landmarks moved onto the fossil surface (following its deformation when the intact mesh
    is given), and which sit on surviving, unbroken surface away from break edges."""
    if intact_pts is not None:
        A, t = fit_affine(intact_pts, fv)
        L = L @ A.T + t
    d, j = cKDTree(fv).query(L)
    near_break = (cKDTree(fv[broken]).query(fv[j])[0] < guard * size) if broken.any() else np.zeros(len(L), bool)
    return fv[j], (d < tol * size) & ~broken[j] & ~near_break

# ----------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------
def perturb(fossils, out_dir):
    rng = np.random.default_rng(SEED)
    atlas = load_mrk_json(ATLAS)[0]
    lms = {f.lower(): os.path.join(LANDMARKS, f) for f in os.listdir(LANDMARKS)}
    intact = {os.path.splitext(f)[0].lower(): os.path.join(INTACT, f) for f in os.listdir(INTACT)} if os.path.isdir(INTACT) else {}
    os.makedirs(os.path.join(out_dir, "orig", "models"), exist_ok=True)
    rows, missing = [], []
    for path in sorted(glob.glob(os.path.join(fossils, "*.vtk"))):
        fname = os.path.basename(path)
        stem = os.path.splitext(fname)[0]
        name = stem.split("_fossil")[0]
        lm_path = lms.get((name + ".mrk.json").lower())
        if not lm_path:
            print(f"skip  {fname}: no {name}.mrk.json in {LANDMARKS}")
            continue
        mesh = pv.read(path)
        fv = np.asarray(mesh.points, float)
        seg_path = os.path.join(BROKEN, stem + ".seg.nrrd")
        if os.path.exists(seg_path):
            broken = load_broken_seg(seg_path, fv).astype(bool)
        else:
            print(f"\033[33mwarn  {fname}: no {seg_path}, all surface treated as intact\033[0m")
            missing.append(dict(fossil=fname, seg=seg_path))
            broken = np.zeros(len(fv), bool)
        size = np.linalg.norm(np.ptp(fv, axis=0))
        intact_path = intact.get(name.lower())
        intact_pts = None
        if intact_path:
            iv = np.asarray(pv.read(intact_path).points, float)
            intact_pts = iv[rng.choice(len(iv), min(len(iv), 30000), replace=False)]
        L, labels = load_mrk_json(lm_path)
        if len(atlas) != len(L):
            raise SystemExit(f"atlas has {len(atlas)} landmarks, {lm_path} has {len(L)}")
        P, ok = usable_landmarks(L, fv, broken, size, intact_pts, TOL, GUARD)
        idx = np.flatnonzero(ok)
        if KEEP:
            k = rng.integers(KEEP[0], KEEP[1] + 1)
            idx = np.sort(rng.choice(idx, min(k, len(idx)), replace=False))
        row = dict(fossil=fname, n_landmarks=len(L), n_intact=int(ok.sum()), n_used=len(idx),
                   used=" ".join(labels[i] for i in idx), followed_deformation=intact_pts is not None)
        if len(idx) < 3:
            print(f"skip  {fname}: only {len(idx)} landmark(s) on intact surface, alignment needs 3")
            rows.append(dict(row, note="fewer than 3 usable landmarks"))
            continue

        tree = cKDTree(fv)
        Pj = fv[tree.query(P[idx] + rng.normal(0, JITTER * size, (len(idx), 3)))[1]]   # placement error
        R, rot_deg = random_rotation(rng, ROTATE)
        s = float(np.exp(rng.uniform(*np.log(SCALE))))
        t = rng.normal(size=3)
        t *= rng.uniform(0, TRANSLATE * size) / np.linalg.norm(t)
        c = fv.mean(0)
        # Random scan pose: scale and rotate about the centroid c, then shift by t
        out = mesh.copy()
        fv_posed = s * (fv - c) @ R.T + c + t
        out.points = fv_posed
        out.point_data["broken"] = broken.astype(np.uint8)        # labels travel with the mesh from here on
        Pj_posed = s * (Pj - c) @ R.T + c + t
        if "Normals" in out.point_data:
            out.point_data.remove("Normals")
        out.save(os.path.join(out_dir, "orig", "models", fname))
        save_mrk_json(Pj_posed, os.path.join(out_dir, "orig", "lms", stem.split("_hollow")[0] + ".mrk.json"),
                      lm_path, idx)
        row.update(scale=s, rot_deg=rot_deg, shift_pct=100 * np.linalg.norm(t) / size)

        sa, Ra, ta = align_to_template(atlas[idx], Pj_posed)        # what align_model_to_ssm.py will do
        aligned = sa * fv_posed @ Ra + ta                            # posed fossil after alignment
        row.update(alignment_error(fv, aligned, size, s * sa, R.T @ Ra))
        rows.append(row)
        print(f"ok    {fname}: {len(idx)}/{len(L)} landmarks, alignment error {row['rms_err_pct']:.2f}% of size")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "perturb_log.csv"), index=False)
    if missing:
        pd.DataFrame(missing).to_csv(os.path.join(out_dir, "missing_segs.csv"), index=False)

def sufficiency(out_path):
    rng = np.random.default_rng(SEED)
    atlas = load_mrk_json(ATLAS)[0]
    intact = {os.path.splitext(f)[0].lower(): os.path.join(INTACT, f) for f in os.listdir(INTACT)}
    rows = []
    for lm_path in sorted(glob.glob(os.path.join(LANDMARKS, "*.mrk.json"))):
        name = os.path.basename(lm_path)[:-len(".mrk.json")]
        mesh_path = intact.get(name.lower())
        L, labels = load_mrk_json(lm_path)
        if not mesh_path or len(L) != len(atlas):
            print(f"skip  {name}: " + ("no mesh" if not mesh_path else f"{len(L)} landmarks, atlas has {len(atlas)}"))
            continue
        v = np.asarray(pv.read(mesh_path).points, float)
        size = np.linalg.norm(np.ptp(v, axis=0))
        tree = cKDTree(v)
        verts = v[rng.choice(len(v), min(len(v), 3000), replace=False)]
        N = len(L)
        subsets = [np.array(c) for c in itertools.combinations(range(N), 3)]
        if TRIPLETS:
            subsets = [subsets[i] for i in rng.choice(len(subsets), min(TRIPLETS, len(subsets)), replace=False)]
        subsets += [np.sort(rng.choice(N, k, replace=False)) for k in range(4, N) for _ in range(SUBSETS)]
        subsets += [np.arange(N), np.arange(N)]                     # all landmarks, without then with jitter
        for n_sub, idx in enumerate(subsets):
            jitter = 0.0 if n_sub == len(subsets) - 2 else JITTER
            P = v[tree.query(L[idx] + rng.normal(0, jitter * size, (len(idx), 3)))[1]]
            sv = np.linalg.svd(P - P.mean(0), compute_uv=False) / size
            s, R, t = align_to_template(atlas[idx], P)
            rows.append(dict(specimen=name, k=len(idx), jitter=jitter,
                             landmarks=" ".join(labels[i] for i in idx) if len(idx) == 3 else "", spread2=sv[1],
                             **alignment_error(verts, s * verts @ R + t, size, s, R)))
        print(f"ok    {name}: {len(subsets)} landmark subsets")
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)

    noisy = df[df.jitter > 0]
    good = (noisy.rms_err_pct < GOOD).groupby(noisy.k).mean()
    summary = noisy.groupby("k").rms_err_pct.describe(percentiles=[0.5, 0.9])[["50%", "90%"]]
    summary[f"share under {GOOD}%"] = good
    print(f"\nAlignment error (RMS, % of bone size) by number of landmarks, {JITTER:.1%} placement error:")
    print(summary.round(2).to_string())
    print(f"floor (all landmarks, no placement error): median {df[df.jitter == 0].rms_err_pct.median():.2f}%")
    tri = noisy[noisy.k == 3]
    print("\nTriplets by how spread out they are (2nd spread axis, quartiles), median error %:")
    print(tri.groupby(pd.qcut(tri.spread2, 4), observed=True).rms_err_pct.median().round(2).to_string())
    print("\nBest triplets across specimens (median error %):")
    print(tri.groupby("landmarks").rms_err_pct.median().sort_values().head(10).round(2).to_string())

if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "perturb":
        perturb(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "sufficiency":
        sufficiency(sys.argv[2])
    else:
        sys.exit(__doc__)