#!/usr/bin/env python3
"""
Descriptive stats: % broken vs. completion error, per MorphoWeave segment.

Uses the same sampling, intact mask, Chamfer and L1 definitions as
NSM.evaluation.region_errors, but splits the error by segment instead of intact/filled.
Place in nsm/paper_fossils/ and run from there, after shape_completion_eval.py.
Relative paths (--csv, --label_dir, completed_path in the CSV) resolve against nsm/;
pass --root if the repo root is somewhere else.

Segment labels per GT mesh, from --seg_dir, matched by GT file stem:
  <stem>_seg_00.ply ... <stem>_seg_06.ply   MorphoWeave output, one submesh per segment (default)
  <stem>.npy / .txt / .csv   one integer label per GT vertex (same order as the GT .vtk)
  <stem>.vtk / .ply / .vtp   a mesh carrying a per-point label array (--seg_array, default "seg");
                             labels are transferred to GT samples by nearest point, so this mesh
                             does not need the same vertices as the GT, only the same frame.

Example:
  cd nsm/paper_fossils
  python segment_region_stats.py \
    --csv run_v72/shape_completion/fine_tuning_encoder_fossilized_preprocessed_snap_nobroken/train/best_cfg_predictions/inference_summary.csv \
    --seg_dir path/to/morphoweave_segs
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

# Lives in nsm/paper_fossils/; NSM/ and the eval outputs are one level up in nsm/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from NSM.evaluation import uniform_surface_sample, gt_intact_mask


def seg_lookup(gt_path, gt_mesh, seg_dir, seg_array, align_tol=0.02):
    """Return (points, labels) used to label GT surface samples by nearest point."""
    stem = os.path.splitext(os.path.basename(gt_path))[0]

    # MorphoWeave output: one submesh per segment, <stem>_seg_00.ply ... <stem>_seg_06.ply
    seg_files = sorted(glob.glob(os.path.join(seg_dir, f"{stem}_seg_*.ply")))
    if seg_files:
        gt_pts = np.asarray(gt_mesh.points)
        size = np.linalg.norm(np.ptp(gt_pts, axis=0))
        meshes, labels = [], []
        for f in seg_files:
            m = pv.read(f)
            meshes.append(m)
            labels.append(np.full(m.n_points, int(re.search(r"_seg_(\d+)\.ply$", f).group(1))))
        lab = np.concatenate(labels)
        merged = pv.merge(meshes, merge_points=False) if len(meshes) > 1 else meshes[0]
        pts = np.asarray(merged.points)

        # Segments must sit on the GT surface. If they don't (different normalisation or pose),
        # match centroid + scale, then refine with rigid ICP.
        gap = np.median(cKDTree(gt_pts).query(pts)[0]) / size
        scale_ratio = np.linalg.norm(np.ptp(pts, axis=0)) / size
        centre_off = np.linalg.norm(pts.mean(0) - gt_pts.mean(0)) / size
        if gap > align_tol or abs(scale_ratio - 1) > 0.1 or centre_off > 0.1:
            s = size / np.linalg.norm(np.ptp(pts, axis=0))
            pts = (pts - pts.mean(0)) * s + gt_pts.mean(0)
            moved = pv.PolyData(pts).align(pv.PolyData(gt_pts), max_iterations=200)
            pts = np.asarray(moved.points)
            new_gap = np.median(cKDTree(gt_pts).query(pts)[0]) / size
            print(f"  {stem}: segments off GT by {gap:.3f} of size; after alignment {new_gap:.3f}")
            if new_gap > align_tol:
                raise ValueError(f"segments for {stem} could not be aligned to GT (gap {new_gap:.3f})")
        return pts, lab

    for ext in (".npy", ".txt", ".csv"):
        p = os.path.join(seg_dir, stem + ext)
        if os.path.exists(p):
            lab = np.load(p) if ext == ".npy" else np.loadtxt(p, delimiter="," if ext == ".csv" else None)
            lab = np.asarray(lab).astype(int).ravel()
            pts = np.asarray(gt_mesh.points)
            if len(lab) != len(pts):
                raise ValueError(f"{p}: {len(lab)} labels vs {len(pts)} GT vertices")
            return pts, lab
    for ext in (".vtk", ".vtp", ".ply"):
        p = os.path.join(seg_dir, stem + ext)
        if os.path.exists(p):
            m = pv.read(p)
            return np.asarray(m.points), np.asarray(m.point_data[seg_array]).astype(int).ravel()
    raise FileNotFoundError(f"No segment file for {stem} in {seg_dir}")


def resolve(path, root):
    """Relative paths in the CSV / defaults are relative to the repo root, not the cwd."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(root, path)


def per_mesh(row, args):
    label_path = os.path.join(resolve(args.label_dir, args.root), row.mesh.replace("_hollow_cut", "").replace("_align", ""))
    gt_raw = pv.read(resolve(row.gt_path, args.root))
    seg_pts, seg_lab = seg_lookup(row.gt_path, gt_raw, args.seg_dir, args.seg_array)

    # --- identical to NSM.evaluation.region_errors ---
    gt = gt_raw.extract_surface(algorithm=None).triangulate().compute_normals(point_normals=True, auto_orient_normals=True)
    compl = pv.read(resolve(row.completed_path, args.root)).extract_surface(algorithm=None).triangulate()
    sg = np.asarray(uniform_surface_sample(gt, args.n_samples))
    sp = np.asarray(uniform_surface_sample(compl, args.n_samples))
    intact = gt_intact_mask(sg, label_path)
    d_gc = cKDTree(sp).query(sg)[0]                 # GT -> completion
    d_cg, j = cKDTree(sg).query(sp)                 # completion -> GT; region of nearest GT sample
    normals = gt.point_normals[cKDTree(np.asarray(gt.points)).query(sg)[1]]
    size = np.linalg.norm(np.ptp(sg, axis=0))
    rng = np.random.default_rng(0)
    probe = pv.PolyData(sg + normals * rng.normal(0, 0.01 * size, (len(sg), 1)))
    l1 = np.abs(np.asarray(probe.compute_implicit_distance(gt)["implicit_distance"])
                - np.asarray(probe.compute_implicit_distance(compl)["implicit_distance"]))
    # --------------------------------------------------

    seg_g = seg_lab[cKDTree(seg_pts).query(sg)[1]]  # segment of each GT sample
    seg_c = seg_g[j]                                # segment of each completion sample

    out = []
    for s in np.unique(seg_lab):
        g, c = seg_g == s, seg_c == s
        gb = g & ~intact
        out.append(dict(
            mesh=row.mesh, seg=int(s),
            area_frac=g.mean(),                      # share of the vertebra's surface in this segment
            pct_broken=100 * (1 - intact[g].mean()) if g.any() else np.nan,
            chamfer_seg=0.5 * (d_gc[g].mean() + d_cg[c].mean()) if g.any() and c.any() else np.nan,
            l1_seg=l1[g].mean() if g.any() else np.nan,
            l1_seg_broken_part=l1[gb].mean() if gb.any() else np.nan,
            chamfer_overall=row.chamfer,
            l1_overall=l1.mean(),
        ))
    return out


def rho(x, y):
    ok = x.notna() & y.notna()
    if ok.sum() < 5 or x[ok].nunique() < 2:
        return np.nan, np.nan
    r = spearmanr(x[ok], y[ok])
    return r.correlation, r.pvalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seg_dir", required=True)
    ap.add_argument("--seg_array", default="seg")
    ap.add_argument("--label_dir", default="fossilized", help="same as LABEL_DIR in shape_completion_eval.py")
    ap.add_argument("--root", default=REPO_ROOT, help="repo root (nsm/) that relative paths resolve against")
    ap.add_argument("--n_samples", type=int, default=20000)
    ap.add_argument("--out_dir", default=None, help="default: next to the CSV")
    args = ap.parse_args()
    args.csv = resolve(args.csv, args.root)
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.csv), "segment_stats")
    os.makedirs(out_dir, exist_ok=True)

    summ = pd.read_csv(args.csv).dropna(subset=["chamfer"])
    rows = []
    for row in summ.itertuples():
        try:
            rows += per_mesh(row, args)
            print("ok ", row.mesh)
        except Exception as e:
            print("ERR", row.mesh, e)
    long = pd.DataFrame(rows)
    long.to_csv(os.path.join(out_dir, "segment_per_mesh.csv"), index=False)

    stats = []
    for s, d in long.groupby("seg"):
        st = dict(
            seg=s,
            n_meshes=len(d),
            area_frac=d.area_frac.mean(),
            n_any_broken=int((d.pct_broken > 0).sum()),
            n_over_50pct_broken=int((d.pct_broken > 50).sum()),
            mean_pct_broken=d.pct_broken.mean(),
            median_pct_broken=d.pct_broken.median(),
            median_chamfer_seg=d.chamfer_seg.median(),
            median_l1_seg=d.l1_seg.median(),
            median_l1_on_broken_part=d.l1_seg_broken_part.median(),
        )
        for name, y in [("chamfer_seg", d.chamfer_seg), ("l1_seg", d.l1_seg),
                        ("chamfer_overall", d.chamfer_overall), ("l1_overall", d.l1_overall)]:
            st[f"rho_{name}"], st[f"p_{name}"] = rho(d.pct_broken, y)
        stats.append(st)
    stats = pd.DataFrame(stats).set_index("seg")
    stats.to_csv(os.path.join(out_dir, "segment_summary.csv"))

    pd.set_option("display.width", 250, "display.max_columns", 50)
    print("\nHow often each segment is broken")
    print(stats[["n_meshes", "area_frac", "n_any_broken", "n_over_50pct_broken",
                 "mean_pct_broken", "median_pct_broken"]].round(3).to_string())
    print("\nError per segment (medians across meshes)")
    print(stats[["median_chamfer_seg", "median_l1_seg", "median_l1_on_broken_part"]].round(4).to_string())
    print("\nSpearman: % broken in segment vs ...")
    print(stats[[c for c in stats.columns if c.startswith(("rho_", "p_"))]].round(3).to_string())
    print(f"\nWrote {out_dir}/segment_per_mesh.csv and segment_summary.csv")


if __name__ == "__main__":
    main()