#!/usr/bin/env python3
"""
shape_metrics.py: separate size/placement error from shape error in shape completion runs.

    python check_shape_metrics.py run_v72/shape_completion                 # every inference_summary_*.csv in it
    python check_shape_metrics.py <run>/best_cfg_predictions/inference_summary.csv [more ...]

For each completed mesh vs its ground truth (paths from inference_summary.csv):
  chamfer        NSM.evaluation.chamfer_distance, as in the inference summaries (same units)
  scale          similarity scale that best fits the completion onto the GT (>1 = completion too big)
  (the three chamfer columns use the same function, so they compare directly)
  signed_dist    mean signed distance of completion points from the GT surface (>0 = outside, inflated)
  chamfer_rigid  after a rigid fit (rotation + shift): placement removed, size kept
  chamfer_sim    after a similarity fit (+ scale): placement and size removed, shape only
  fscore_1pct    share of points within 1% of bone size, both ways (F-score; less forgiving of blur)
                 signed_dist and fscore_1pct are point-to-surface, in % of GT bone size
Writes shape_metrics_<run>.csv next to each summary (runs that already have one are skipped;
delete it to redo), and shape_metrics_medians.csv with one row per run. Run from the repo root.
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree

from NSM.evaluation import chamfer_distance, uniform_surface_sample

N = 20000
SEED = 0
PATH_FIX = ("/home/k.wolcott/", "/home/kaubricot/Downloads/")   # TO DO: (server prefix, local prefix) for old summaries

def local(path):
    """The path as given, or with PATH_FIX applied if it doesn't exist here."""
    if isinstance(path, str) and not os.path.exists(path) and PATH_FIX and path.startswith(PATH_FIX[0]):
        return PATH_FIX[1] + path[len(PATH_FIX[0]):]
    return path

def to_surface(pts, mesh):
    """Signed distance from each point to a closed surface (positive = outside). The raw sign
    depends on which way the mesh's triangles face, so it is checked at a point far outside."""
    far = np.asarray(mesh.center) + 10 * np.linalg.norm(np.ptp(mesh.points, axis=0))
    d = pv.PolyData(np.vstack([pts, far])).compute_implicit_distance(mesh)["implicit_distance"]
    return np.asarray(d[:-1]) * np.sign(d[-1])

def fit(src_pts, mesh, dst, scaling, iters=40):
    """ICP of a completion onto the GT (rigid, or similarity with scaling), fitted on sample
    points. Returns (the completion mesh moved by the fit, size of the completion relative to
    the GT: >1 = completion too big)."""
    tree, x, v, s_tot = cKDTree(dst), src_pts.copy(), np.asarray(mesh.points, float).copy(), 1.0
    for _ in range(iters):
        y = dst[tree.query(x)[1]]
        mx, my = x.mean(0), y.mean(0)
        U, S, Vt = np.linalg.svd((x - mx).T @ (y - my))
        d = np.sign(np.linalg.det(U @ Vt))
        U[:, -1] *= d
        R = U @ Vt
        s = (S[:-1].sum() + d * S[-1]) / ((x - mx) ** 2).sum() if scaling else 1.0
        x, v = s * (x - mx) @ R + my, s * (v - mx) @ R + my
        s_tot *= s
    out = mesh.copy()
    out.points = v
    return out, 1.0 / s_tot

def score(compl_path, gt_path):
    compl = pv.read(compl_path).triangulate().extract_surface(algorithm=None)
    gt = pv.read(gt_path).triangulate().extract_surface(algorithm=None)
    a, b = uniform_surface_sample(compl, N), uniform_surface_sample(gt, N)   # for the fits and the F-score
    size = np.linalg.norm(np.ptp(b, axis=0))

    d_ab = to_surface(a, gt)
    rigid, _ = fit(a, compl, b, scaling=False)
    sim, scale = fit(a, compl, b, scaling=True)
    thr = 0.01 * size
    prec = (np.abs(d_ab) < thr).mean()
    rec = (np.abs(to_surface(b, compl)) < thr).mean()
    return dict(chamfer=chamfer_distance(compl, gt, n_samples=N),
                chamfer_rigid=chamfer_distance(rigid, gt, n_samples=N),
                chamfer_sim=chamfer_distance(sim, gt, n_samples=N),
                scale=scale,
                signed_dist=100 * d_ab.mean() / size,
                fscore_1pct=2 * prec * rec / max(prec + rec, 1e-12))

def find_summaries(args):
    """Expand folders to their inference_summary_*.csv files; keep files as given."""
    out = []
    for a in args:
        out += sorted(glob.glob(os.path.join(a, "inference_summary_*.csv"))) if os.path.isdir(a) else [a]
    return out

def names(path):
    """(run label, output csv) for a summary file."""
    base = os.path.basename(path)
    if base.startswith("inference_summary_"):                 # shape_completion/inference_summary_<run>.csv
        run = base[len("inference_summary_"):-len(".csv")]
        return run, os.path.join(os.path.dirname(path), f"shape_metrics_{run}.csv")
    run = path.split("shape_completion/")[-1].split("/best_cfg_predictions")[0]
    return run, os.path.join(os.path.dirname(path), "shape_metrics.csv")

def main(args):
    summaries = find_summaries(args)
    if not summaries:
        sys.exit(f"no inference_summary_*.csv found in {' '.join(args)}")
    print(f"{len(summaries)} run(s)")
    medians = {}
    for i, path in enumerate(summaries, 1):
        run, out_path = names(path)
        if os.path.exists(out_path):
            out = pd.read_csv(out_path)
            print(f"[{i}/{len(summaries)}] {run}: already done ({len(out)} meshes)")
        else:
            np.random.seed(SEED)                             # NSM's samplers use np.random
            df = pd.read_csv(path)
            if not {"completed_path", "gt_path"} <= set(df.columns):
                print(f"[{i}/{len(summaries)}] {run}: skipped, no completed_path/gt_path columns")
                continue
            df = df.dropna(subset=["completed_path"])
            rows, missing = [], 0
            t0 = time.time()
            for r in df.itertuples():
                compl, gt = local(r.completed_path), local(r.gt_path)
                if not (os.path.exists(compl) and os.path.exists(gt)):
                    missing += 1
                    continue
                try:
                    rows.append(dict(mesh=r.mesh, **score(compl, gt)))
                except Exception as e:  # noqa: BLE001
                    print(f"    skip {r.mesh}: {type(e).__name__}: {e}")
            out = pd.DataFrame(rows)
            if len(out):
                out.to_csv(out_path, index=False)
            print(f"[{i}/{len(summaries)}] {run}: {len(out)} meshes in {time.time() - t0:.0f}s"
                  + (f", {missing} missing files" if missing else ""), flush=True)
        if len(out):
            medians[run] = out.drop(columns="mesh").median().rename(None)
            medians[run]["n"] = len(out)
    if not medians:
        return
    table = pd.DataFrame(medians).T
    table["n"] = table["n"].astype(int)
    folder = args[0] if len(args) == 1 and os.path.isdir(args[0]) else "."
    table.to_csv(os.path.join(folder, "shape_metrics_medians.csv"))
    pd.set_option("display.width", 250)
    print("\nMedians (chamfer columns in mesh units, as in the run summaries; signed_dist in % of bone size):")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1:]) if len(sys.argv) > 1 else sys.exit(__doc__)