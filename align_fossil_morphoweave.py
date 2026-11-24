#!/usr/bin/env python3
"""
align_fossil_morphoweave.py: align fossils to the atlas frame with MorphoWeave's SSM pose search
(surface point cloud, optionally helped by landmarks), instead of landmarks alone.

    python align_fossil_morphoweave.py perturbed/
    python align_fossil_morphoweave.py perturbed_preprocessed/
    python align_fossil_morphoweave.py perturbed/ --reverse   # 2nd process from the end, logs to pose_log_mw*_rev.csv

Reads <dir>/orig/models/*.vtk (+ <dir>/orig/lms/*.mrk.json from perturb_align_scale_fossils.py),
writes <dir>/aligned_mw_surf/ (surface only) and <dir>/aligned_mw_lms/ (surface + landmarks), each with
models/*_align.vtk and lms/*_align.mrk.json (landmarks moved with the same pose), plus <dir>/pose_log_mw.csv.
With PER_FOSSIL_COVERAGE these get a "_per_foss_cover" suffix (aligned_mw_lms_per_foss_cover/, pose_log_mw_per_foss_cover.csv). If the unperturbed
fossils are in ORIG_DIR (same vertex order), the log also gets the alignment error, measured
like perturb_log.csv (RMS vertex displacement, % of bone size).

Run it on raw fossil surfaces, not hollow_cut shells (the double wall confuses the fit):
align first, then preprocess. Requires numpy scipy pandas pyvista rustcpd==4.0.0 and the
MorphoWeave ShapeCompletion source (MW_PYTHON below).
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import pyvista as pv

from NSM.alignment import load_mrk_json, save_mrk_json, rms_err_pct

# ----------------------------------------------------------------------------
# Paths and settings
# ----------------------------------------------------------------------------
MW_PYTHON = "/home/kaubricot/Downloads/SlicerMorphoWeave-perf-robust-pose-search/MorphoWeaveShapeCompletion/Resources/Python"   # TO DO
ATLAS_DIR = "atlas_db"           # TO DO: ssm_model.npz, sparse_landmarks.mrk.json, dense_correspondences.mrk.json, template_model.ply
ORIG_DIR = "fossilized"          # unperturbed fossils, for measuring alignment error (None to skip)
USE_LANDMARKS = (False, True)    # run surface-only, landmark-assisted, or both

PER_FOSSIL_COVERAGE = True       # TO DO: True = each fossil's own coverage ('complete' in FOSSIL_LOG)
                                 #        False = the same COVERAGE for every fossil
FOSSIL_LOG = "fossilized/fossilize_log.csv"   # TO DO: fossilize.py log, for PER_FOSSIL_COVERAGE
COVERAGE = 0.75                  # coverage for all fossils (or fallback for fossils missing from FOSSIL_LOG)
PIPELINE = "robust"              # MorphoWeave's recommended pose search

SEED = 0
RAS_TO_LPS = np.array([-1.0, -1.0, 1.0])   # set to np.ones(3) if your ssm_model.npz is already LPS

sys.path.insert(0, MW_PYTHON)
import rustcpd  # noqa: E402
from MorphoWeaveShapeCompletionCore import CompletionSettings, match_landmarks, run_completion  # noqa: E402

def load_coverage(log_path):
    """{fossil file name: 'complete' (fraction of the bone present)}, from fossilize.py's log."""
    if not log_path or not os.path.exists(log_path):
        return {}
    log = pd.read_csv(log_path)
    log = log[(log["status"] == "ok") & log["complete"].notna()]
    return dict(zip(log["output"].map(lambda p: os.path.splitext(os.path.basename(p))[0]), log["complete"].clip(0.05, 1.0)))

def to_atlas(points, scale, rotation, translation):
    """Invert MorphoWeave's SSM->fossil similarity (x_f = s * x @ R + t) to bring the fossil into the SSM frame."""
    return (np.asarray(points, float) - translation) @ rotation.T / scale

def main(perturb_dir, reverse=False):
    z = np.load(os.path.join(ATLAS_DIR, "ssm_model.npz"))
    mean, modes, eig = z["mean_shape"].astype(float), z["modes"].astype(float), z["eigenvalues"].astype(float)
    mean, modes = mean * RAS_TO_LPS, modes * RAS_TO_LPS[:, None]   # npz is RAS (Slicer); meshes and markups are LPS
    template = np.asarray(pv.read(os.path.join(ATLAS_DIR, "template_model.ply")).points, float)
    dense = load_mrk_json(os.path.join(ATLAS_DIR, "dense_correspondences.mrk.json"))[0].astype(float)
    atlas_lm, atlas_labels = load_mrk_json(os.path.join(ATLAS_DIR, "sparse_landmarks.mrk.json"))
    atlas_lm = atlas_lm.astype(float)
    coverage = load_coverage(FOSSIL_LOG) if PER_FOSSIL_COVERAGE else {}
    suffix = "_per_foss_cover" if PER_FOSSIL_COVERAGE else ""
    orig_dir = ORIG_DIR + "_preprocessed" if ORIG_DIR and "preprocessed" in perturb_dir else ORIG_DIR
    mode_names = ["lms" if u else "surf" for u in USE_LANDMARKS]
    print(f"coverage from {FOSSIL_LOG}: {len(coverage)} fossils (others use {COVERAGE})" if PER_FOSSIL_COVERAGE
          else f"coverage {COVERAGE} for all fossils")
    log_path = os.path.join(perturb_dir, f"pose_log_mw{suffix}{'_rev' if reverse else ''}.csv")   # own log per run: no overwrites
    rows = pd.read_csv(log_path).to_dict("records") if os.path.exists(log_path) else []
    for path in sorted(glob.glob(os.path.join(perturb_dir, "orig", "models", "*.vtk")), reverse=reverse):
        fname = os.path.basename(path)
        stem = os.path.splitext(fname)[0]
        if all(os.path.exists(os.path.join(perturb_dir, f"aligned_mw_{m}{suffix}", "models", stem + "_align.vtk")) for m in mode_names):
            continue   # already aligned (all modes)
        mesh = pv.read(path)
        pts = np.asarray(mesh.points, float)
        raw = stem.split("_hollow")[0]                  # fossil name without preprocessing suffix
        lm_name = raw + ".mrk.json"
        lm_path = os.path.join(perturb_dir, "orig", "lms", lm_name)
        has_lm = os.path.exists(lm_path)
        if has_lm:
            lm, _, idx = load_mrk_json(lm_path, return_idx=True)
            lm = lm.astype(float)
        orig_path = os.path.join(orig_dir, fname) if orig_dir else None
        orig = np.asarray(pv.read(orig_path).points, float) if orig_path and os.path.exists(orig_path) else None
        cov = coverage.get(raw, COVERAGE)
        for use_lm, mode in zip(USE_LANDMARKS, mode_names):
            row = dict(fossil=fname, mode=mode, coverage=cov)
            out_dir = os.path.join(perturb_dir, "aligned_mw_" + mode + suffix)
            t0 = time.time()
            try:
                landmarks = None
                if use_lm:
                    labels = [atlas_labels[i] for i in idx]          # pair by index in the full landmark set
                    landmarks = match_landmarks(atlas_labels, atlas_lm, labels, lm, dense)
                    row["n_landmarks"] = landmarks.count
                settings = CompletionSettings(coverage=cov, use_landmarks=use_lm, pipeline=PIPELINE, seed=SEED)
                res = run_completion(pts, mean, modes, eig, settings, rustcpd_module=rustcpd,
                                     landmarks=landmarks, template_points=template)
                aligned = to_atlas(pts, res.world_scale, res.world_rotation, res.world_translation)
                out = mesh.copy()
                out.points = aligned
                if "Normals" in out.point_data:
                    out.point_data.remove("Normals")
                sub = os.path.join(out_dir, "models")
                os.makedirs(sub, exist_ok=True)
                out.save(os.path.join(sub, stem + "_align.vtk"))
                if has_lm:
                    lm_aligned = to_atlas(lm, res.world_scale, res.world_rotation, res.world_translation)
                    save_mrk_json(lm_aligned, os.path.join(out_dir, "lms", stem + "_align.mrk.json"), template_path=lm_path, idx=idx)
                row.update(status="ok", world_scale=res.world_scale, sigma2=res.normalized_sigma2)
                if orig is not None and len(orig) == len(aligned):
                    row["rms_err_pct"] = rms_err_pct(aligned, orig)
            except Exception as e:  # noqa: BLE001
                row.update(status="error", error=f"{type(e).__name__}: {e}")
            row["seconds"] = round(time.time() - t0, 1)
            rows.append(row)
            print(f"{row['status']:5s} {fname} [{row['mode']}, cov {cov:.2f}] "
                  + (f"error {row['rms_err_pct']:.2f}% of size " if "rms_err_pct" in row else row.get("error", ""))
                  + f"({row['seconds']}s)", flush=True)
        pd.DataFrame(rows).to_csv(log_path, index=False)   # saved after each fossil
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--reverse"]
    main(args[0], reverse="--reverse" in sys.argv[1:]) if len(args) == 1 else sys.exit(__doc__)