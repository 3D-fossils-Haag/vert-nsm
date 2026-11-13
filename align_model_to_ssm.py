# Align new fossils using SSM and landmarks from original model training dataset
import numpy as np
import pyvista as pv
import os
import time
import pandas as pd
from NSM.alignment import load_mrk_json, save_mrk_json, align_to_template, rms_err_pct

# Load data
TRAIN_DIR = "run_v72" # TO DO: Choose training directory containing model ckpt and latent codes
mu_L_path = "atlas_db/sparse_landmarks.mrk.json"
mu_L, _ = load_mrk_json(mu_L_path)
mu_M_path = "atlas_db/template_model.ply"
mu_M = pv.read(mu_M_path)

# Save copy of reference SSM mesh and lms in train_dir
os.makedirs(os.path.join(TRAIN_DIR, "fossils/atlas"), exist_ok=True)
mu_M_outpath = os.path.join(TRAIN_DIR, "fossils/atlas", os.path.basename(mu_M_path))
mu_M.save(mu_M_outpath)
mu_L_outpath = os.path.join(TRAIN_DIR, "fossils/atlas", os.path.basename(mu_L_path))
save_mrk_json(mu_L, mu_L_outpath, template_path=mu_L_path)

# Input / output dirs (each with models/ and lms/ subfolders)
EVAL_FOSSILS = True      # TO DO: True = fossil evaluation mode (synthetic fossils from perturb_align_scale_fossils.py)
if EVAL_FOSSILS:
    perturb_dir = "perturbed_preprocessed"                   # TO DO: "perturbed" or "perturbed_preprocessed"
    in_dir = os.path.join(perturb_dir, "orig")
    out_dir = os.path.join(perturb_dir, "aligned")
    orig_dir = "fossilized_preprocessed" if "preprocessed" in perturb_dir else "fossilized"   # unperturbed fossils
    log_path = os.path.join(perturb_dir, "pose_log_lm.csv")   # same columns as align_morphoweave.py's pose_log_mw.csv
else:
    in_dir = "fossils"                          # TO DO: any dir with models/ and lms/ (fossils, new meshes, ...)
    out_dir = in_dir + "_aligned"               # TO DO
rows = []
models_dir = os.path.join(in_dir, "models")
lms_dir = os.path.join(in_dir, "lms")
out_models_dir = os.path.join(out_dir, "models")
out_lms_dir = os.path.join(out_dir, "lms")
os.makedirs(out_models_dir, exist_ok=True)
os.makedirs(out_lms_dir, exist_ok=True)
lm_files = {f.lower(): f for f in os.listdir(lms_dir)}   # case-insensitive landmark lookup

# Loop through meshes and lms to align them
meshes = [f for f in os.listdir(models_dir) if f.endswith(".vtk")]
print(f"Found {len(meshes)} meshes")
for mesh in meshes:
    # Build filepaths
    base = os.path.splitext(mesh)[0]
    lm_file = base.split("_hollow")[0] + ".mrk.json"          # also handles prep_nsm's _hollow_cut
    print("\nInspecting mesh: ", mesh)
    print("and landmark: ", lm_file, "\n")

    # Landmark file matched case-insensitively
    mesh_path = os.path.join(models_dir, mesh)
    lm_path = os.path.join(lms_dir, lm_files[lm_file.lower()]) if lm_file.lower() in lm_files else None
    row = dict(fossil=mesh, mode="lm")
    t0 = time.time()
    if not lm_path:
        print(f"\033[91mMissing landmarks for {base}, skipping.\033[0m")
        if EVAL_FOSSILS:
              row.update(status="error", error="missing landmarks")
              rows.append(row)
              pd.DataFrame(rows).to_csv(log_path, index=False)
        continue

    # Load mesh and landmarks
    Y_M = pv.read(mesh_path)
    Y_L, _, idx = load_mrk_json(lm_path, return_idx=True)

    # Align landmarks
    s_est, R_est, t_est = align_to_template(mu_L[idx], Y_L)
    Y_L_aligned = s_est * Y_L @ R_est + t_est

    # Apply transform to mesh
    Y_M_aligned = Y_M.copy()
    Y_M_aligned.points = s_est * Y_M.points @ R_est + t_est

    # Fix mesh normals
    try:
        Y_M_aligned.point_data.remove("Normals")
    except:
        pass
    Y_M_aligned.compute_normals(cell_normals=False, point_normals=True, inplace=True)

    # Save aligned, scaled mesh and landmarks
    out_mesh_path = os.path.join(out_models_dir, base + "_align.vtk")
    out_lm_path   = os.path.join(out_lms_dir, base + "_align.mrk.json")
    Y_M_aligned.save(out_mesh_path)
    save_mrk_json(Y_L_aligned, out_lm_path, template_path=lm_path, idx=idx)

    # Alignment statistics
    mean_error = np.mean(np.linalg.norm(mu_L[idx] - Y_L_aligned, axis=1))
    print(f"  Mean LM error: {mean_error:.6f}")
    print(f"  Determinant: {np.linalg.det(R_est):.6f}")

    # Fossil evaluation mode: alignment error vs the unperturbed fossil (RMS vertex displacement, % of bone size)
    if not EVAL_FOSSILS:
        continue
    row.update(status="ok", world_scale=s_est, n_landmarks=len(idx))
    orig_path = os.path.join(orig_dir, mesh)
    if os.path.exists(orig_path):
        orig = np.asarray(pv.read(orig_path).points, float)
        aligned = np.asarray(Y_M_aligned.points, float)
        if len(orig) == len(aligned):
            row["rms_err_pct"] = rms_err_pct(aligned, orig)
            print(f"  Alignment error: {row['rms_err_pct']:.2f}% of size")
    row["seconds"] = round(time.time() - t0, 1)
    rows.append(row)
    pd.DataFrame(rows).to_csv(log_path, index=False)   # saved after each fossil