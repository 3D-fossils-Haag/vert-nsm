# Align new fossils using SSM and landmarks from original model training dataset
import numpy as np
import pyvista as pv
import json
import os
import time
import pandas as pd

# Define functions
def load_slicer_landmarks(path):
    with open(path, "r") as f:
        data = json.load(f)
    pts = []
    for cp in data["markups"][0]["controlPoints"]:
        pts.append(cp["position"])
    return np.array(pts)

def load_landmark_indices(path):
    with open(path) as f:
        cps = json.load(f)["markups"][0]["controlPoints"]
    return [int(cp["description"].split("index:")[1]) if "index:" in cp.get("description", "") else i
            for i, cp in enumerate(cps)]

def save_slicer_landmarks(points, output_path, template_path=None, idx=None):
    if template_path is not None:
        with open(template_path, "r") as f:
            data = json.load(f)
    else:
        # minimal Slicer structure
        data = {"markups": [{"controlPoints": []}]}
    control_points = []
    for i, pt in enumerate(points):
        control_points.append({
            "id": str(i),
            "label": str(i),
            "position": pt.tolist(),
            "selected": True,
            "locked": False,
            "visibility": True, 
            "description": f"index:{idx[i]}" if idx is not None else ""})
    data["markups"][0]["controlPoints"] = control_points
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

def align_to_template(mu_L, Y_L, allow_scaling=True):
    # Centroids
    mu_centroid = mu_L.mean(axis=0)
    Y_centroid  = Y_L.mean(axis=0)
    # Center
    mu_c = mu_L - mu_centroid
    Y_c  = Y_L  - Y_centroid
    # Cross covariance
    H = Y_c.T @ mu_c
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    # Reflection correction
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    # Scaling
    if allow_scaling:
        scale = S.sum() / np.sum(Y_c**2)
    else:
        scale = 1.0
    # Translation
    t = mu_centroid - scale * Y_centroid @ R
    # Apply transform
    Y_aligned = scale * Y_L @ R + t
    return Y_aligned, scale, R, t

def apply_transform_to_mesh(mesh, scale, R, t):
    V = mesh.points
    V_new = scale * V @ R + t
    mesh_aligned = mesh.copy()
    mesh_aligned.points = V_new
    return mesh_aligned

def find_ci_file(directory, name):
    if not os.path.isdir(directory):
        return None
    nl = name.lower()
    for f in os.listdir(directory):
        if f.lower() == nl:
            return os.path.join(directory, f)
    return None

# Load data
TRAIN_DIR = "run_v72" # TO DO: Choose training directory containing model ckpt and latent codes
mu_L_path = "atlas_db/sparse_landmarks.mrk.json"
mu_L = load_slicer_landmarks(mu_L_path)
mu_M_path = "atlas_db/template_model.ply"
mu_M = pv.read(mu_M_path)

# Save copy of reference SSM mesh and lms in train_dir
os.makedirs(os.path.join(TRAIN_DIR, "fossils/atlas"), exist_ok=True)
mu_M_outpath = os.path.join(TRAIN_DIR, "fossils/atlas", os.path.basename(mu_M_path))
mu_M.save(mu_M_outpath)
mu_L_outpath = os.path.join(TRAIN_DIR, "fossils/atlas", os.path.basename(mu_L_path))
save_slicer_landmarks(mu_L, mu_L_outpath, template_path=mu_L_path)

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

# Loop through meshes and lms to align them
meshes = [f for f in os.listdir(models_dir) if f.endswith(".vtk")]
print(f"Found {len(meshes)} meshes")
for mesh in meshes:
    # Build filepaths
    base = os.path.splitext(mesh)[0]
    lm_file = base.split("_hollow")[0] + ".mrk.json"          # also handles prep_nsm's _hollow_cut
    print("\nInspecting mesh: ", mesh)
    print("and landmark: ", lm_file, "\n")

    # Case insensitive matching
    mesh_path = find_ci_file(models_dir, mesh)
    lm_path = find_ci_file(lms_dir, lm_file)
    row = dict(fossil=mesh, mode="lm")
    t0 = time.time()
    if not lm_path:
        print(f"\033[91mMissing landmarks for {base}, skipping.\033[0m")
        if EVAL_FOSSILS:
            row.update(status="error", error="missing landmarks")
            rows.append(row)
        continue

    # Load mesh and landmarks
    Y_M = pv.read(mesh_path)
    Y_L = load_slicer_landmarks(lm_path)

    # Load landmark indices to match even if some missing
    idx = load_landmark_indices(lm_path)

    # Align landmarks
    Y_L_aligned, s_est, R_est, t_est = align_to_template(mu_L[idx], Y_L, allow_scaling=True)
    # Apply transform to mesh
    Y_M_aligned = apply_transform_to_mesh(Y_M, s_est, R_est, t_est)

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
    save_slicer_landmarks(Y_L_aligned, out_lm_path, template_path=lm_path, idx=idx)

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
            size = np.linalg.norm(np.ptp(orig, axis=0))
            row["rms_err_pct"] = 100 * np.sqrt(np.mean(np.sum((aligned - orig) ** 2, 1))) / size
            print(f"  Alignment error: {row['rms_err_pct']:.2f}% of size")
    row["seconds"] = round(time.time() - t0, 1)
    rows.append(row)
    pd.DataFrame(rows).to_csv(log_path, index=False)   # saved after each fossil