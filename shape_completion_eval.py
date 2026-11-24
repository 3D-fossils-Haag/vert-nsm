# Fine-tune parameters for shape completion of partial vertebrae
# Use partial vertebrae dataset made with create_partial_meshes.py for validation against ground truth meshes

import os
import glob
import torch
import numpy as np
import pandas as pd
import json
import sys
import pyvista as pv
import pymskt.mesh.meshes as meshes
import vtk
import random
from NSM.helper_funcs import load_config, load_model_and_latents, convert_ply_to_vtk, fixed_point_coords, safe_load_mesh_scalars 
from NSM.optimization import normalize_mesh, get_norm_params, encode_latent, encode_latent_pointnet, reconstruct_mesh_from_latent, build_sdf_dataset, optimize_latent_partial
from NSM.evaluation import strip_partial_mesh_name, build_partial_gt_mesh_pairs, uniform_surface_sample, chamfer_distance, load_best_cfg_from_csv, region_errors
from NSM.fossils import undo_pose_error, undo_pose, optimize_latent_pose, POSE_STANDARD, snap_to_partial, mask_broken_region

# Monkey Patch into pymskt.mesh.meshes.Mesh
meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
meshes.Mesh.point_coords = property(fixed_point_coords)
import time

# ============================================================================
# Settings (TO DO)
# ============================================================================
TRAIN_DIR = "run_v72"          # training directory with model ckpt and latent codes
CKPT = "2500"                  # checkpoint to analyze
split = "test"                 # split to evaluate: "train", "val" or "test"
N_INF = "all"                  # meshes to run, or "all"
fast_mode = True               # True = PointNet encoder (+ refinement), False = 2-phase latent optimization (slow)
encoder_path = TRAIN_DIR + "/encoder/checkpoints/encoder.pt"   # fast mode only
LOAD_BEST_CFG_FROM_CSV = True  # False = manual cfg below
downsample_experiment = False  # True = score every model on one common set of meshes (run_v72 partials)

# Fossils: set these exactly as in the grid search run the best cfg comes from
fossils = True                 # True = score synthetic fossils against their intact meshes
preprocessed = False           # True = preprocessed (smoothed, hole-filled) fossils; also masks broken SDF samples
perturbed = True               # True = perturbed + realigned fossils
stage = "aligned_mw_surf_per_foss_cover"   # perturbed only:
                               #   "aligned"        = landmark alignment (align_model_to_ssm.py)
                               #   "aligned_mw_lms" = MorphoWeave, surface + landmarks
                               #   "aligned_mw_surf"= MorphoWeave, surface only
                               #   + "_per_foss_cover" for per-fossil coverage
POSE_CORRECTION = True         # refine latent + small pose correction of the input points
GRID_SPLIT = "train"           # split the grid search ran on
LABEL_DIR = "fossilized"       # raw fossils with the 'broken' labels used for scoring
GT_PATH_FIX = ("/home/k.wolcott/", "/home/kaubricot/Downloads/")   # KW local patch: (old, new) prefix of GT paths, None on devbox

# ============================================================================
# Derived (no need to edit)
# ============================================================================
LC_PATH = TRAIN_DIR + '/latent_codes/' + CKPT + '.pth'
MODEL_PATH = TRAIN_DIR + '/model/' + CKPT + '.pth'
MESH_LIST = "common_eval_meshes.json" if downsample_experiment else None
PARTIAL_DIR = "run_v72" if downsample_experiment else TRAIN_DIR
val_sum_fn = PARTIAL_DIR + "/shape_completion/meshes/partial_meshing_summary.json"
encoder_ckpt = os.path.abspath(encoder_path)

# Grid search run name, built the same way as in shape_completion_grid_search.py
SNAP = fossils                          # rigidly snap each completion onto the input fossil
MASK_BROKEN = fossils and preprocessed  # leave out SDF samples whose nearest fossil surface is broken
SELECT_BY = "mean_chamfer_filled" if fossils else "mean_cd"
GRID_RUN = "fine_tuning_encoder" if fast_mode else "fine_tuning"
if fossils:
    if not perturbed:
        stage = "orig"
    base = "fossilized_preprocessed" if preprocessed else LABEL_DIR   # unperturbed copies
    perturb_dir = "perturbed_preprocessed" if preprocessed else "perturbed"
    FOSSIL_DIR = f"{perturb_dir}/{stage}/models" if perturbed else base
    pose_cfg = POSE_STANDARD
    GRID_RUN += "_" + FOSSIL_DIR.rstrip("/").replace("/", "_")
    if POSE_CORRECTION:
        GRID_RUN += "_pose"
if SNAP:
    GRID_RUN += "_snap"
if MASK_BROKEN:
    GRID_RUN += "_nobroken"
if fossils:
    BEST_CFG_CSV = f"{TRAIN_DIR}/shape_completion/{GRID_RUN}/{GRID_SPLIT}/trial_scores.csv"
    tag = GRID_RUN.split("fine_tuning_encoder" if fast_mode else "fine_tuning", 1)[1]   # run settings, in the output name
else:
    BEST_CFG_CSV = f"{TRAIN_DIR}/shape_completion/{GRID_RUN}/trial_scores.csv"
    tag = "_common" if downsample_experiment else ""

# Load model config
config = load_config(config_path=TRAIN_DIR + '/model_params_config.json')
device = config.get("device", "cuda:0")

# Load model and latent codes
model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)
mean_latent = latent_codes.mean(dim=0, keepdim=True)

# Load best optimization config from shape_completion_grid_search.py (or manually enter)
if LOAD_BEST_CFG_FROM_CSV:
    best_cfg = load_best_cfg_from_csv(BEST_CFG_CSV, fast_mode, device, select_by=SELECT_BY)
    cfg_source = f"CSV ({BEST_CFG_CSV})"

else:        # TO DO: Manually enter chosen vals
    if fast_mode:
        best_cfg = {"iters": 200, # TO DO: Refine pointnet encoding with additional iterations (Ex: 300-500)
                    "top_k": 466,
                    "lr": 1e-3,
                    "lambda_reg": 1e-6,
                    "clamp": None,
                    "latent_std": torch.tensor(0.4401),
                    "sched_step": 300,
                    "sched_gamma": 0.9,
                    "batch_infer": 32768,
                    "gridN": 256,
                    "n_samples": None}
    else:
        best_cfg = {"top_k": 466,
                    "iters1": 5000,
                    "iters2": 8000,
                    "lr1": 1e-4,
                    "lr2": 1e-4,
                    "lambda1": 1e-2,
                    "lambda2": 1e-7,
                    "clamp1": 1,
                    "clamp2": None,
                    "latent_std": torch.tensor(0.4401),
                    "sched_step": 800,
                    "sched_gamma1": 0.7,
                    "sched_gamma2": 0.9,
                    "batch_infer": 32768,
                    "gridN": 256}
    cfg_source = "Manual entry"
        
encoding_strategy = "PointNet encoder" if fast_mode else "2-phase latent optimization"
print(f"\n\nEncoding strategy : {encoding_strategy}")
print(f"Config source     : {cfg_source}")
print(f"Config values     :")
for k, v in best_cfg.items():
    print(f"  {k:<20} {v}")

# Build validation ground truth dataset
ds_split_keys = {"train": "list_mesh_paths", "val": "val_paths", "test": "test_paths"}
split_key = ds_split_keys[split]
if MESH_LIST:
    mesh_paths = json.load(open(MESH_LIST))
    print(f"\n\nUsing {len(mesh_paths)} meshes from {MESH_LIST}")
else:
    mesh_paths = config[split_key]
    print(f"\n\nFound {len(mesh_paths)} meshes in config['{split_key}'] (split='{split}')")
mesh_names = {strip_partial_mesh_name(p) for p in mesh_paths}

# Load partial meshing summary
# Find corresponding partial meshes to use for shape completion against ground truth meshes
if fossils:
    # Pair "<name>_fossil*.vtk" with the intact "<name>" mesh in this split
    gt_paths = {os.path.splitext(os.path.basename(p))[0]: p.replace(*GT_PATH_FIX) if GT_PATH_FIX else p
                for p in mesh_paths}
    pairs = sorted((p, gt_paths[os.path.basename(p).split("_fossil")[0]])
                   for p in glob.glob(os.path.join(FOSSIL_DIR, "*.vtk"))
                   if os.path.basename(p).split("_fossil")[0] in gt_paths)
else:
    with open(val_sum_fn, "r") as f:
        partial_mesh_summary = json.load(f)
    pairs = build_partial_gt_mesh_pairs(partial_mesh_summary, mesh_names, MESH_LIST or split_key)

# Build outfpath
outfpath = TRAIN_DIR + '/shape_completion/evaluation/'
os.makedirs(outfpath, exist_ok=True)
mode_name = "encoder" if fast_mode else "2phase"
summary_df_fpath = outfpath + f"{mode_name}_{split}{tag}_chamfer.csv"
COLS = ["mesh", "chamfer", "gt_path",
        "intact_frac", "chamfer_intact", "l1_intact", "chamfer_filled", "l1_filled",
        "chamfer_shape", "intact_frac_shape", "chamfer_intact_shape", "l1_intact_shape", "chamfer_filled_shape", "l1_filled_shape"]
csv_file = open(summary_df_fpath, 'w', newline='')   # new file each run, rows appended below
csv_file.write(",".join(COLS) + "\n")
print(f"\nSaving results to: {summary_df_fpath}")

# Loop through meshes using best parameters from grid search
inf_subset = random.sample(pairs, N_INF) if N_INF != "all" else pairs
for i, (pm_path, gt_path) in enumerate(inf_subset, start=1):    
    try:
        print(f"\033[32m\n=== Processing {os.path.basename(pm_path)} ===\033[0m")
        print(f"\033[32m\n=== {i} of {len(inf_subset)} ===\033[0m")
        vert_fname = pm_path
        # Convert plys to vtks
        if '.ply' in vert_fname:
            ply_fname = vert_fname
            mesh, vert_fname = convert_ply_to_vtk(ply_fname, save=True)

        # Latent encoding
        if fast_mode:
            # Build the SDF dataset using all surface samples
            points, sdf_vals, sdf_dataset, sample_dict = build_sdf_dataset(vert_fname, config, n_samples=None if MASK_BROKEN else best_cfg.get('n_samples'))   # None = all points
            if MASK_BROKEN:   # only keep SDF samples from intact surface
                center, max_radius = get_norm_params(sdf_dataset, sample_dict, vert_fname)
                points, sdf_vals = mask_broken_region(points, sdf_vals, vert_fname, center, max_radius, best_cfg.get('n_samples'))
           
            # Fast mode: one-shot encoder instead of full latent optimization 
            print("\n-----Fast mode: encoding latent (single forward pass)----\n")
            latent_opt = encode_latent_pointnet(encoder_ckpt, points, sdf_vals, device)
            if best_cfg['iters'] > 0:
                print(f"Refining encoder latent for {best_cfg['iters']} iters (lr={best_cfg['lr']}, lambda={best_cfg['lambda_reg']})...")
                latent_opt, _ = optimize_latent_partial(model, points.squeeze(), sdf_vals, config['latent_size'], latent_init=latent_opt, top_k=best_cfg['top_k'],
                                                        iters=best_cfg['iters'], lr=best_cfg['lr'], lambda_reg=best_cfg['lambda_reg'], clamp_val=best_cfg['clamp'],
                                                        latent_std=best_cfg['latent_std'], scheduler_step=best_cfg['sched_step'],
                                                        scheduler_gamma=best_cfg['sched_gamma'], batch_inference_size=best_cfg['batch_infer'], multi_stage=True, device=device)

        else:
            # Setup your dataset with just one mesh
            points, sdf_vals, sdf_dataset, sample_dict = build_sdf_dataset(vert_fname, config, n_samples=None if MASK_BROKEN else best_cfg.get('n_samples') or 240)
            if MASK_BROKEN:   # only keep SDF samples from intact surface
                center, max_radius = get_norm_params(sdf_dataset, sample_dict, vert_fname)
                points, sdf_vals = mask_broken_region(points, sdf_vals, vert_fname, center, max_radius, best_cfg.get('n_samples') or 240)

            # Encode latent via 2 stage optimization (auto-decoder framework)
            latent_opt = encode_latent(decoder=model, points=points.squeeze(), sdf_vals=sdf_vals, latent_dim=latent_codes.shape[1], 
                                    mean_latent=mean_latent, latent_codes=latent_codes, top_k_reg=best_cfg['top_k'], latent_std=best_cfg['latent_std'],
                                    iters1=best_cfg['iters1'], iters2=best_cfg['iters2'], lr1=best_cfg['lr1'], lr2=best_cfg['lr2'], 
                                    lambda_reg1=best_cfg['lambda1'], lambda_reg2=best_cfg['lambda2'], clamp_val1=best_cfg['clamp1'], clamp_val2=best_cfg['clamp2'], 
                                    scheduler_step1=best_cfg['sched_step'], scheduler_step2=best_cfg['sched_step'], scheduler_gamma1=best_cfg['sched_gamma1'],
                                    scheduler_gamma2=best_cfg['sched_gamma2'], batch_inference_size=best_cfg['batch_infer']) 

        # Pose correction (as in the grid search run), then reconstruction mesh from latent
        pose = None
        if fossils and POSE_CORRECTION:
            latent_opt, pose = optimize_latent_pose(model, points.squeeze(), sdf_vals, latent_opt, pose_cfg, device=device)
        mesh_out = reconstruct_mesh_from_latent(vert_fname, model, latent_opt, best_cfg)
        if pose is not None:
            mesh_out = undo_pose(mesh_out, pose)   # back onto the fossil, before normalize_mesh

        # Normalize and scale output using model config training params
        center, max_radius = get_norm_params(sdf_dataset, sample_dict, vert_fname)
        mesh_pv = normalize_mesh(mesh_out, vert_fname, config, center, max_radius)
        if SNAP:
            fos = pv.read(vert_fname)
            fpts = np.asarray(fos.points, float)
            if MASK_BROKEN and "broken" in fos.point_data:   # snap onto intact surface only
                fpts = fpts[np.asarray(fos.point_data["broken"]) == 0]
            mesh_pv = snap_to_partial(mesh_pv, fpts[::5])
        
        # Calculate chamfer distance between shape completed and original-ground truth mesh
        pred = mesh_pv.clean().triangulate().extract_surface(algorithm=None)
        gt = pv.read(gt_path).triangulate().extract_surface(algorithm=None)
        cd = chamfer_distance(pred, gt, n_samples=20000)
        print(f"\n{os.path.basename(pm_path)} Chamfer={cd:.4f}\n") 
        regions = {}
        if fossils:
            label_path = os.path.join(LABEL_DIR, os.path.basename(pm_path).replace("_hollow_cut", "").replace("_align", ""))
            regions = region_errors(pred, gt, label_path)                            # intact / filled
            ref_path = os.path.join(base, os.path.basename(pm_path).replace("_align", ""))   # unperturbed copy, same vertices as the input
            pred_shape = undo_pose_error(pred, pm_path, ref_path)                    # alignment error removed
            regions["chamfer_shape"] = chamfer_distance(pred_shape, gt, n_samples=20000)
            regions.update({k + "_shape": v for k, v in region_errors(pred_shape, gt, label_path).items()})
            print({k: round(float(v), 4) for k, v in regions.items()})
        # Log results to csv
        row = pd.DataFrame([{'mesh': os.path.basename(pm_path), 'chamfer': cd, 'gt_path': gt_path, **regions}])
        row.reindex(columns=COLS).to_csv(csv_file, index=False, header=False)
        csv_file.flush()

    except Exception as e:
        print(f"\033[31mERROR processing {os.path.basename(pm_path)}: {e}\033[0m")
        # Log results to csv
        row = pd.DataFrame([{'mesh': os.path.basename(pm_path), 'chamfer': None, 'gt_path': gt_path}])
        row.reindex(columns=COLS).to_csv(csv_file, index=False, header=False)
        csv_file.flush()
        continue

# Save results
csv_file.close()
summary_df = pd.read_csv(summary_df_fpath)
print(f"\nMedians across {summary_df['chamfer'].notna().sum()} of {len(summary_df)} meshes:")
print(summary_df.drop(columns=["mesh", "gt_path"]).median().round(4).to_string())
print("Outputs logged to: ", summary_df_fpath)

sys.stdout.flush()
sys.stderr.flush()
os._exit(0)