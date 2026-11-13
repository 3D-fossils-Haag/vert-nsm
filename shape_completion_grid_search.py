# Fine-tune parameters for shape completion of partial vertebrae
# Use partial vertebrae dataset made with create_partial_meshes.py for validation against ground truth meshes

import os
import torch
import numpy as np
import pandas as pd
import json
import sys
import glob
import pyvista as pv
import pymskt.mesh.meshes as meshes
import vtk
import random
from NSM.helper_funcs import load_config, load_model_and_latents, convert_ply_to_vtk, fixed_point_coords, safe_load_mesh_scalars 
from NSM.optimization import normalize_mesh, get_norm_params, encode_latent, encode_latent_pointnet, reconstruct_mesh_from_latent, build_sdf_dataset, optimize_latent_partial, undo_pose, optimize_latent_pose, POSE_STANDARD, POSE_LOW_COMPLETE, snap_to_partial
from NSM.evaluation import strip_partial_mesh_name, build_partial_gt_mesh_pairs, uniform_surface_sample, chamfer_distance, load_best_cfg_from_csv, grid_search, grid_search_pointnet, region_errors, undo_pose_error
# Monkey Patch into pymskt.mesh.meshes.Mesh
meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
meshes.Mesh.point_coords = property(fixed_point_coords)
import time

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Define training directory
TRAIN_DIR = "run_v72" # TO DO: Choose training directory containing model ckpt and latent codes
CKPT = '2500' # TO DO: Choose the ckpt value you want to analyze results for
LC_PATH =  TRAIN_DIR + '/latent_codes' + '/' + CKPT + '.pth'
MODEL_PATH = TRAIN_DIR +  '/model' + '/' + CKPT + '.pth'

# Downsample experiment: every model is scored on one common set of meshes,
# using the partials generated once for run_v72
downsample_experiment = False  # TO DO: Set to True if running for downsample experiment
PARTIAL_DIR = "run_v72" if downsample_experiment else TRAIN_DIR

# Build remaining parameters
val_sum_fn = PARTIAL_DIR + "/shape_completion/meshes/partial_meshing_summary.json"
N_TRIALS = 15   # TO DO: Choose the number of trials for the grid search
N_TRIAL_INF = 30   # TO DO: Choose the number of meshes to use for inference in each grid search trial
N_FINAL_INF = 50   # TO DO: Choose how many meshes to use for final inference with best config reconstruction parameters from grid search
split = "train"  # TO DO: Which dataset split to use - "train", "val", or "test"

# Fast mode - custom trained PointNet encoder
fast_mode = True  # TO DO: Use fast mode to encode via PointNet or using 2-phase latent optimization (slow)
if fast_mode == True:
    OUTDIR = TRAIN_DIR + "/shape_completion/fine_tuning_encoder" 
    encoder_path = TRAIN_DIR + "/encoder/checkpoints/encoder.pt" # TO DO: Point to encoder ckpt
    encoder_ckpt = os.path.abspath(encoder_path)
else: 
    OUTDIR = TRAIN_DIR + "/shape_completion/fine_tuning"

# Use fossil data?
fossils = True  # TO DO: True = score synthetic fossils against their intact meshes
if fossils:
    LABEL_DIR = "fossilized"                       # raw fossils, carry the 'broken' labels
    preprocessed = True                           # TO DO: True = score the hollow_cut meshes
    perturbed = True                               # TO DO: True = score perturbed (+ realigned) fossils
    stage = "aligned_mw_lms_per_foss_cover"    # TO DO: "orig" = perturbed only (not realigned)
                      #        "aligned" = perturbed and/or preprocessed, then realigned from landmarks (align_model_to_ssm.py)
                      #        "aligned_mw_lms" = perturbed and/or preprocessed, then realigned with MorphoWeave, surface + landmarks
                      #        "aligned_mw_surf" = perturbed and/or preprocessed, then realigned with MorphoWeave, surface only
                      #        add "_per_foss_cover" to either MorphoWeave stage for per-fossil coverage (e.g. "aligned_mw_lms_cov")
                      #        add "_per_foss_cover" to either MorphoWeave stage for per-fossil coverage (e.g. "aligned_mw_lms_per_foss_cover")
    base = "fossilized_preprocessed" if preprocessed else LABEL_DIR
    perturb_dir = "perturbed_preprocessed" if preprocessed else "perturbed"
    FOSSIL_DIR = f"{perturb_dir}/{stage}/models" if perturbed else base
    PERTURB_LOG = f"{perturb_dir}/perturb_log.csv" if perturbed else None
    PER_FOSSIL_COV = "per_foss_cover" in stage                     # per-fossil coverage run
    MW_MODE = "lms" if "_lms" in stage else "surf"                 # mode column in the MorphoWeave pose log
    MW_LOG = (f"{perturb_dir}/pose_log_mw{'_per_foss_cover' if PER_FOSSIL_COV else ''}.csv"
              if perturbed and stage.startswith("aligned_mw") else None)
    OUTDIR += "_" + FOSSIL_DIR.rstrip("/").replace("/", "_")

    POSE_CORRECTION = True    # TO DO
    LOW_COMPLETE = False      # TO DO: True = fossils < 50% complete (wider limits, 7 starts, 2x iterations)
    pose_cfg = POSE_LOW_COMPLETE if LOW_COMPLETE else POSE_STANDARD
    if POSE_CORRECTION:
        OUTDIR += "_pose_low" if LOW_COMPLETE else "_pose"

    SNAP = True   # TO DO: rigidly snap each completion onto the input fossil (fixes NSM placement offset)
    if SNAP:
        OUTDIR += "_snap"

# One outdir by data split
OUTDIR = os.path.join(OUTDIR, split)

# Load model config
config = load_config(config_path=TRAIN_DIR + '/model_params_config.json')
device = config.get("device", "cuda:0")

# Load model and latent codes
model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)
mean_latent = latent_codes.mean(dim=0, keepdim=True)

# Build validation ground truth dataset
ds_split_keys = {"train": "list_mesh_paths", "val": "val_paths", "test": "test_paths"}
split_key = ds_split_keys[split]
mesh_names = {strip_partial_mesh_name(p) for p in config[split_key]}
print(f"Found {len(config[split_key])} meshes in config['{split_key}'] (split='{split}')")

# Find files if for fossils or modern specimens
if fossils:
    # Pair "<name>_fossil*.vtk" with the intact "<name>" mesh in this split
    #gt_paths = {os.path.splitext(os.path.basename(p))[0]: p for p in config[split_key]} # KW TO DO: patch for working locally not on devox
    gt_paths = {os.path.splitext(os.path.basename(p))[0]: p.replace("/home/k.wolcott/", "/home/kaubricot/Downloads/")
                for p in config[split_key]}
    pairs = sorted((p, gt_paths[os.path.basename(p).split("_fossil")[0]])
                   for p in glob.glob(os.path.join(FOSSIL_DIR, "*.vtk"))
                   if os.path.basename(p).split("_fossil")[0] in gt_paths)
    print(f"{len(pairs)} fossil/intact pairs in '{split}'")

    #ONLY = "agamidae_agama_atra_uf180711_07-t2"   # KW TO DO: quick patch, None = all fossils
    #if ONLY:
        #pairs = [pr for pr in pairs if os.path.basename(pr[0]).startswith(ONLY + "_fossil")]
        #print(f"Filtered to {len(pairs)} pair(s) matching {ONLY}")

else:
    with open(val_sum_fn, "r") as f:
        partial_mesh_summary = json.load(f)
    pairs = build_partial_gt_mesh_pairs(partial_mesh_summary, mesh_names, split_key)

# Stop early if matching fails
if not pairs:
    sys.exit(f"No mesh pairs found for config['{split_key}']")

# Find the best hyperparameters for shape completion using random grid search
if fast_mode: 
    # Fast mode: one-shot encoder instead of full latent optimization 
    best_cfg, trial_rows = grid_search_pointnet(pairs, model, config, mean_latent, latent_codes, device, encoder_ckpt,
                                                out_dir=OUTDIR,
                                                n_trials=N_TRIALS, valN=N_TRIAL_INF,
                                                log_path_csv=OUTDIR + "/trial_scores.csv",
                                                label_dir=LABEL_DIR if fossils else None, 
                                                ref_dir=base if fossils and perturbed else None, 
                                                snap=SNAP)
else:   
    # Encode latent via 2 stage optimization (auto-decoder framework)
    best_cfg, trial_rows = grid_search(pairs, model, config, mean_latent, latent_codes, device, 
                                        out_dir=OUTDIR,
                                        n_trials=N_TRIALS, valN=N_TRIAL_INF,
                                        log_path_csv=OUTDIR + "/trial_scores.csv",
                                        label_dir=LABEL_DIR if fossils else None, 
                                        ref_dir=base if fossils and perturbed else None, 
                                        snape=SNAP)
# Build inference subset
n_final = min(N_FINAL_INF, len(pairs))
if n_final < N_FINAL_INF:
    print(f"\033[33mOnly {len(pairs)} pairs available; using all instead of {N_FINAL_INF}\033[0m")
inf_subset = random.sample(pairs, n_final)

# Loop through meshes using best parameters from grid search
best_summary_log = []
for i, (pm_path, gt_path) in enumerate(inf_subset, start=1):    
    try:
        print(f"\033[32m\n=== Processing {os.path.basename(pm_path)} ===\033[0m")
        print(f"\033[32m\n=== {i} of {len(inf_subset)} ===\033[0m")
        # Make a new dir to save predictions
        vert_fname = pm_path
        outfpath = OUTDIR + '/best_cfg_predictions/' + os.path.splitext(os.path.basename(vert_fname))[0] # TO DO: Adjust to desired outpath
        print("Making a new directory to save model predictions and outputs at: ", outfpath)
        os.makedirs(outfpath, exist_ok=True)

        # Convert plys to vtks
        if '.ply' in vert_fname:
            ply_fname = vert_fname
            mesh, vert_fname = convert_ply_to_vtk(ply_fname, save=True)

        # Latent encoding
        if fast_mode:
            # Build the SDF dataset using all surface samples
            points, sdf_vals, sdf_dataset, sample_dict = build_sdf_dataset(vert_fname, config, n_samples=best_cfg['n_samples']) # Use all points instead of downsampling by n_samples
           
            # Fast mode: one-shot encoder instead of full latent optimization 
            print("\n-----Fast mode: encoding latent (single forward pass)----\n")
            latent_opt = encode_latent_pointnet(encoder_ckpt, points, sdf_vals, device)
            if best_cfg['iters'] > 0:
                latent_opt, _ = optimize_latent_partial(decoder=model, partial_pts=points.squeeze(), sdfs=sdf_vals, latent_dim=latent_codes.shape[1], latent_init=latent_opt, 
                                                        top_k=best_cfg['top_k'], iters=best_cfg['iters'], lr=best_cfg['lr'], lambda_reg=best_cfg['lambda_reg'], 
                                                        clamp_val=best_cfg['clamp'], latent_std=best_cfg['latent_std'], scheduler_step=best_cfg['sched_step'], 
                                                        scheduler_gamma=best_cfg['sched_gamma'], batch_inference_size=best_cfg['batch_infer'], multi_stage=True, device=device)        

        else:
            # Setup your dataset with just one mesh
            points, sdf_vals, sdf_dataset, sample_dict = build_sdf_dataset(vert_fname, config, n_samples=240)

            # Encode latent via 2 stage optimization (auto-decoder framework)
            latent_opt = encode_latent(decoder=model, points=points.squeeze(), sdf_vals=sdf_vals, latent_dim=latent_codes.shape[1], 
                                        mean_latent=mean_latent, latent_codes=latent_codes, top_k_reg=best_cfg['top_k'], latent_std=best_cfg['latent_std'],
                                        iters1=best_cfg['iters1'], iters2=best_cfg['iters2'], lr1=best_cfg['lr1'], lr2=best_cfg['lr2'], 
                                        lambda_reg1=best_cfg['lambda1'], lambda_reg2=best_cfg['lambda2'], clamp_val1=best_cfg['clamp1'], clamp_val2=best_cfg['clamp2'], 
                                        scheduler_step1=best_cfg['sched_step'], scheduler_step2=best_cfg['sched_step'], scheduler_gamma1=best_cfg['sched_gamma1'],
                                        scheduler_gamma2=best_cfg['sched_gamma2'], batch_inference_size=best_cfg['batch_infer']) 

        # Pose correction for fossils (refines latent + a small pose correction of the input points)
        pose = None
        if fossils and POSE_CORRECTION:
            latent_opt, pose = optimize_latent_pose(model, points.squeeze(), sdf_vals, latent_opt, pose_cfg, device=device)

        # Reconstruction mesh from latent
        mesh_out = reconstruct_mesh_from_latent(vert_fname, model, latent_opt, best_cfg)
        if pose is not None:
            mesh_out = undo_pose(mesh_out, pose)   # back onto the fossil, before normalize_mesh

        # Normalize and scale output using model config training params
        center, max_radius = get_norm_params(sdf_dataset, sample_dict, vert_fname)
        mesh_pv = normalize_mesh(mesh_out, vert_fname, config, center, max_radius)
        if SNAP:
            mesh_pv = snap_to_partial(mesh_pv, np.asarray(pv.read(vert_fname).points, float)[::5])
        
        # Save to file
        mesh_pv = mesh_pv.clean().triangulate()
        for arr in ['RegionId', 'vtkOriginalCellIds']:
            if arr in mesh_pv.cell_data.keys():
                mesh_pv.cell_data.remove(arr)
        base_name = os.path.splitext(os.path.basename(vert_fname))[0]
        outfname = f"{base_name}_completed.vtk"
        compl_path = os.path.join(outfpath, outfname)
        mesh_pv.save(compl_path)

        # Calculate chamfer distance between shape completed and original-ground truth mesh
        compl = pv.read(compl_path).triangulate().extract_surface(algorithm=None)
        gt = pv.read(gt_path).triangulate().extract_surface(algorithm=None)
        cd = chamfer_distance(compl, gt, n_samples=20000)
        print(f"\n{os.path.basename(pm_path)} Chamfer={cd:.4f} → {compl_path}\n")
        regions = {}
        if fossils:
            label_path = os.path.join(LABEL_DIR, os.path.basename(pm_path).replace("_hollow_cut", "").replace("_align", ""))
            regions = region_errors(compl, gt, label_path)
            print({k: round(float(v), 4) for k, v in regions.items()})
        
        if fossils and perturbed:
            ref_path = os.path.join(base, os.path.basename(pm_path).replace("_align", ""))   # unperturbed copy
            compl_shape = undo_pose_error(compl, pm_path, ref_path)
            regions["chamfer_shape"] = chamfer_distance(compl_shape, gt, n_samples=20000)
            regions.update({k + "_shape": v for k, v in region_errors(compl_shape, gt, label_path).items()})
        
        p = pose.cpu().numpy() if pose is not None else np.zeros(7)
        best_summary_log.append({'mesh': os.path.basename(pm_path),
                                 'chamfer': cd,
                                 'completed_path': compl_path,
                                 'gt_path': gt_path,
                                 'pose_rot_deg': np.degrees(np.linalg.norm(p[:3])),
                                 'pose_trans': np.linalg.norm(p[3:6]),
                                 'pose_scale': np.exp(p[6]),
                                 **regions})

    except Exception as e:
        print(f"\033[31mERROR processing {os.path.basename(pm_path)}: {e}\033[0m")
        best_summary_log.append({'mesh': os.path.basename(pm_path),
                                 'chamfer': None,
                                 'completed_path': None,
                                 'gt_path': gt_path})
        continue

summary_df = pd.DataFrame(best_summary_log)
summary_df_fpath = OUTDIR + '/best_cfg_predictions/inference_summary.csv'
summary_df.to_csv(summary_df_fpath, index=False)
print(f"\nMean Chamfer across {len(summary_df)} meshes: {summary_df['chamfer'].mean():.4f}")
print("Outputs logged to: ", summary_df_fpath)

# Copy of the summary next to the other runs, named after this run
run_name = os.path.relpath(OUTDIR, os.path.join(TRAIN_DIR, "shape_completion")).replace(os.sep, "_")
summary_df.to_csv(os.path.join(TRAIN_DIR, "shape_completion", f"inference_summary_{run_name}.csv"), index=False)

# Relate alignment error to reconstruction error (perturbed + aligned fossils only)
if fossils and PERTURB_LOG:
    log = pd.read_csv(PERTURB_LOG)
    if MW_LOG:   # MorphoWeave stages: swap in MorphoWeave's measured alignment error
        mw = pd.read_csv(MW_LOG)
        mw = mw[mw["mode"] == MW_MODE][["fossil", "rms_err_pct"]]
        log = log.drop(columns=["rms_err_pct", "rot_err_deg", "scale_err_pct"]).merge(mw, on="fossil", how="left")
    log["key"] = log.fossil.str.replace("_hollow_cut", "").str.replace(".vtk", "")
    summary_df["key"] = summary_df.mesh.str.replace("_hollow_cut", "").str.replace("_align", "").str.replace(".vtk", "")
    merged = summary_df.merge(log, on="key")
    merged.to_csv(OUTDIR + "/best_cfg_predictions/inference_vs_alignment.csv", index=False)
    if stage == "aligned":
        cols = ["rms_err_pct", "rot_err_deg", "scale_err_pct", "n_used"]
    elif MW_LOG:
        cols = ["rms_err_pct", "n_used"]
    else:
        cols = ["rot_deg", "scale", "shift_pct"]

    corr = pd.DataFrame({"chamfer_intact": merged[cols].corrwith(merged.chamfer_intact, method="spearman"),
                         "chamfer_intact_shape": merged[cols].corrwith(merged.chamfer_intact_shape, method="spearman")})
    print(f"\n\n\n\033[32m{corr.round(3).to_string()}\033[0m\n\n\n")
    merged.to_csv(os.path.join(TRAIN_DIR, "shape_completion", f"inference_vs_alignment_{run_name}.csv"), index=False)

sys.stdout.flush()
sys.stderr.flush()
os._exit(0)