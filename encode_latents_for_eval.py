"""
Bulk encode a dataset split into latent space for classification evaluation.

Each mesh gets its own latent by optimization against the trained decoder
(auto-decoder inference), rather than reusing the latent codes learned during
training. Writes <output_dir>/latent_codes_<tag or dataset_split>.pth, a
[N, latent_size] tensor whose row order matches the mesh list it encoded, for
classification_eval.py --encoded_latents to read.

Usage
-----
Main analysis (run per model: run_v72, run_v73h, run_v73c):

python encode_latents_for_eval.py --model_root run_v72 --ckpt 2500 --output_dir classification/evaluation/encoded_latents --dataset_split train
python encode_latents_for_eval.py --model_root run_v72 --ckpt 2500 --output_dir classification/evaluation/encoded_latents --dataset_split val
python encode_latents_for_eval.py --model_root run_v72 --ckpt 2500 --output_dir classification/evaluation/encoded_latents --dataset_split test

Downsample experiment, common validation meshes from build_downsample_val_ds.py
(run per model: run_10spec, run_30spec, run_50spec, run_70spec, run_v72):

 python encode_latents_for_eval.py --model_root run_10spec --ckpt 3000 --output_dir classification/evaluation/encoded_latents --dataset_split val --mesh_list common_eval_meshes.json --tag downsample_val_common

Note: 
Optimization length and step size are --iterations (default 1000) and
--learning_rate (default 1e-3); keep them identical across models being compared.
Pass classification_eval.py the same --dataset_split and --tag so it finds the
file and lists the queries in the same order.
"""
import argparse
import os
import sys
import numpy as np
import json

import torch
import pymskt.mesh.meshes as meshes
import pymskt.mesh.meshTools as meshTools

from NSM.helper_funcs import fixed_point_coords, load_config, load_model_and_latents, safe_load_mesh_scalars, convert_ply_to_vtk
from NSM.optimization import get_top_k_pcs, optimize_latent, build_sdf_dataset

# Monkey Patch into pymskt.mesh.meshes.Mesh
meshes.Mesh.load_mesh_scalars = safe_load_mesh_scalars
meshes.Mesh.point_coords = property(fixed_point_coords)

def resolve_model_root(root_dir, ckpt):
    paths = {"model_params_config.json": os.path.join(root_dir, "model_params_config.json"),
             "model":                    os.path.join(root_dir, "model", f"{ckpt}.pth"),
             "latent_codes":             os.path.join(root_dir, "latent_codes", f"{ckpt}.pth")}
    missing = [k for k, v in paths.items() if not os.path.isfile(v)]
    if missing:
        raise ValueError(f"{root_dir}: missing {missing} (ckpt {ckpt})")
    return tuple(paths.values())

def _load_model_bundle(args, device):
    config_path, model_path, latent_path = resolve_model_root(args.model_root, args.ckpt)
    config = load_config(config_path)
    print("Classification model root: {}".format(args.model_root))
    print("Classification config: {}".format(config_path))
    print("Classification model: {}".format(model_path))
    print("Classification latent codes: {}".format(latent_path))
    print("Classification device: {}".format(device))
    print("Classification iterations: {}".format(args.iterations))
    
    model, _, latent_codes = load_model_and_latents(model_path, latent_path, config, device)
    mean_latent = latent_codes.mean(dim=0, keepdim=True)
    _, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.99)
    
    return config, model, latent_codes, mean_latent, top_k_reg

def _encode_split(split_name, mesh_paths, config, model, latent_codes, mean_latent, top_k_reg, device, args):
    if not mesh_paths:
        print("No meshes found for split: {}".format(split_name))
        return

    print("\n--- Encoding {} split ({} meshes) ---".format(split_name, len(mesh_paths)))
    optimized_latents = []

    for i, mesh_path in enumerate(mesh_paths, start=1):
        base = os.path.splitext(os.path.basename(mesh_path))[0]
        print(f"\033[32m\n=== Optimizing {base} ===\033[0m")
        print(f"\033[32m\n=== {i} of {len(mesh_paths)} ===\033[0m")
        
        # 1. Prepare Mesh
        _, prepared_path = convert_ply_to_vtk(mesh_path)
        
        # 2. Build Dataset (exactly as done in the baseline)
        points, sdf_vals, _, _ = build_sdf_dataset(prepared_path, config, config["n_pts_per_object"])
        
        # 3. Optimize Latent
        latent = optimize_latent(model, points.squeeze(), sdf_vals, config["latent_size"], 
                                 top_k_reg, mean_latent, latent_codes, 
                                 iters=args.iterations, lr=args.learning_rate, device=device)  
        optimized_latents.append(latent.detach().cpu())

    # Stack into [N, latent_size]
    stacked_latents = torch.stack(optimized_latents)
    if stacked_latents.dim() == 3 and stacked_latents.shape[1] == 1:
        stacked_latents = stacked_latents.squeeze(1)

    out_file = os.path.join(args.model_root, args.output_dir, "latent_codes_{}.pth".format(split_name))
    torch.save(stacked_latents, out_file)
    print("Saved {} latents to: {}".format(split_name, out_file))

def encode_datasets(args):
    repository_root = os.path.dirname(os.path.abspath(__file__))
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    
    os.makedirs(os.path.join(args.model_root, args.output_dir), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config, model, latent_codes, mean_latent, top_k_reg = _load_model_bundle(args, device)

    # Encode files fromd dataset split (train, val, test)
    ds_split_keys = {"train": "list_mesh_paths", "val": "val_paths", "test": "test_paths"}
    split_key = ds_split_keys[args.dataset_split]
    ds_paths = json.load(open(args.mesh_list)) if args.mesh_list else config[split_key]
    _encode_split(args.tag or args.dataset_split, ds_paths, config, model, latent_codes, mean_latent, top_k_reg, device, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode train, val, and test sets into latents for evaluation.")
    parser.add_argument("--model_root", required=True, help="Path to the model directory (e.g. run_vXX)")
    parser.add_argument("--ckpt", required=True, help="Numeric checkpoint (Ex: 3000)")
    parser.add_argument("--output_dir", required=True, help="Directory to save latent_codes_{train,val,test}.pth (e.g. classification/evaluation/encoded_latents)")
    parser.add_argument("--dataset_split", choices=["train", "val", "test"])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--mesh_list", default=None,
                        help="JSON list of mesh paths to encode instead of the split")
    parser.add_argument("--tag", default=None,
                        help="output name, latent_codes_<tag>.pth (default: dataset split)")
    args = parser.parse_args()
    
    encode_datasets(args)