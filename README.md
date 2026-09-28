# Vert-NSM
Neural shape modelling for biologically-informed shape completion, classification, and geometric morphometric analysis of vertebrate specimens. 

# Introduction
This code uses generative deep learning models to understand the skeletal anatomy of lizards and some snakes (Squamata). This code is forked and modified from [gattia/NSM](https://github.com/gattia/NSM) following the terms of the [GNU Affero GPL 3.0 License](https://www.gnu.org/licenses/agpl-3.0.en.html). See [Original NSM Documentation](http://anthonygattiphd.com/NSM/). 

![Isomap GIF](https://github.com/aubricot/nsm/blob/main/images/isomap_4way_splitscreen_C-T-L_avg.gif)
*Figure 1: Traversing an isomap of the NSM trained model latent space using travelling salesman and k-nearest neighbors. Video animation made using [isomap_video.py](https://github.com/aubricot/nsm/blob/main/isomap_video.py)*

# Installation

```bash
# Create and activate conda environment
conda create -n NSM python=3.10
conda activate NSM

# Install pytorch and dependencies
conda install pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -c conda-forge -c defaults

# Install NSM
mkdir NSM
cd NSM
git clone https://github.com/3D-fossils-Haag/nsm.git
cd nsm
python -m pip install -r requirements.txt
pip install -e .

```
# Demos
Check out our demos to build NSM fully in the Google Colab runtime environment and interactively evaluate our tools with demo data in under 10 minutes, no need to connect to your Google Drive!

:arrow_right: :lizard: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/nsm/blob/main/demos/classification_demo.ipynb) Click here to try out classification of unknown fossils.


:arrow_right: :lizard: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aubricot/nsm/blob/main/demos/shape_completion_demo.ipynb) Click here to try out shape completion for partial fossils.

:arrow_right: :mirror_ball: Interactively explore our latent space and dataset on [3d-fossils-haag.github.io/vert-nsm-figs/](https://3d-fossils-haag.github.io/vert-nsm-figs/).


# Example Usage
Please refer to the project [Wiki](https://github.com/3D-fossils-Haag/nsm/wiki) for detailed instructions on using NSM.

## Training

1. Put your `.vtk` meshes in a single folder (ex: `vertebrae_meshes`).
2. In [train_model.py](), update the lines marked `# TO DO:` — the mesh folder
   (`folder_vtk`) and the train/val/test split fractions.
3. Adjust hyperparameters in [vertebrae_config.json]().
4. Train:

```bash
conda activate NSM
cd NSM/nsm
python train_model.py --run_name run_v1
```

Meshes are split 80/15/5 with a fixed seed (42), so runs are reproducible.

**Loss options**
- No flag — standard NSM loss.
- `--contrastive_loss` — contrastive NSM loss. Set `contrastive_weight` in the config (0.01 recommended).
- `--hierarchy_loss` — hierarchy-aware NSM loss. Set `hierarchy_weight` (0.01 recommended); `hierarchy_warmup` and `hierarchy_margins` are optional.

To reproduce paper analysis on downsampled dataset sizes, pass `--splits splits.json` to reuse precomputed splits (from `paper/build_downsampled_datasets.py`) instead of the random split.

**Outputs** — written to `{run_name}/`:

```
run_v1/
├── model/                      # checkpoints (.pth)
├── latent_codes/               # latent codes (.pth)
└── model_params_config.json    # config that includes parameters and data split filenames

./nsm_sdf_cache/run_v1/         # SDF samples, reused when load_cache is true
```
## Inference

### 1. Load a trained model

A finished training run looks like this:

```python
from NSM.helper_funcs import load_config, load_model_and_latents,

# Define checkpoint and paths
CKPT = '2500' # TO DO: Choose the ckpt value you want to analyze results for
LC_PATH = 'latent_codes' + '/' + CKPT + '.pth'
MODEL_PATH = 'model' + '/' + CKPT + '.pth'

# Load model config
config = load_config(config_path='model_params_config.json')
device = config.get("device", "cuda:0")

# Load model and latent codes
model, latent_ckpt, latent_codes = load_model_and_latents(MODEL_PATH, LC_PATH, config, device)

```

### 2. Generate a mesh from a latent

Any latent vector can be decoded into a surface. Here we decode the mean of the training latents — the average shape the model learned.

```python
from NSM.optimization import reconstruct_mesh_from_latent
import pyvista as pv

# Get the mean of the latent codes
mean_latent = latent_codes.mean(dim=0, keepdim=True)

# Reconstruction mesh from latent
mesh_out = reconstruct_mesh_from_latent(vert_fname=None, model=model, latent_opt=mean_latent.to(device), config=config)

# Save mesh
mesh_pv = pv.wrap(mesh_out.mesh).clean().triangulate().extract_surface(algorithm=None)
mesh_pv.save("mean_shape.vtk")
```

### 3. Complete a partial mesh

Encode a partial scan into latent space, then decode a complete surface.
There are two ways to get the latent:

| Method | Speed | Use when |
|---|---|---|
| PointNet encoder | seconds | you have a trained encoder and non-noisy data |
| 2-phase optimization | minutes | you want the best reconstruction accuracy for noisy data (ex: fossils) |

```python
from NSM.helper_funcs import convert_ply_to_vtk
from NSM.optimization import (build_sdf_dataset, encode_latent, encode_latent_pointnet, optimize_latent_partial,
     reconstruct_mesh_from_latent, get_norm_params, normalize_mesh, get_top_k_pcs)

# Define input filename
vert_fname = "partial_mesh.vtk"
if ".ply" in vert_fname:
     _, vert_fname = convert_ply_to_vtk(path) for .ply

# Get vals for optimization
_, top_k_reg = get_top_k_pcs(latent_codes, threshold=0.99)
latent_std = latent_codes.std().mean()
```

**Option A — PointNet encoder (fast)**

```python
points, sdf_vals, sdf_dataset, sample_dict = build_sdf_dataset(
    vert_fname, config, n_samples=None)   # None = use all surface samples

latent_opt = encode_latent_pointnet(
    f"{TRAIN_DIR}/encoder/checkpoints/encoder.pt", points, sdf_vals, device)

# Optional, but recommended: a few refinement iterations sharpen the encoder's guess
latent_opt, _ = optimize_latent_partial(
    model, points.squeeze(), sdf_vals, config["latent_size"],
    latent_init=latent_opt, top_k=top_k_reg,
    iters=200, lr=1e-3, lambda_reg=1e-6, clamp_val=None, latent_std=latent_std,
    scheduler_step=300, scheduler_gamma=0.9,
    batch_inference_size=32768, multi_stage=True, device=device)
```

**Option B — 2-phase latent optimization (accurate)**

Phase 1 finds the right neighbourhood of latent space; phase 2 refines local surface detail.

```python
points, sdf_vals, sdf_dataset, sample_dict = build_sdf_dataset(vert_fname, config, n_samples=240)

latent_opt = encode_latent(
    decoder=model, points=points.squeeze(), sdf_vals=sdf_vals,
    latent_dim=latent_codes.shape[1], mean_latent=mean_latent, latent_codes=latent_codes,
    top_k_reg=top_k_reg, latent_std=latent_std,
    iters1=5000, iters2=8000, lr1=1e-4, lr2=1e-4,
    lambda_reg1=1e-2, lambda_reg2=1e-7, clamp_val1=1, clamp_val2=None,
    scheduler_step1=800, scheduler_step2=800,
    scheduler_gamma1=0.7, scheduler_gamma2=0.9,
    batch_inference_size=32768)
```

**Decode and save**

`normalize_mesh` puts the output back into the input mesh's original position and scale.

```python
mesh_out = reconstruct_mesh_from_latent(vert_fname, model, latent_opt, config)

center, max_radius = get_norm_params(sdf_dataset, sample_dict, vert_fname)
mesh_pv = normalize_mesh(mesh_out, vert_fname, config, center, max_radius)
mesh_pv = mesh_pv.clean().triangulate().extract_surface(algorithm=None)
mesh_pv.save("completed.vtk")
```

> The optimization values above are examples. Use
> `shape_completion_grid_search.py` to tune them for your model, and
> `shape_completion.py` to run completion over a whole directory of meshes.

## Evaluation

Both scripts run against a trained model directory (e.g. `run_v72/`) and the
splits saved in its `model_params_config.json`.

### Shape completion

Scored as Chamfer distance between the completed mesh and its ground truth.

**1. Divide meshes into N segments using MorphoWeave in 3D Slicer**  
See directions in the project Wiki ([Evaluation using synthetically fragmented meshes](https://github.com/3D-fossils-Haag/nsm/wiki/Evaluation:-shape-completion-using-synthetically-fragmented-meshes)). 

**2. Make partial meshes** — subtracts segment PLYs from the ground truth meshes:

```bash
python create_partial_meshes.py ground_truth/ segments/ run_v72/shape_completion/meshes --seg_ids 6
```

**3. Tune encoding parameters** — random search on the val split. Update the
`# TO DO:` variables, then:

```bash
python shape_completion_grid_search.py
```

**4. Score a split** — update the `# TO DO:` variables (`TRAIN_DIR`, `CKPT`,
`split`, `fast_mode`), then:

```bash
python shape_completion_eval.py
```

Per-mesh scores are written to
`run_v72/shape_completion/evaluation/{encoder|2phase}_{split}_chamfer.csv`.

### Classification

Nearest-neighbour retrieval in latent space: each query vertebra takes the labels
of the most similar training vertebra (cosine). Labels come from
`--species_list` (default `lizard_species_list.csv`).

`--eval_level` sets what is hidden from the gallery when scoring a query:

- `loo` — the vertebra itself (optimistic upper bound)
- `specimen` — the whole individual (the realistic test)
- `species` / `genus` — every specimen of that species or genus

**Train split:**

```bash
python classification_eval.py --model_root run_v72 --ckpt 2500 \
    --dataset_split train --eval_level specimen
```

**Val or test split** — encode the query latents first:

```bash
python encode_latents_for_eval.py --model_root run_v72 \
    --output_dir classification/evaluation/encoded_latents --dataset_split val

python classification_eval.py --model_root run_v72 --ckpt 2500 \
    --dataset_split val --encoded_latents --eval_level specimen
```

Outputs go to `run_v72/classification/evaluation/{split}/{level}_{base|latent_opt}/`:
`metrics_summary.csv`, `metrics.json`, `predictions.csv`, and a per-class report
and confusion matrix for each category (family, genus, species, broad taxon,
region, position, life history).

# License

This code is forked and modified from [https://github.com/gattia/NSM](https://github.com/gattia/NSM) following the terms of the [GNU Affero GPL 3.0 License](https://www.gnu.org/licenses/agpl-3.0.en.html) and [NSM License](https://github.com/gattia/nsm/blob/main/LICENSE). See [NOTICE](https://github.com/3D-fossils-Haag/nsm/blob/main/NOTICE).

## Citation
If you use this code or the trained models in your research, please cite this repository
```
Wolcott et al. 2026. “Vert-NSM” GitHub repository. https://github.com/3D-fossils-Haag/vert-nsm (accessed YYYY-MM-DD).
```
