## Paper analysis and figure code
Code for all figures and tables in the manuscript. Each script/notebook reads
processed data and writes outputs into a designated directory.

## Requirements

Run these analyses in an environment with the `NSM` package and its dependencies installed, as described in the repository-level `README.md`. 

Command-line scripts include a **Run** block below. Notebooks are run top to
bottom after editing the configuration cell at the top (paths, `TRAIN_DIR`,
`CKPT`, output directory).

## At a Glance

**Manuscript figures and tables**
| File | Manuscript output |
| --- | --- |
| `shape_completion_table.py` | Table 1 ; Table S9|
| `classification_tables.py` | Table 2; Tables S3–S4 ; Table S10|
| `life_hist_grid_fig.ipynb` | Figure 1 |
| `network_architecture_icons.ipynb` | Figure 2 schematic elements |
| `conf_matr_schaema.ipynb` | Figure 3 schematic element |
| `classify_vertebrae_inference_fig.py` | Figure 3 schematic element |
| `shape_completion_figures.py` | Figure 4 |
| `classification_figures.py` | Figure 5 |
| `PCA_tSNE_UMAP_paper_figs.ipynb` | Figure 6 |
| `LDA_paper_fig.ipynb` | Figure 7 ; Table S8 |
| `GMM_spearman_mesh_grid.ipynb` | Figures 8 and 9 ; Table S5 |
| `morphol_disparity_lms_latents.ipynb` | Figures 10 and S2 ; Tables S6-S7 |
| `latent_interp_figs.ipynb` | Figures 11 and S3 |
| `downsample_figure.ipynb` | Figure S1 |
| `build_downsample_val_ds.py` | Tables S9-S10 |
| `run_v72_split_downsample_exp.sh` | File S3 |
| `../lizard_species_list.csv` | Table S1 |

**Interactive 3D renderings, plots, and videos** on the [vert-nsm-figs GitHub Pages site](https://3d-fossils-haag.github.io/vert-nsm-figs/)

| File | Interactive output |
| --- | --- |
| `render_landmarks.ipynb` | Atlas 3D viewers: `mean_landmarks.html`, `dense_correspondences.html`, `lollipop_vecs.html` |
| `../inspect_data/inspect_meshes_video.py` | Mesh inspection video: `inspect_meshes_4panel.mp4` |
| `../data_viz/pc_video_4way.py` | PC1–PC5 traversal videos (±1.5 SD): `pc{1..5}_4way_256p_1.5xpc.mp4` |
| `PCA_tSNE_UMAP_paper_figs.ipynb` | Per-representation embeddings (NSM latents, 28 sparse landmarks, 4,943 dense correspondences): `run_v72_{pca_1v2,pca_3v4,tSNE,UMAP}.html`, `run_v72_*_28lmks.html`, `run_v72_*_4943lmks.html`<br>Comparison panels by broad taxon: `run_v72_{pca_1v2,pca_3v4,tsne,umap}_comparison.html`<br>Comparison panels by life history: `run_v72_{pca_1v2,pca_3v4,tsne,umap}_comparison_trait.html` |


## Analysis Files

---
### `shape_completion_table.py`

**Produces:** Table 1

Computes mean ± SD and median [IQR] Chamfer distance for five shape-completion models across train, validation, and test splits.

**Inputs**
Chamfer CSVs from `shape_completion_eval.py` for the five models (baseline, encoder, encoder + refinement, hierarchy, contrastive), one per split(`{split}` = `train`, `val`, `test`):
- `shape_completion_eval/v72_{split}_chamfer.csv`
- `shape_completion_eval/v72_encoder_{split}_chamfer.csv`
- `shape_completion_eval/v72_encoder_{split}_chamfer_refine.csv`
- `shape_completion_eval/v73h_{split}_chamfer.csv`
- `shape_completion_eval/v73c_{split}_chamfer.csv`

**Output:** `Table_1_shp_compl_eval.csv`

**Run**

    python shape_completion_table.py --datadir shape_completion_eval/ --outdir shape_completion_eval/
---

### `shape_completion_figures.py`

**Produces:** Figure 4 (A–C)

Generates Chamfer-distance distributions, a train/validation/test generalization-gap plot, and an accuracy–speed tradeoff plot for the Baseline, Encoder, Encoder + refinement, Hierarchy, and Contrastive models.

**Inputs:** The same Chamfer-distance CSV files used by `shape_completion_table.py`.

**Outputs**

- `fig_chamfer_boxplot.png` — Chamfer-distance distributions
- `fig_generalization_gap.png` — Generalization across dataset splits
- `fig_chamfer_tradeoff.png` — Accuracy–speed tradeoff

**Run**

    python shape_completion_figures.py --datadir shape_completion_eval/ --outdir shape_completion_eval/

---

### `classification_tables.py`

**Produces:** Table 2 and Tables S3–S5

Reads evaluation results from `classification_eval.py` and summarizes Top-1 and Top-5 classification performance for genus and normalized spinal position (20% bins) under leave-one-vertebra-out (LOO) and leave-one-specimen-out (LOSO) masking. It also produces per-class recall and sample-size correlation tables.

**Inputs**
Prediction metrics CSVs/JSONs from `classification_eval.py` for every model passed to `--roots` (run_v72, run_v73h, run_v73c), and every condition evaluated (`{split}` = `train`, `val`, `test`; `{eval_level}` = `loo`, `specimen`, `species`, `genus`; `{latents}` = `base` or `latent_opt`):

- `<run>/classification/evaluation/<split>/<split>_<eval_level>_<latents>/predictions.csv`
- `<run>/classification/evaluation/<split>/<split>_<eval_level>_<latents>/metrics_summary.csv`
- `<run>/classification/evaluation/<split>/<split>_<eval_level>_<latents>/metrics.json`

**Outputs**
- `Table_2_classif_eval.csv` — Top-1 and Top-5 accuracy for `genus` and `position_20`
- `Table_S3_classification_stats.csv` — metrics across runs, splits, masking schemes, latent conditions, and levels
- `Table_S4_classif_by_spec_num.csv` — per-class recall and sample size
- `Table_S5_classif_by_spec_spearmans.csv` — correlations between performance and support

**Run**

    python classification_tables.py --roots ../run_v72 ../run_v73h ../run_v73c --outdir classification/evaluation/paper/

---

### `classification_figures.py`

**Produces:** Figure 5 (A–D)

Renders Top-5 LOSO confusion matrices for normalized spinal position, life-history strategy, broad taxonomic group, and spinal region. The default configuration uses optimized latents, the training split, specimen-level masking, and Top-5 recall on the diagonal.

**Input**
Requires the following outputs from first running classification_eval.py: 
- `<run>/classification/evaluation/train/train_specimen_latent_opt/predictions.csv`

**Outputs**

- `fig_r_A_position_20.png`
- `fig_r_B_life_history.png`
- `fig_r_C_broad_taxon.png`
- `fig_r_D_region.png`
- `fig_r_colorbar.png`

**Run**

    python classification_figures.py --roots ../run_v72 --outdir classification/evaluation/paper/

---

### `life_hist_grid_fig.ipynb`

**Produces:** Figure 1

Renders exemplar vertebrae for six life-history strategies—arboreal, burrowing, grass-swimmer, saxicolous, terrestrial, and snake—in lateral and posterior views. Sparse landmark locations are overlaid as red spheres and renders are assembled into a 4 × 3 panel.

**Inputs**

- VTK mesh files specified in the model configuration
- Sparse landmark files in `LM_DIR` (`.mrk.json`)
- `lizard_species_list.csv`
- Model configuration and checkpoint

**Output:** `fig_f_exemplar_lifehist_vert.png` (configured as `OUT_PANEL`)

---

### `PCA_tSNE_UMAP_paper_figs.ipynb`

**Produces:** Figure 6

Computes and visualizes PCA (PC1–PC2 and PC3–PC4), t-SNE, and UMAP for sparse landmarks, dense correspondences, and NSM latent codes. Points are colored by broad taxonomic group; coordinates are also exported for downstream analyses.

**Inputs**

- VTK mesh files
- Sparse landmark files (`LM_DIR`)
- Dense correspondence files (`DENSE_LM_DIR`)
- NSM latent codes from the checkpoint in `TRAIN_DIR`
- `lizard_species_list.csv`

**Outputs**

- `{RUN}_pca_1v2_comparison.png`
- `{RUN}_pca_3v4_comparison.png`
- `{RUN}_tsne_comparison.png`
- `{RUN}_umap_comparison.png`
- Combined publication panel (static `.png` and interactive `.html`)
- `*_points_for_stats.csv` files containing PCA, t-SNE, and UMAP coordinates

---

### `LDA_paper_fig.ipynb`

**Produces:** Figure 7, Table S8

Applies linear discriminant analysis to sparse landmarks, dense correspondences, and NSM latents at 90%, 95%, and 99% variance thresholds. Produces a 3 × 3 life-history visualization grid and calculates MANOVA statistics.

**Inputs**

- VTK mesh files
- Sparse-landmark, dense-correspondence, and NSM-latent representations
- `lizard_species_list.csv`

**Outputs**

- `{RUN}_lda_trait_3x3_thresholds.png`
- `{RUN}_lda_trait_accuracy_table.csv`

---

### `GMM_spearman_mesh_grid.ipynb`

**Produces:** Figures K and M

Computes pairwise Spearman rank correlations between principal-component scores from sparse landmarks, dense correspondences, and NSM latents. It also generates PC1–PC4 traversal grids for landmark-based warps and NSM reconstructions, shown in front and side views.

**Inputs**

- VTK mesh files
- Sparse landmark files (`.mrk.json`)
- NSM model checkpoint and latent codes
- PC-score exports from `PCA_tSNE_UMAP_paper_figs.ipynb`
- Outputs from `save_pc_snapshots.py` and `pc_snapshot_grid.py` for NSM traversals

**Outputs**

- `spearman_pc_correlations.png`
- Spearman correlation tables (CSV)
- Landmark and NSM PC traversal PNGs
- Combined traversal panel

---

### `morphol_disparity_lms_latents.ipynb`

**Produces:** Figure 10 and Figure S1

Compares morphological disparity across lizard families and life-history categories using sparse landmarks and NSM latents reduced to 84 principal components. Ranks are compared between representations with dumbbell plots.

**Inputs**

- VTK mesh files
- Sparse landmark files in `LM_DIR`
- Atlas mean landmarks (`MEAN_LMS_FN`)
- NSM latent codes from the model checkpoint
- `lizard_species_list.csv`

**Outputs**

- `disparity_shift_by_trait_specimen.csv`
- `disparity_dumbbell_shift_by_trait.png` — family-level comparison
- `disparity_dumbbell_shift_by_trait_specimen.png` — life-history comparison

---

### `latent_interp_figs.ipynb`

**Produces:** Figure 11 and Figure S2

Interpolates linearly between mean latent codes or PointNet-encoded mesh endpoints, decodes each interpolation step to a surface mesh, renders lateral and posterior views, and assembles the images into a figure. Absolute mesh paths can be used as endpoints to encode fossil specimens.

**Inputs**

- NSM model checkpoint (`TRAIN_DIR`, `CKPT`)
- VTK or PLY mesh files, or training-set latent-code selections

**Outputs**

- Stitched interpolation figure in the configured `OUT_DIR`
- Per-step renderings in `OUT_DIR/row{N}_{lbl_a}_to_{lbl_b}/`

**Run:** Set `PAIRS`, `TRAIN_DIR`, `CKPT`, and `N_STEPS` in the configuration cell; then run `run_render()` followed by `run_stitch()`.

---

### `conf_matr_schaema.ipynb`

**Produces:** Figure 3 schematic element

Creates a synthetic five-class, row-normalized confusion-matrix schematic for use in a methods or inference illustration.

**Input:** None; the synthetic matrix is generated within the notebook.

**Output:** `conf_matrix_schematic.png`

---

### `classify_vertebrae_inference_fig.py`

**Produces:** Figure 3 inference panel(s)

Encodes a novel vertebral mesh into NSM latent space through latent optimization, identifies the five nearest training latents using cosine similarity, and displays the query and nearest neighbors in PCA or t-SNE latent space. Rendered with large points to be optimized for a small schaematic panel of a figure.

**Inputs**

- Novel mesh (`.vtk` or `.ply`)
- NSM model checkpoint and training latent codes
- Model configuration JSON

**Outputs**

- `latent_space_pca_pca_regularized_95pct_cos.png`
- `latent_space_tsne_pca_regularized_95pct_cos.png`
- `similar_meshes_pca_regularized_95pct_cos.txt`

---

### `network_architecture_icons.ipynb`

**Produces:** Figure 2 schematic elements

Generates the triplanar, CNN, MLP, and loss icons used to assemble the network architecture diagram.

**Input:** None; the elements are generated within the notebook.

**Output:** 
- `svg_pieces/10_scene_3d.png`
- `svg_pieces/11_faceon.png`
- `svg_pieces/13_triplanar_corner.png`
- `svg_pieces/cnn_blocks.svg`
- `svg_pieces/dense_blocks.svg`
- `svg_pieces/global_z.svg`
- `svg_pieces/loss.svg`
- `svg_pieces/mlp_blocks.svg`
- `svg_pieces/plane_features.svg`
- `svg_pieces/reshape_2x2.svg`
- `svg_pieces/triplanar.svg`

---

### `run_v72_split_downsample_exp.sh`

**Produces:** File S3

Writes the full-size (95-specimen) split as a JSON file in the same format as the downsampled splits, so that the complete model and the reduced-sampling models are trained and evaluated through an identical code path. Reads the split already resolved in the `run_v72` config and recovers the specimen behind each mesh by matching normalized basenames against the specimen column of the species list.

**Inputs**

- `../run_v72/model_params_config.json` (`list_mesh_paths`, `val_paths`, `test_paths`)
- `../lizard_species_list.csv`

**Output:** `splits/downsample_95spec.json` — train, validation, and test mesh paths plus the contributing specimen list; specimen and mesh counts are printed on completion.

**Run**

    bash run_v72_split_downsample_exp.sh

---

### `build_downsampled_datasets.py`

**Produces:** Tables S9-S10

Builds taxon-stratified downsampled datasets (10/30/50/70 specimens) for the sampling-effort experiments, to show how performance degrades as sample size decreases.

**Input:** folder used to train the final models (vertebrae_meshes/*)

**Output:** 
- `splits/downsample_10spec.json`
- `splits/downsample_30spec.json`
- `splits/downsample_50spec.json`
- `splits/downsample_70spec.json`
- 
Pass a split file to training with `python train_model.py --splits splits/downsample_10spec.json`.

---

### `build_downsample_val_ds.py`

**Produces:** Tables S9-S10

Assembles the common evaluation set for the sampling-effort experiment. Each downsampled run has its own split, so comparing them fairly requires scoring every model on the same meshes, none of which any model has seen. Takes every mesh across all runs and keeps those absent from all of their training sets, keyed by basename because runs refer to the same mesh by different absolute paths.

**Input**
Resolved configurations for every run passed to `--roots` (run_10spec, run_30spec, run_50spec, run_70spec, run_v72):

- `<run>/model_params_config.json`

**Output:** `../common_eval_meshes.json` — mesh paths held out by every run, written beside the runs so that `shape_completion_eval.py`, `encode_latents_for_eval.py`, and `classification_eval.py` resolve it from `NSM/nsm`. A per-run summary of training and held-out counts and the specimen composition of the query set are printed on completion.

**Run**

    python build_downsample_val_ds.py --roots ../run_10spec ../run_30spec ../run_50spec ../run_70spec ../run_v72

---

### `downsample_figure.ipynb`

**Produces:** Figure S1

Plots shape-completion error and classification accuracy against training-set size for the five sampling-effort models (10, 30, 50, 70, and 95 specimens). Every model is scored on the shared held-out meshes from `build_downsample_val_ds.py`, so the curves differ only in how many specimens each model was trained on. Chamfer distances are summarized as median and IQR because the distributions are right-skewed, and each run is compared with the full training set, and with the next step in the series, using paired Wilcoxon signed-rank tests with Holm correction applied within each family of tests. Only reachable classification metrics are plotted, since under specimen masking a class with a single specimen has no same-label gallery entry remaining.

**Inputs**

- `<run>/shape_completion/evaluation/2phase_val_common_chamfer.csv` for each run in `RUNS`, produced by `shape_completion_eval.py` with `downsample_experiment = True`
- `../downsample_tables/Table_S3_classification_stats.csv`, produced by `classification_tables.py` on the common validation meshes:

      python classification_tables.py --roots ../run_10spec ../run_30spec ../run_50spec ../run_70spec ../run_v72 \
          --table-splits downsample_val_common --no-supplement \
          --name Table_downsample_classif.csv --outdir ../downsample_tables

**Outputs** (written to `OUTDIR`, by default `../downsample_tables/`)

- `Fig_downsample_levels.png` — Chamfer distance and top-1/top-5 LOO and LOSO accuracy against training-set size
- `Data_S_shape_completion_chamfer_raw.csv` — per-mesh Chamfer distances for every run
- `downsample_stats.txt` — Chamfer descriptives, paired Wilcoxon results, and the reachable-accuracy table

---

## Notes

- Model weights and configuration files for reported models: https://huggingface.co/BioVisionLab/models.
- Final training dataset (`vertebrae_meshes.zip`), along with meshes, sparse landmarks, dense correspondences, and the atlas SSM produced by MorphoWeave in 3D Slicer: https://huggingface.co/datasets/BioVisionLab/train_data/tree/main
- Interactive 3D renderings are available at https://3d-fossils-haag.github.io/vert-nsm-figs/
