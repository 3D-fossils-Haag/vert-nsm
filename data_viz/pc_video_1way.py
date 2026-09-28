# Single-view PC traversal video, with a PC-score slider overlay.
#
# Same per-frame pipeline as build_warp_grid (warp_mesh -> pv_to_o3d -> render_cameras
# -> crop_top_right), just swept continuously and written to mp4 instead of tiled.
#
#   MODE = "landmarks" -> TPS-warp the atlas mesh along a sparse-landmark PC
#   MODE = "latents"   -> decode NSM latents along a latent PC (as in pc_video_4way.py)

import os, gc
import numpy as np
import cv2
import pyvista as pv
from pathlib import Path

from NSM.helper_funcs import (NumpyTransform, pv_to_o3d, load_config, render_cameras,
                              load_model_and_latents, generate_and_render_mesh)
from NSM.morphometrics import gm_prcomp, pc_shape
from NSM.plotting import (load_mrk_json, view_rotation, make_renderers, make_material,
                          warp_mesh, crop_top_right)
from NSM.traverse_latents import generate_latent_path_plot

# ── Config ───────────────────────────────────────────────────────────────────
MODE     = "landmarks"                 # TO DO: "landmarks" or "latents"
RUN      = "run_v72"
PC_idx   = 0                           # 0-based (PC1 -> 0)
AMPLIFY  = 1.5
FLIP_PC  = False                       # match the FLIP_PCS convention in the grid notebook

VIEW          = "side"
VIEW_ROT_DEG  = {"side": 13.0, "front": 90.0 + 13.0}
WIDTH, HEIGHT = 640, 480
BG_COLOR      = [0.839, 1.0, 0.996]    # aquamarine, as in the grid figures
N_SEG         = 45                     # frames per leg (low -> 0 -> high -> 0 -> low)
FPS           = 15
SLIDER_W = 300          # PC slider width as a fraction of the video frame

# landmarks mode
ATLAS_RUN    = "2026_07-15_13_06_22/"
DROPBOX_ROOT = Path("/home/k.wolcott/UFL Dropbox/Katherine Wolcott/"
                    "neural_shape_models/final_dataset_aug26/atlas/")
# latents mode
CKPT, N_PTS_PER_AXIS, RECON_GRID_ORIGIN = "2500", 256, 1.0

train_dir = Path.cwd().parent / RUN
os.chdir(train_dir)
config        = load_config(config_path="model_params_config.json")
all_vtk_files = [os.path.basename(f) for f in config["list_mesh_paths"]]
OUT_DIR = Path("pc_videos"); OUT_DIR.mkdir(exist_ok=True)

import matplotlib.pyplot as plt
from PIL import Image
import io
def generate_latent_path_plot(projections, proj_val, min_proj, max_proj, PC_idx, width=600, height=240):
    print(f"projections shape: {projections.shape}")
    print(f"proj_val: {proj_val}")
    # Ensure projections is a 1D array
    projections = projections.flatten()
    # Plot latent points
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=300)
    # Set the background color to black
    fig.patch.set_facecolor('black')  # Black background for the figure
    ax.set_facecolor('black')  # Black background for the axes
    # Plot latent points
    ax.set_xlim(min_proj, max_proj)
    ax.plot([min_proj, max_proj], [0, 0], color='paleturquoise', alpha=0.2, linewidth=1)
    # Plot the current latent point (use proj_val directly and ensure it's scalar)
    ax.scatter(proj_val, 0, color='deeppink', s=10)
    # Customize the plot
    ax.set_yticks([])  # Hide y-axis ticks
    ax.set_xticks([0])
    ax.set_xticklabels(['0'])
    ax.grid(False)
    ax.legend().set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Path (PC{str((PC_idx+1))})", fontsize=10, color='white', pad=2)
    plt.tight_layout(pad=0.2)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=False)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    img_np = np.array(img)[..., :3]  # Drop alpha if any
    return img_np

# ── PC axis + a mesh(score) callable ─────────────────────────────────────────
if MODE == "landmarks":
    LM_DIR     = DROPBOX_ROOT / ATLAS_RUN / "alignedLMs"
    ATLAS_DIR  = DROPBOX_ROOT / ATLAS_RUN / "atlas"
    atlas_mesh = pv.read(str(ATLAS_DIR / "atlas_model.ply"))
    mean_lms, _ = load_mrk_json(ATLAS_DIR / "atlas_sparse_landmarks.mrk.json")

    lm_coords_3d = np.stack([load_mrk_json(LM_DIR / (os.path.splitext(f)[0] + ".mrk.json"))[0]
                             for f in all_vtk_files])
    pca         = gm_prcomp(lm_coords_3d)
    projections = pca["x"][:, PC_idx]

    def mesh_for_score(score, _i=[0]):
        warped, tps = warp_mesh(atlas_mesh, mean_lms, pc_shape(pca, PC_idx, score=score))
        del tps
        surf = warped.extract_surface(algorithm="dataset_surface").triangulate()
        del warped
        surf = surf.compute_normals(cell_normals=False, point_normals=True,
                                    inplace=False, auto_orient_normals=True)
        return pv_to_o3d(surf)

elif MODE == "latents":
    device = config.get("device", "cuda:0")
    model, _, latent_codes = load_model_and_latents(f"model/{CKPT}.pth",
                                                    f"latent_codes/{CKPT}.pth", config, device)
    latents_np  = latent_codes.numpy()
    centered    = latents_np - latents_np.mean(axis=0)
    pc_vec      = np.linalg.svd(centered, full_matrices=False)[2][PC_idx]
    projections = centered.dot(pc_vec)
    center_sample = latents_np[np.argmin(np.linalg.norm(centered, axis=1))]

    voxel_size = (RECON_GRID_ORIGIN * 2) / (N_PTS_PER_AXIS - 1)
    def mesh_for_score(score, _i=[0]):
        m = generate_and_render_mesh(center_sample + score * pc_vec, total_frames, _i[0],
                                     device, model, N_PTS_PER_AXIS,
                                     (-RECON_GRID_ORIGIN,) * 3, voxel_size,
                                     np.zeros(3), 1.0, NumpyTransform(np.eye(4)), 1, _i[0])
        _i[0] += 1
        return m
else:
    raise ValueError(f"Unknown MODE: {MODE}")

sign = -1.0 if FLIP_PC else 1.0
lo, hi = AMPLIFY * projections.min(), AMPLIFY * projections.max()
scores = np.concatenate([np.linspace(lo, 0, N_SEG), np.linspace(0, hi, N_SEG),
                         np.linspace(hi, 0, N_SEG), np.linspace(0, lo, N_SEG)])
total_frames = len(scores)

# ── Render ───────────────────────────────────────────────────────────────────
renderers = make_renderers(WIDTH, HEIGHT)
for r in renderers:
    r.scene.set_background(list(BG_COLOR) + [1.0])
mat = make_material()
rot = view_rotation(VIEW_ROT_DEG[VIEW])

video_path = OUT_DIR / f"pc{PC_idx + 1}_{MODE}_{VIEW}_{AMPLIFY}xpc.mp4"
out_video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                            FPS, (WIDTH, HEIGHT))

for i, score in enumerate(scores):
    try:
        o3d_mesh = mesh_for_score(sign * score)
        o3d_mesh.compute_vertex_normals()
        o3d_mesh.rotate(rot, center=o3d_mesh.get_center())

        # step 0 of a 1-step sweep -> the same fixed camera every grid cell used
        combined = render_cameras(renderers, o3d_mesh, 0, mat, 1, n_rotations=1)
        del o3d_mesh
        frame = crop_top_right(combined, WIDTH, HEIGHT).copy()
        del combined

        raw = generate_latent_path_plot(projections, score / AMPLIFY,
                                        projections.min(), projections.max(),
                                        PC_idx, width=260, height=60)
        slider = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        sh = max(1, round(slider.shape[0] * SLIDER_W / slider.shape[1]))
        slider = cv2.resize(slider, (SLIDER_W, sh), interpolation=cv2.INTER_AREA)
        h, w, _ = slider.shape
        x0, y0 = (WIDTH - w) // 2, HEIGHT - h - 20
        frame[y0:y0 + h, x0:x0 + w] = slider

        out_video.write(frame)
        print(f"  frame {i + 1}/{total_frames}  score={score:+.3f}", flush=True)

    except Exception as e:
        print(f"  Error at frame {i}: {e}")
        import traceback; traceback.print_exc()
    finally:
        gc.collect()

out_video.release()
print(f"Video saved → {video_path.resolve()}")
