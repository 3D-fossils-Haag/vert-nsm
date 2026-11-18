# Functions for handling fossil data

import os

import nrrd
import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from NSM.helper_funcs import get_sdfs

def load_broken_seg(path, points, name="broken"):
    """0/1 broken label per point, from a Slicer segmentation (.seg.nrrd) with a segment called
    `name` (else any segment) in the same frame as the points. Handles several segments, layers,
    RAS or LPS."""
    seg, h = nrrd.read(path)
    value, layer = None, 0
    for k, v in h.items():                                         # find the segment by name
        if k.endswith("_Name") and v.strip().lower() == name:
            p = k[:-len("_Name")]
            value, layer = int(h.get(p + "_LabelValue", 1)), int(h.get(p + "_Layer", 0))
    if seg.ndim == 4:                                              # layers on the first axis
        seg = seg[layer]
    D = np.array([r for r in h["space directions"] if r is not None and not np.any(np.isnan(r))], float)
    o = np.asarray(h["space origin"], float)
    pts = np.asarray(points, float)
    if h.get("space", "").startswith("right-anterior"):            # meshes are LPS
        pts = pts * [-1, -1, 1]
    ijk = np.round((pts - o) @ np.linalg.inv(D)).astype(int)
    inside = np.all((ijk >= 0) & (ijk < seg.shape), axis=1)
    out = np.zeros(len(pts), np.uint8)
    v = seg[tuple(ijk[inside].T)]
    out[inside] = (v == value) if value is not None else (v > 0)
    return out

def mask_broken_region(points, sdf_vals, mesh_path, center, max_radius, n_samples=None, seed=0):
    """Leave out SDF samples whose nearest fossil surface is broken (the mesh's 'broken' point
    labels), then keep n_samples of the rest (None = all). points are normalized: world =
    points * max_radius + center (as in normalize_mesh)."""
    mesh = pv.read(mesh_path)
    if "broken" not in mesh.point_data:
        print(f"\033[33mwarn  {mesh_path}: no 'broken' labels, keeping all SDF samples\033[0m")
        broken = np.zeros(mesh.n_points, bool)
    else:
        broken = np.asarray(mesh.point_data["broken"]).astype(bool)
    points, sdf_vals = points.reshape(-1, 3), sdf_vals.reshape(-1, 1)
    xyz = points.detach().cpu().numpy() * np.asarray(max_radius, float) + np.asarray(center, float)
    idx = np.flatnonzero(~broken[cKDTree(np.asarray(mesh.points, float)).query(xyz)[1]])
    if n_samples is not None and n_samples < len(idx):
        idx = np.sort(np.random.default_rng(seed).choice(idx, n_samples, replace=False))
    print(f"SDF samples: kept {len(idx)} of {len(points)} (broken surface left out)")
    idx = torch.as_tensor(idx, device=points.device)
    return points[idx], sdf_vals[idx]

# Pose correction limits (normalized frame, max radius = 1). Standard ~ MorphoWeave's ~2% alignment error
# plus fragment-normalization offsets; low-complete (<50%) allows more, with restarts and more iterations.
POSE_STANDARD = dict(lr_pose=1e-3, lambda_pose=1e-2, max_rot_deg=12, max_trans=0.15, max_log_scale=0.15,
                     n_starts=1, start_rot_deg=0, iters_scale=1)
POSE_LOW_COMPLETE = dict(lr_pose=2e-3, lambda_pose=1e-3, max_rot_deg=30, max_trans=0.35, max_log_scale=0.35,
                         n_starts=7, start_rot_deg=15, iters_scale=2)


def undo_pose_error(compl, fossil_path, ref_path):
    """Move a completion from the aligned fossil's frame into the ground truth frame, using the exact similarity
    between the aligned fossil and its unperturbed copy (same vertex order). This removes the alignment error, so
    Chamfer/L1 measure shape error only. Returns compl unchanged if the meshes don't correspond."""
    if not ref_path or not os.path.exists(ref_path):
        return compl
    X = np.asarray(pv.read(fossil_path).points, float)
    Y = np.asarray(pv.read(ref_path).points, float)
    if len(X) != len(Y):
        return compl
    mx, my = X.mean(0), Y.mean(0)
    U, S, Vt = np.linalg.svd((X - mx).T @ (Y - my))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / ((X - mx) ** 2).sum()
    out = compl.copy()
    out.points = s * (np.asarray(compl.points, float) - mx) @ R + my
    return out

def snap_to_partial(mesh_out, partial_pts, iters=30, trim=0.8):
    """Rigidly move the completion so the input fossil's points lie on its surface."""
    pts = np.asarray(mesh_out.points, float)
    for _ in range(iters):
        dist, j = cKDTree(pts).query(partial_pts)
        keep = dist <= np.quantile(dist, trim)          # ignore worst 20% (broken edges, misfit)
        p, q = pts[j[keep]], partial_pts[keep]
        mp, mq = p.mean(0), q.mean(0)
        U, _, Vt = np.linalg.svd((p - mp).T @ (q - mq))
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        pts = (pts - mp) @ (U @ D @ Vt) + mq
    out = mesh_out.copy()
    out.points = pts
    return out

def rotation_matrix(r):
    """Rotation matrix from an axis-angle vector."""
    theta = torch.sqrt((r * r).sum() + 1e-12)
    k = r / theta
    z = r.new_zeros(())
    K = torch.stack([torch.stack([z, -k[2], k[1]]), torch.stack([k[2], z, -k[0]]), torch.stack([-k[1], k[0], z])])
    return torch.eye(3, dtype=r.dtype, device=r.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)

def optimize_latent_pose(decoder, partial_pts, sdfs, latent_init, pose_cfg, iters=500, lr=1e-5, lambda_reg=1e-5,
                         batch_inference_size=32768, device='cuda'):
    """Refine a latent together with a small pose correction of the input points. Returns (latent, pose)."""
    decoder = decoder.to(device).eval()
    pts0 = torch.as_tensor(partial_pts, dtype=torch.float32).to(device)
    target = torch.as_tensor(sdfs, dtype=torch.float32).to(device).reshape(-1, 1)
    latent0 = latent_init.clone().detach().to(device)
    max_rot = np.radians(pose_cfg["max_rot_deg"])
    starts = [torch.zeros(7)]
    for axis in range(3):
        for sign in (1, -1):
            p = torch.zeros(7)
            p[axis] = sign * np.radians(pose_cfg["start_rot_deg"])
            starts.append(p)
    best = None
    for start in starts[:pose_cfg["n_starts"]]:
        latent = latent0.clone().requires_grad_(True)
        pose = start.clone().to(device).requires_grad_(True)
        optimizer = torch.optim.Adam([{"params": [latent], "lr": lr}, {"params": [pose], "lr": pose_cfg["lr_pose"]}])
        for step in range(int(iters * pose_cfg["iters_scale"])):
            optimizer.zero_grad()
            s = torch.exp(pose[6])
            pts = s * pts0 @ rotation_matrix(pose[:3]).T + pose[3:6]   # x' = s * x @ R^T + t
            preds = get_sdfs(decoder, pts, latent, batch_size=batch_inference_size, device=device)
            sdf_loss = F.l1_loss(preds, s * target)                     # SDF scales with the points
            loss = sdf_loss + lambda_reg * F.mse_loss(latent, latent0) + pose_cfg["lambda_pose"] * (pose ** 2).sum()
            loss.backward()
            optimizer.step()
            with torch.no_grad():                                       # keep the correction within limits
                rot = pose[:3].norm()
                if rot > max_rot:
                    pose[:3] *= max_rot / rot
                trans = pose[3:6].norm()
                if trans > pose_cfg["max_trans"]:
                    pose[3:6] *= pose_cfg["max_trans"] / trans
                pose[6].clamp_(-pose_cfg["max_log_scale"], pose_cfg["max_log_scale"])
        fit = sdf_loss.item() / s.item()
        print(f"pose correction: rot {np.degrees(pose[:3].norm().item()):.2f} deg, trans {pose[3:6].norm().item():.4f}, "
              f"scale {s.item():.4f}, sdf loss {fit:.3e}")
        if best is None or fit < best[0]:
            best = (fit, latent.detach(), pose.detach())
    return best[1], best[2]

def undo_pose(mesh_out, pose):
    """Move a reconstructed mesh back onto the input (call before normalize_mesh)."""
    mesh_pv = mesh_out if isinstance(mesh_out, pv.PolyData) else mesh_out.extract_surface(algorithm=None)
    p = pose.cpu()
    R = rotation_matrix(p[:3]).numpy()
    mesh_pv = mesh_pv.copy()
    mesh_pv.points = (np.asarray(mesh_pv.points) - p[3:6].numpy()) / float(torch.exp(p[6])) @ R
    return mesh_pv