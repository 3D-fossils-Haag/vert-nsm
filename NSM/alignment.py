# Functions related to alignment of meshes and landmarks

import json
import os

import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from NSM.helper_funcs import get_sdfs

# Load landmarks file (.mrk.json)
def load_mrk_json(path, return_idx=False):
    with open(path, "r") as f:
        data = json.load(f)
    markups = data.get("markups", [])
    if not markups:
        raise ValueError(f"No 'markups' found in {path}")
    control_points = [cp for cp in markups[0].get("controlPoints", []) if cp.get("position") is not None]
    points = np.asarray([cp["position"] for cp in control_points], dtype=np.float32)
    labels = [cp.get("label") for cp in control_points]
    if not return_idx:
        return points, labels
    # atlas index of each point ('index:N' in description, else position in file)
    idx = [int(cp["description"].split("index:")[1]) if "index:" in cp.get("description", "") else i
           for i, cp in enumerate(control_points)]
    return points, labels, idx

# Save landmarks file (.mrk.json). Each point copies the template's control point with the same
# atlas index (label etc.) and records 'index:<i>' in its description.
def save_mrk_json(points, output_path, template_path=None, idx=None):
    idx = list(range(len(points))) if idx is None else idx
    data, tmpl = {"markups": [{"controlPoints": []}]}, {}
    if template_path is not None:
        with open(template_path, "r") as f:
            data = json.load(f)
        cps = data["markups"][0]["controlPoints"]
        tmpl = {int(cp["description"].split("index:")[1]) if "index:" in cp.get("description", "") else i: cp
                for i, cp in enumerate(cps)}
    data["markups"][0]["controlPoints"] = [
        dict(tmpl.get(i, {"id": str(i), "label": str(i), "selected": True, "locked": False, "visibility": True}),
             position=[float(v) for v in p], description=f"index:{i}")
        for p, i in zip(points, idx)]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

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

# Similarity (Procrustes) fit of Y_L onto template landmarks mu_L, applied as s * X @ R + t
def align_to_template(mu_L, Y_L, allow_scaling=True):
    mt, mm = mu_L.mean(0), Y_L.mean(0)
    U, S, Vt = np.linalg.svd((Y_L - mm).T @ (mu_L - mt))
    d = np.sign(np.linalg.det(U @ Vt))
    U[:, -1] *= d                                       # no reflections
    R = U @ Vt
    s = (S[:-1].sum() + d * S[-1]) / np.sum((Y_L - mm) ** 2) if allow_scaling else 1.0
    return s, R, mt - s * mm @ R

# Pose correction limits (normalized frame, max radius = 1). Standard ~ MorphoWeave's ~2% alignment error
# plus fragment-normalization offsets; low-complete (<50%) allows more, with restarts and more iterations.
POSE_STANDARD = dict(lr_pose=1e-3, lambda_pose=1e-2, max_rot_deg=12, max_trans=0.15, max_log_scale=0.15,
                     n_starts=1, start_rot_deg=0, iters_scale=1)
POSE_LOW_COMPLETE = dict(lr_pose=2e-3, lambda_pose=1e-3, max_rot_deg=30, max_trans=0.35, max_log_scale=0.35,
                         n_starts=7, start_rot_deg=15, iters_scale=2)

def snap_to_partial(mesh_out, partial_pts, iters=30, trim=0.8):
    """Rigidly move the completion so the input fossil's points lie on its surface."""
    pts = np.asarray(mesh_out.points, float)
    for _ in range(iters):
        dist, j = cKDTree(pts).query(partial_pts)
        keep = dist <= np.quantile(dist, trim)          # ignore worst 20% (hollow-cut inner wall, broken edges)
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

def rms_err_pct(aligned, orig, size=None):
    """RMS vertex displacement as % of bone size (bounding-box diagonal of orig, unless size is given)."""
    size = np.linalg.norm(np.ptp(orig, axis=0)) if size is None else size
    return 100 * np.sqrt(np.mean(np.sum((aligned - orig) ** 2, 1))) / size