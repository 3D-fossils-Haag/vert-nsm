# Functions related to alignment of meshes and landmarks

import json
import os

import numpy as np

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

# Similarity (Procrustes) fit of Y_L onto template landmarks mu_L, applied as s * X @ R + t
def align_to_template(mu_L, Y_L, allow_scaling=True):
    mt, mm = mu_L.mean(0), Y_L.mean(0)
    U, S, Vt = np.linalg.svd((Y_L - mm).T @ (mu_L - mt))
    d = np.sign(np.linalg.det(U @ Vt))
    U[:, -1] *= d                                       # no reflections
    R = U @ Vt
    s = (S[:-1].sum() + d * S[-1]) / np.sum((Y_L - mm) ** 2) if allow_scaling else 1.0
    return s, R, mt - s * mm @ R

def rms_err_pct(aligned, orig, size=None):
    """RMS vertex displacement as % of bone size (bounding-box diagonal of orig, unless size is given)."""
    size = np.linalg.norm(np.ptp(orig, axis=0)) if size is None else size
    return 100 * np.sqrt(np.mean(np.sum((aligned - orig) ** 2, 1))) / size