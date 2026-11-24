#!/usr/bin/env python3
"""
export_broken_seg.py: write each fossil's broken surface as a 3D Slicer segmentation
('<fossil name>.seg.nrrd', one segment called 'broken'). Paper analysis only: stands in for
the file a real user paints in Slicer.
 
    python paper_fossils/export_broken_seg.py fossilized/ fossilized_broken/
    python paper_fossils/export_broken_seg.py fossilized/ fossilized_broken/ --intact vertebrae_meshes/
    python paper_fossils/export_broken_seg.py fossilized/ fossilized_broken/ --reverse   # 2nd process, from the end
 
Labels come from the 'broken' label fossilize.py saves on each fossil. For fossils saved
without it, pass --intact (the modern meshes, matched by name): each is fitted onto its
fossil (allowing shear and squashing) and surface far from it is treated as broken.
"""
import argparse
import os
 
import nrrd
import numpy as np
import trimesh
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
 
from fossilize import MESH_EXTS, load_mesh
 
TOL = 0.015      # --intact only: surface farther than this (x bone size) is broken
VOXELS = 220     # voxels across the longest side (same as preprocess_fossilized.py)
THICK = 4        # shell thickness either side of the surface, voxels
GROW = 10        # grow the broken patches along the surface, voxels (rough painting spills over)
SMOOTH = 5       # close gaps and round edges along the surface, voxels (brush-like patches)
 
 
def surface_dist(mesh, sources, limit):
    """Geodesic (along mesh edges) distance from the source vertices, up to limit (inf beyond)."""
    e, w = mesh.edges_unique, mesh.edges_unique_length
    n = len(mesh.vertices)
    G = sparse.coo_matrix((np.r_[w, np.zeros(len(sources))],                     # super-source n -> sources
                           (np.r_[e[:, 0], np.full(len(sources), n)], np.r_[e[:, 1], sources])), shape=(n + 1, n + 1)).tocsr()
    return dijkstra(G, directed=False, indices=n, limit=limit)[:n]

def save_broken_seg(mesh, broken, path, voxels=220, thick=4, grow=0, smooth=0):
    """Write a 'broken' segment like a brush painting: the broken vertices grown `grow` voxels and
    smoothed (closing, `smooth` voxels) along the surface, so it never jumps across thin bone or
    gaps; then voxels within `thick` of the surface whose nearest vertex is broken. Voxel size =
    longest side / `voxels`."""
    points, broken = np.asarray(mesh.vertices, float), np.asarray(broken).astype(bool)
    pitch = np.ptp(points, 0).max() / voxels
    if grow or smooth:            # along the surface: never jumps across thin bone or gaps
        sel = surface_dist(mesh, np.flatnonzero(broken), (grow + smooth) * pitch) <= (grow + smooth) * pitch
        if smooth:
            sel = ~(surface_dist(mesh, np.flatnonzero(~sel), smooth * pitch) <= smooth * pitch) if (~sel).any() else sel
        broken = sel
    pad = thick + 2
    origin = points.min(0) - pad * pitch
    shape = np.ceil((points.max(0) + pad * pitch - origin) / pitch).astype(int) + 1
    near = np.zeros(shape, bool)
    near[tuple(np.round((points - origin) / pitch).astype(int).T)] = True
    near = ndimage.distance_transform_edt(~near) <= thick + 1
    ijk = np.argwhere(near)
    d, j = cKDTree(points).query(ijk * pitch + origin)
    seg = np.zeros(shape, np.uint8)
    seg[tuple(ijk[(d <= thick * pitch) & broken[j]].T)] = 1
    header = {"type": "uint8", "dimension": 3, "space": "left-posterior-superior",
              "space directions": np.eye(3) * pitch, "space origin": origin, "kinds": ["domain"] * 3, "encoding": "gzip",
              "Segment0_ID": "broken", "Segment0_Name": "broken", "Segment0_Color": "1 0 0",
              "Segment0_LabelValue": "1", "Segment0_Layer": "0",
              "Segment0_Extent": " ".join(f"0 {n - 1}" for n in shape),
              "Segmentation_MasterRepresentation": "Binary labelmap",
              "Segmentation_ContainedRepresentationNames": "Binary labelmap|"}
    nrrd.write(path, seg, header)


def labels_from_intact(fossil, intact, tol=0.015, max_angle=45, iters=40):
    """Label fossil vertices broken where they are far from the intact mesh, or where
    they face a different way than the intact surface there (fracture faces cut
    across the original surface, even through thin parts where they stay close to
    it). The intact mesh is first fitted onto the fossil with a trimmed affine ICP,
    which absorbs shear and flattening."""
    S = fossil.scale
    fv, fn = np.array(fossil.vertices), np.array(fossil.vertex_normals)
    src, fi = trimesh.sample.sample_surface_even(intact, 200000, seed=0)
    src_n = intact.face_normals[fi]
    A, t = np.eye(3), fossil.centroid - intact.centroid      # intact -> fossil
    q = fv[np.random.default_rng(0).choice(len(fv), min(len(fv), 30000), replace=False)]
    for it in range(iters):
        d, j = cKDTree(src @ A.T + t).query(q)
        keep = d <= np.quantile(d, 0.7 if it < iters // 2 else 0.85)   # ignore breaks
        sol, *_ = np.linalg.lstsq(np.c_[src[j[keep]], np.ones(keep.sum())], q[keep], rcond=None)
        A, t = sol[:3].T, sol[3]
    d, j = cKDTree(src @ A.T + t).query(fv)
    n_int = src_n[j] @ np.linalg.inv(A)                       # normals under the affine map
    n_int /= np.linalg.norm(n_int, axis=1, keepdims=True)
    facing = np.einsum("ij,ij->i", fn, n_int)
    broken = (d > tol * S) | (facing < np.cos(np.radians(max_angle)))
    # tidy: drop specks (texture can tilt a few normals), grow a little over the rims
    L = trimesh.smoothing.laplacian_calculation(fossil, equal_weight=True)
    b = broken.astype(float)
    for _ in range(4):
        b = L @ b
    broken = b > 0.5
    for _ in range(3):
        broken = (L @ broken.astype(float)) > 0
    return broken.astype(np.uint8)

 
 
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Write fossils' broken surface as Slicer segmentations.")
    p.add_argument("input", help="folder of fossil meshes")
    p.add_argument("output", help="output folder for <fossil name>.seg.nrrd")
    p.add_argument("--intact", help="folder of intact meshes, for fossils saved without a 'broken' label")
    p.add_argument("--reverse", action="store_true", help="go through the files from the end (run alongside a forward run)")
    a = p.parse_args()
    os.makedirs(a.output, exist_ok=True)
    for f in sorted(os.listdir(a.input), reverse=a.reverse):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in MESH_EXTS:
            continue
        out_path = os.path.join(a.output, stem + ".seg.nrrd")
        if os.path.exists(out_path):                    # done already (this run, the other run, or an earlier one)
            continue
        mesh = load_mesh(os.path.join(a.input, f))
        if a.intact:                                    # match fossil "name_fossil..." to intact "name.*"
            hits = [g for g in os.listdir(a.intact) if os.path.splitext(g)[0] == stem.split("_fossil")[0]]
            if not hits:
                print(f"skip  {f}: no intact mesh")
                continue
            broken = labels_from_intact(mesh, load_mesh(os.path.join(a.intact, hits[0])), TOL)
        else:
            broken = mesh.vertex_attributes.get("broken")
            if broken is None:
                print(f"skip  {f}: no 'broken' label (pass --intact)")
                continue
        save_broken_seg(mesh, broken, out_path, VOXELS, THICK, GROW, SMOOTH)
        print(f"ok    {f}: broken {100 * np.mean(broken):.1f}% of vertices")