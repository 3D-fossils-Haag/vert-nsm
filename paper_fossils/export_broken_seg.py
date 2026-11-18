#!/usr/bin/env python3
"""
export_broken_seg.py: write each fossil's broken surface as a 3D Slicer segmentation
('<fossil name>.seg.nrrd', one segment called 'broken'). Paper analysis only: stands in for
the file a real user paints in Slicer.
 
    python paper_fossils/export_broken_seg.py fossilized/ fossilized_broken/
    python paper_fossils/export_broken_seg.py fossilized/ fossilized_broken/ --intact vertebrae_meshes/
 
Labels come from the 'broken' label fossilize.py saves on each fossil. For fossils saved
without it, pass --intact (the modern meshes, matched by name): each is fitted onto its
fossil (allowing shear and squashing) and surface far from it is treated as broken.
"""
import argparse
import os
 
import nrrd
import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
 
from fossilize import MESH_EXTS, load_mesh
 
TOL = 0.015      # --intact only: surface farther than this (x bone size) is broken
VOXELS = 220     # voxels across the longest side (same as preprocess_fossilized.py)
THICK = 4        # shell thickness either side of the surface, voxels
GROW = 10        # grow the broken patches over the surface, voxels (rough painting spills over)
SMOOTH = 5       # close gaps and round edges, voxels (brush-like patches)
 
 
def save_broken_seg(points, broken, path, voxels=220, thick=4, grow=0, smooth=0):
    """Write a 'broken' segment: voxels within `thick` voxels of the surface whose nearest
    vertex is broken, then grown `grow` voxels over the surface and smoothed (closing +
    majority, `smooth` voxels) like a brush painting. Voxel size = longest side / `voxels`."""
    points, broken = np.asarray(points, float), np.asarray(broken).astype(bool)
    pitch = np.ptp(points, 0).max() / voxels
    pad = thick + grow + smooth + 2
    origin = points.min(0) - pad * pitch
    shape = np.ceil((points.max(0) + pad * pitch - origin) / pitch).astype(int) + 1
    near = np.zeros(shape, bool)                                   # voxels near the surface
    near[tuple(np.round((points - origin) / pitch).astype(int).T)] = True
    near = ndimage.distance_transform_edt(~near) <= thick + 1
    ijk = np.argwhere(near)
    d, j = cKDTree(points).query(ijk * pitch + origin)
    shell = np.zeros(shape, bool)
    shell[tuple(ijk[d <= thick * pitch].T)] = True
    seg = np.zeros(shape, bool)
    seg[tuple(ijk[(d <= thick * pitch) & broken[j]].T)] = True
    if grow:                                                       # spill over the rims
        seg = ndimage.distance_transform_edt(~seg) <= grow
    if smooth:                                                     # brush-like: close gaps, round edges
        seg = ndimage.distance_transform_edt(ndimage.distance_transform_edt(~seg) <= smooth) > smooth
        k = 2 * smooth + 1
        seg = ndimage.uniform_filter(seg.astype(np.float32), size=k) > 0.5
    seg = (seg & shell).astype(np.uint8)                           # stay on the surface
    header = {"type": "uint8", "dimension": 3, "space": "left-posterior-superior",
              "space directions": np.eye(3) * pitch, "space origin": origin,
              "kinds": ["domain"] * 3, "encoding": "gzip",
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
    a = p.parse_args()
    os.makedirs(a.output, exist_ok=True)
    for f in sorted(os.listdir(a.input)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in MESH_EXTS:
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
        save_broken_seg(mesh.vertices, broken, os.path.join(a.output, stem + ".seg.nrrd"), VOXELS, THICK, GROW, SMOOTH)
        print(f"ok    {f}: broken {100 * np.mean(broken):.1f}% of vertices")