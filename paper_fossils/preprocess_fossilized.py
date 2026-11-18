#!/usr/bin/env python3
"""
preprocess_fossilized.py: prepare fossil meshes for NSM shape completion: a smoothed,
hole-filled, closed surface. Broken surface is kept (not cut away); its 'broken' label is
read from the fossil's segmentation (see export_broken_seg.py) at the new vertices.
 
    python paper_fossils/preprocess_fossilized.py fossil.vtk fossil_prep.vtk --broken fossil.seg.nrrd
    python paper_fossils/preprocess_fossilized.py fossilized/ fossilized_preprocessed/ --broken fossilized_broken/ --workers 8
 
Run from the repo root. In folder mode outputs keep the input file names (the folder
marks the preprocessing), so downstream scripts match them to the raw fossils as is.
 
Steps, in voxels (defaults match the manual workflow):
    labelmap (interior filled) -> grow margin 3 -> smooth 6 -> fill holes 6 ->
    shrink back 3 -> drop specks under 1000 voxels -> surface mesh -> 'broken' labels
 
Labels come from '<fossil name>.seg.nrrd' (a 'broken' segment, as painted in 3D Slicer or
written by export_broken_seg.py). Without one the output has no 'broken' label.
 
Requires: numpy scipy trimesh scikit-image (vtk for .vtk), and fossilize.py
in the same folder.
"""
from __future__ import annotations
 
import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
 
import numpy as np
import trimesh
from scipy import ndimage
 
from fossilize import MESH_EXTS, load_mesh, save_mesh
from NSM.fossils import load_broken_seg
 
# all sizes in voxels unless noted
DEFAULTS = dict(
    voxels=220,        # voxels across the bone's longest side (sets the voxel size)
    margin=3,          # grow margin
    smooth=6,          # median smoothing kernel
    fill=6,            # fill holes (closing) kernel
    shrink=3,          # shrink back after filling holes (= margin: undoes the growth, keeps bone size)
    min_island=1000,   # keep every piece at least this big, not just the largest
)
 
 
# ----------------------------------------------------------------------------
# The Segment Editor steps
# ----------------------------------------------------------------------------
def _ball_close(occ, r):
    """Morphological closing with a ball of radius r voxels (fills holes/gaps)."""
    dil = ndimage.distance_transform_edt(~occ) <= r
    return ndimage.distance_transform_edt(dil) > r
 
 
def prepare(mesh, voxels=220, margin=3, smooth=6, fill=6, shrink=3, min_island=1000, **_):
    """Run the preprocessing on one mesh. Returns (smoothed mesh, info dict)."""
    pitch = mesh.extents.max() / voxels
    pad = margin + smooth + fill + 4
    vox = mesh.voxelized(pitch).fill()                        # solid labelmap (closes internal cavities)
    occ = np.pad(vox.matrix, pad)
    origin = vox.transform[:3, 3] - pad * pitch               # world position of voxel (0,0,0)
 
    occ = ndimage.distance_transform_edt(~occ) <= margin      # grow margin
    k = smooth + (smooth % 2 == 0)                            # median of a binary = majority
    occ = ndimage.uniform_filter(occ.astype(np.float32), size=k) > 0.5
    occ = _ball_close(occ, fill / 2)                          # fill holes
    if shrink:                                                # shrink back (undo the margin's growth)
        occ = ndimage.distance_transform_edt(occ) > shrink
 
    lab, n = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    if n == 0:
        raise ValueError("nothing left after smoothing")
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.isin(lab, 1 + np.flatnonzero(sizes >= min(min_island, sizes.max())))  # drop specks only
 
    from skimage.measure import marching_cubes
    field = ndimage.gaussian_filter(keep.astype(np.float32), 0.6)   # gentle, like Slicer's surface smoothing
    verts, faces, _, _ = marching_cubes(field, 0.5)
    out = trimesh.Trimesh(verts * pitch + origin, faces, process=True)
    trimesh.repair.fix_normals(out)
 
    info = dict(voxel_size=pitch, pieces=n, kept_pct=round(100 * sizes.max() / max(sizes.sum(), 1), 1))
    return out, info
 
 
# ----------------------------------------------------------------------------
# Files, folders, command line
# ----------------------------------------------------------------------------
def _one(job):
    src, dst, seg, opts = job
    if os.path.exists(dst):                                   # done already (this run or an earlier one)
        return f"skip  {os.path.basename(dst)}: exists"
    t0 = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, info = prepare(load_mesh(src), **opts)
            note = "no 'broken' segmentation"
            if seg and os.path.exists(seg):                   # labels at the new vertices
                out.vertex_attributes["broken"] = load_broken_seg(seg, out.vertices)
                note = f"broken {100 * out.vertex_attributes['broken'].mean():.1f}%"
            save_mesh(out, dst)
        return f"ok    {os.path.basename(dst)} ({time.time() - t0:.0f}s) {note}, {info['pieces']} piece(s), " \
               f"voxel {info['voxel_size']:.4g}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR {os.path.basename(src)}: {type(e).__name__}: {e}"
 
 
def main():
    p = argparse.ArgumentParser(description="Prepare fossil meshes for NSM: smoothed, hole-filled closed "
                                            "surface, 'broken' labels carried over (automates the Slicer steps).")
    p.add_argument("input", help="fossil mesh, or folder of them")
    p.add_argument("output", help="output mesh, or output folder")
    p.add_argument("--broken", help="'broken' segmentation (.seg.nrrd), or folder of them named "
                                    "<fossil name>.seg.nrrd")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v, help=f"default {v}")
    p.add_argument("--workers", type=int, default=1, help="files in parallel")
    p.add_argument("--suffix", default="", help="added to output names in folder mode (default: none, same names)")
    a = p.parse_args()
    opts = {k: getattr(a, k) for k in DEFAULTS}
 
    if not os.path.isdir(a.input):
        print(_one((a.input, a.output, a.broken, opts)))
        return
    os.makedirs(a.output, exist_ok=True)
    jobs, missing = [], []
    for f in sorted(os.listdir(a.input)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in MESH_EXTS:
            continue
        seg = os.path.join(a.broken, stem + ".seg.nrrd") if a.broken else None
        if not seg or not os.path.exists(seg):
            print(f"\033[33mwarn  {f}: no {seg or 'segmentation (--broken)'}, output has no 'broken' label\033[0m")
            missing.append(f"{f},{seg or ''}")
        jobs.append((os.path.join(a.input, f), os.path.join(a.output, stem + a.suffix + ext), seg, opts))
    if missing:
        with open(os.path.join(a.output, "missing_segs.csv"), "w") as fh:
            fh.write("fossil,seg\n" + "\n".join(missing) + "\n")
    print(f"{len(jobs)} file(s), {a.workers} worker(s)")
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            for f in as_completed([ex.submit(_one, j) for j in jobs]):
                print(f.result(), flush=True)
    else:
        for j in jobs:
            print(_one(j), flush=True)
 
 
if __name__ == "__main__":
    main()