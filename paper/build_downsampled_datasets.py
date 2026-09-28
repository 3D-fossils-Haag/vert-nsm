"""
Build taxon-stratified downsampled datasets (30/50/70 specimens) for the
sampling-effort experiments.

Run from nsm/paper/:
    python build_downsampled_datasets.py

Meshes are read from nsm/vertebrae_meshes/*.vtk and each mesh is assigned to a
specimen by longest-prefix match against 'specimen' in lizard_species_list.csv.
Specimens are drawn per broad_taxon_for_plotting group: 2 guaranteed per group,
the remainder sampled uniformly from what is left (so groups keep their natural
relative weighting). Meshes of the selected specimens are then shuffled and cut
80/15/5 train/val/test, exactly as train_model.py does.

Writes nsm/paper/splits/downsample_{N}spec.json with the keys train_model.py /
shape_completion_eval.py / classification_eval.py expect:
    list_mesh_paths, val_paths, test_paths
"""

import os, json, glob, re
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # nsm/
MESH_DIR = os.path.join(ROOT, 'vertebrae_meshes')
CSV = os.path.join(ROOT, 'lizard_species_list.csv')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'splits')
N_SPECIMENS = [10, 30, 50, 70]
SEED = 42

norm = lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower())  # ignore _ - and spaces when matching names
 
df = pd.read_csv(CSV).dropna(subset=['specimen', 'broad_taxon_for_plotting'])
specimens = sorted(df.specimen, key=len, reverse=True)  # longest first = unambiguous prefix match
taxon = dict(zip(df.specimen, df.broad_taxon_for_plotting))
 
# Map every mesh to its specimen
meshes = sorted(glob.glob(os.path.join(MESH_DIR, '*.vtk')))
mesh_spec = {m: next((s for s in specimens if norm(os.path.basename(m)).startswith(norm(s))), None) for m in meshes}
unmatched = sorted({os.path.basename(m).rsplit('-', 1)[0] for m, s in mesh_spec.items() if s is None})
print(f"{len(meshes)} meshes | {len(set(mesh_spec.values()) - {None})} specimens matched | "
      f"{len(unmatched)} unmatched (no taxon / name mismatch): {unmatched[:5]}")
 
groups = {}
for s in set(mesh_spec.values()) - {None}:
    groups.setdefault(taxon[s], []).append(s)
 
os.makedirs(OUT_DIR, exist_ok=True)
for n in N_SPECIMENS:
    rng = np.random.default_rng(SEED)
    shuffled = {k: list(rng.permutation(sorted(v))) for k, v in sorted(groups.items())}
    order = sorted(shuffled, key=lambda k: -len(shuffled[k]))                 # biggest taxa first
    seeded = order[:min(len(order), n // 2)]                                  # taxa that get their 2
    keep = [s for k in seeded for s in shuffled[k][:2]]                       # >=2 per taxon
    pool = [s for k in order for s in shuffled[k][2 if k in seeded else 0:]]
    keep = set(keep) | set(rng.choice(pool, n - len(keep), replace=False))    # rest, weighted by taxon size
    if len(seeded) < len(order):
        print(f"NOTE: n={n} is too small for 2 per taxon ({len(order)} taxa); only the {len(seeded)} largest are guaranteed.")
 
    paths = sorted(m for m, s in mesh_spec.items() if s in keep)
    rng.shuffle(paths)
    n_train, n_test = int(0.8 * len(paths)), int(0.15 * len(paths))
    n_val = len(paths) - n_train - n_test
    split = {'list_mesh_paths': sorted(paths[:n_train]),
             'val_paths': sorted(paths[n_train:n_train + n_val]),
             'test_paths': sorted(paths[n_train + n_val:]),
             'specimens': sorted(keep)}
 
    fn = os.path.join(OUT_DIR, f'downsample_{n}spec.json')
    json.dump(split, open(fn, 'w'), indent=1)
    counts = pd.Series([taxon[s] for s in keep]).value_counts().to_dict()
    print(f"\n{n} specimens -> {len(paths)} meshes "
          f"({len(split['list_mesh_paths'])}/{len(split['val_paths'])}/{len(split['test_paths'])} train/val/test)"
          f"\n  {counts}\n  saved {fn}")