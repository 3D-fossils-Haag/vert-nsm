"""Downsample train evaluation experiments: Collect the meshes that no run used for training.

Comparing models fairly requires scoring them all on the same meshes, and on
meshes none of them has seen. Each run has its own split, so this takes every
mesh in the dataset and keeps the ones absent from all the runs' training sets.
Writes the result as a JSON list of paths.

Usage
---
python build_downsample_val_ds.py --roots ../run_10spec ../run_30spec ../run_v72

"""

import argparse
import collections
import json
import os
import re
import sys

SPLIT_KEYS = ["list_mesh_paths", "val_paths", "test_paths"]

# run directories relative to NSM/nsm/paper
ROOTS = ["../run_10spec", "../run_30spec", "../run_50spec", "../run_70spec",
         "../run_v72"]

# written beside the runs so the eval scripts can find it from NSM/nsm
OUT = "../common_eval_meshes.json"


def load_configs(roots):
    cfgs = {}
    for root in roots:
        path = os.path.join(root, "model_params_config.json")
        if not os.path.isfile(path):
            sys.exit(f"No model_params_config.json in {root}")
        cfgs[os.path.basename(root.rstrip("/"))] = json.load(open(path))
    return cfgs


def collect(cfgs):
    """Collect all meshes and all trained-on meshes, keyed by basename.

    Runs use different absolute paths for the same mesh, so basename is identity.
    """
    paths, trained, seen = {}, set(), set()
    for cfg in cfgs.values():
        for key in SPLIT_KEYS:
            for p in cfg.get(key, []):
                base = os.path.basename(p)
                paths.setdefault(base, p)
                seen.add(base)
                if key == "list_mesh_paths":
                    trained.add(base)
    return paths, seen, trained


def specimen_of(mesh):
    """Strip the vertebra suffix: ..._21-t16.ply or ..._002.vtk -> specimen."""
    return re.sub(r"_\d+(-[ctlCTL]\d+)?\.[^.]+$", "", mesh).lower()


def report(cfgs, seen, trained, safe):
    print(f"{'run':<14}{'train':>8}{'held out':>10}{'of safe':>9}")
    print("-" * 41)
    for run, cfg in cfgs.items():
        tr = {os.path.basename(p) for p in cfg.get("list_mesh_paths", [])}
        held = {os.path.basename(p) for k in SPLIT_KEYS[1:]
                for p in cfg.get(k, [])}
        print(f"{run:<14}{len(tr):>8}{len(held):>10}{len(held & safe):>9}")

    print(f"\n{len(seen)} meshes total, {len(trained)} trained on by at least "
          f"one run, {len(safe)} held out by all.")

    if not safe:
        sys.exit("\nNo mesh is held out from every run; nothing to write.")

    spec = collections.Counter(specimen_of(m) for m in safe)
    print(f"\n{len(spec)} specimens contribute to the query set:")
    for name, n in spec.most_common():
        print(f"  {n:>4}  {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=ROOTS,
                    help=f"run directories (default: {' '.join(ROOTS)})")
    ap.add_argument("--out", default=OUT,
                    help="where to write the mesh list (default: %(default)s)")
    ap.add_argument("--limit", type=int, default=None,
                    help="keep only the first N meshes, for a quick trial run")
    args = ap.parse_args()

    cfgs = load_configs(args.roots)
    paths, seen, trained = collect(cfgs)
    safe = seen - trained

    report(cfgs, seen, trained, safe)

    # sorted so the list is stable across invocations and runs stay comparable
    common = [paths[m] for m in sorted(safe)]
    if args.limit:
        common = common[:args.limit]
        print(f"\nlimited to first {len(common)}")

    json.dump(common, open(args.out, "w"), indent=1)
    print(f"\nwrote {len(common)} mesh paths -> {args.out}")


if __name__ == "__main__":
    main()