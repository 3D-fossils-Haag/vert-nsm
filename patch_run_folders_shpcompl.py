#!/usr/bin/env python3
"""
patch_run_folders.py: move existing grid search runs into the per-split layout.

    1. <shape_completion>/<run_name>/*  ->  <shape_completion>/<run_name>/train/*   (every run first)
    2. <run_name>/<split>/best_cfg_predictions/inference_summary.csv
           ->  <shape_completion>/inference_summary_<run_name>_<split>.csv     (same for inference_vs_alignment.csv)

Run folders are those whose name starts with "fine_tuning". Anything already in a train/,
val/ or test/ subfolder is left where it is. Prints what it would do; set DRY_RUN = False
to do it.

    python patch_run_folders.py
"""
import glob
import os
import shutil

SHAPE_COMPLETION = "run_v72/shape_completion"    # TO DO
OLD_SPLIT = "train"                              # split the existing, un-split runs belong to
DRY_RUN = False                                   # TO DO: False to actually move and copy
SPLITS = ("train", "val", "test")
SUMMARIES = ("inference_summary.csv", "inference_vs_alignment.csv")

runs = [r for r in sorted(glob.glob(os.path.join(SHAPE_COMPLETION, "fine_tuning*"))) if os.path.isdir(r)]

# 1. move every run's top-level contents into <run>/train/
print("Step 1: move into train/")
for run in runs:
    run_name = os.path.basename(run)
    to_move = [e for e in sorted(os.listdir(run)) if e not in SPLITS]
    if not to_move:
        continue
    dest = os.path.join(run, OLD_SPLIT)
    print(f"\n  {run_name}  ->  {os.path.abspath(dest)}/")
    for e in to_move:
        print(f"      {e}{'/' if os.path.isdir(os.path.join(run, e)) else ''}")
    if DRY_RUN:
        continue
    os.makedirs(dest, exist_ok=True)
    for e in to_move:
        if os.path.exists(os.path.join(dest, e)):
            print(f"    skip {e}: already exists in {OLD_SPLIT}/")
            continue
        shutil.move(os.path.join(run, e), os.path.join(dest, e))
    # the summaries record where each completed mesh was saved: point them at the new place
    for csv in glob.glob(os.path.join(dest, "best_cfg_predictions", "*.csv")):
        with open(csv) as f:
            text = f.read()
        with open(csv, "w") as f:
            f.write(text.replace(f"{run_name}/best_cfg_predictions/", f"{run_name}/{OLD_SPLIT}/best_cfg_predictions/"))

# 2. copy each run's summaries up to shape_completion/
print("\nStep 2: copy summaries")
for run in runs:
    run_name = os.path.basename(run)
    for split in SPLITS:
        for fname in SUMMARIES:
            src = os.path.join(run, split, "best_cfg_predictions", fname)
            if DRY_RUN and split == OLD_SPLIT and not os.path.exists(src):
                src = os.path.join(run, "best_cfg_predictions", fname)   # where it is until step 1 runs
            if not os.path.exists(src):
                continue
            dst = os.path.join(SHAPE_COMPLETION, f"{fname[:-4]}_{run_name}_{split}.csv")
            print(f"  {os.path.abspath(os.path.join(run, split, 'best_cfg_predictions', fname))}\n"
                  f"    -> {os.path.abspath(dst)}")
            if not DRY_RUN:
                shutil.copy2(src, dst)

if DRY_RUN:
    print("\nDry run: nothing changed. Set DRY_RUN = False to apply.")