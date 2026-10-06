"""
Table 2: shape completion chamfer descriptives by model and split.

Example usage

Main analysis
python shape_completion_table.py --datadir shape_completion_eval/ --outdir shape_completion_eval/

Downsample dataset training experiment
python shape_completion_table.py --runs ../run_10spec ../run_30spec ../run_50spec ../run_70spec ../run_v72 --outdir ../downsample_tables
"""
import argparse, os, sys
import numpy as np
import pandas as pd

MODELS = {
    "Baseline":       "v72_{split}_chamfer.csv",
    "Encoder":        "v72_encoder_{split}_chamfer.csv",
    "Encoder+refine": "v72_encoder_{split}_chamfer_refine.csv",
    "Hierarchy":      "v73h_{split}_chamfer.csv",
    "Contrastive":    "v73c_{split}_chamfer.csv"}

ORDER  = ["Baseline", "Encoder", "Encoder+refine", "Hierarchy", "Contrastive"]
SPLITS = ["train", "val", "test"]
SCALE  = 1e3

def items_flat(datadir):
    return [(label, split, os.path.join(datadir, tmpl.format(split=split)))
            for label, tmpl in MODELS.items() for split in SPLITS]

def items_runs(roots, mode, tag):
    return [(os.path.basename(r.rstrip("/")), "val",
             os.path.join(r, "shape_completion", "evaluation",
                          f"{mode}_val{tag}_chamfer.csv"))
            for r in roots]

def read_one(path):
    first = open(path).readline()
    df = pd.read_csv(path) if first.startswith("mesh,") else \
         pd.read_csv(path, header=None, names=["mesh", "chamfer", "gt_path"])
    df["mesh"] = df["mesh"].str.replace("_partial.ply", "", regex=False)
    if df["mesh"].duplicated().any():
        df = df.groupby("mesh", as_index=False)["chamfer"].mean()
    return df

def load_all(items):
    """items: (label, split, path) triples."""
    frames = []
    for label, split, path in items:
        if not os.path.exists(path):
            print(f"  missing {path} -- skipping"); continue
        df = read_one(path)
        df["model"], df["split"] = label, split
        frames.append(df[["mesh", "chamfer", "model", "split"]])
    if not frames:
        sys.exit("No chamfer CSVs found")
    return pd.concat(frames, ignore_index=True)

def table2(d, outdir, name="Table_2_shp_compl_eval.csv"):
    rows = []
    order = [m for m in ORDER if m in set(d.model)] + \
            [m for m in dict.fromkeys(d.model) if m not in ORDER]
    for split in [s for s in SPLITS if s in set(d.split)]:
        for model in order:
            c = d[(d.split == split) & (d.model == model)]["chamfer"].dropna().to_numpy() * SCALE
            if not len(c): continue
            rows.append({
                "Split": split, "Model": model, "n": len(c),
                "Mean ± SD": f"{c.mean():.2f} ± {c.std(ddof=1):.2f}",
                "Median [IQR]": f"{np.median(c):.2f} [{np.percentile(c,25):.2f}–{np.percentile(c,75):.2f}]"})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(outdir, name), index=False)
    print(tab.to_string(index=False))
    return tab

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default=".")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--runs", nargs="+", default=None,
                    help="run dirs; reads <run>/shape_completion/evaluation/")
    ap.add_argument("--mode", default="2phase", choices=["2phase", "encoder"])
    ap.add_argument("--tag", default="_common", help="file tag, '' for none")
    ap.add_argument("--name", default=None, help="output filename")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.runs:
        items = items_runs(args.runs, args.mode, args.tag)
        name = args.name or "Table_downsample_shp_compl.csv"
    else:
        items = items_flat(args.datadir)
        name = args.name or "Table_2_shp_compl_eval.csv"
    table2(load_all(items), args.outdir, name)

if __name__ == "__main__":
    main()