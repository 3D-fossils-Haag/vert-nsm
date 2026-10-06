"""Manuscript tables from the classification results.

Reads the tree written by classification_eval.py:

    <run>/classification/evaluation/<tag or split>/<split>_<eval_level>_<latents>/
        metrics_summary.csv   report_<category>.csv   metrics.json
        predictions.csv

Writes four files by default:
    Table_2_classif_eval.csv            top-1 / top-5 accuracy for genus and
                                        spinal position, LOO vs LOSO
    Table_S3_classification_stats.csv   one row per run x split x eval_level x
                                        latents x level -- the minimum needed to
                                        recompute everything quoted in the paper
    Table_S4_classif_by_spec_num.csv    per-class recall vs how much data each
                                        class had, from predictions.csv
    Table_S5_classif_by_spec_spearmans.csv  the correlations behind Table S4

Table 2 and Table S3 cover every run given. S4 and S5 cover run_v72 only and go
in a subfolder of --outdir named for it.

Usage
-----
Main analysis

python classification_tables.py --roots ../run_v72 ../run_v73h ../run_v73c

Downsample experiment, from the common validation meshes:

python classification_tables.py --roots ../run_10spec ../run_30spec ../run_50spec ../run_70spec ../run_v72 \
    --table-splits downsample_val_common --no-supplement \
    --name Table_downsample_classif.csv --outdir ../downsample_tables
    
Only the reachable metrics are tabulated: under specimen masking a class with a
single specimen has no same-label gallery entry left once that specimen is
hidden, so its recall is forced to 0. LOO and LOSO therefore choose among
different numbers of classes, which is why each cell carries its own n and
class count.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from NSM.evaluation import (EVAL_LABELS, RUN_LABELS, SPLIT_ORDER, class_support,
                            correlation_rows, find_results, normalize_labels,
                            write_supplement)

# Main root used to report metrics for Tables S4-S5
MAIN_ROOT = "run_v72"

# written relative to NSM/nsm/paper
OUTDIR = "../classification_tables"

# carried through from metrics_summary.csv, in report order
METRICS = ["n_eval", "n_classes", "n_eligible", "n_reachable_classes",
           "top1_accuracy", "top5_accuracy",
           "top1_accuracy_reachable", "top5_accuracy_reachable",
           "macro_f1", "macro_f1_reachable", "weighted_f1"]

LEVELS = ["family", "genus", "species", "region", "position_10", "position_20",
          "life_history"]

# the paper table: which levels, and how they are titled in it
TABLE_LEVELS = {"genus": "Genus", "position_20": "Spinal position (20% bins)"}
TABLE_SPLITS = ("train", "test", "val") 
TABLE_EVALS = ("loo", "specimen")
TABLE_LATENTS = "latent_opt"

S3_NAME = "Table_S3_classification_stats.csv"
T2_NAME = "Table_2_classif_eval.csv"

# the condition Tables S4 and S5 report, matching the published figure
SUPP_CATEGORY = "broad_taxon"
SUPP_EVAL = "specimen"
SUPP_LATENTS = "latent_opt"

# Define functions
def load_summary(res, levels):
    """metrics_summary.csv for one evaluation, tagged with its condition."""
    df = pd.read_csv(res.path)
    if "category" not in df.columns:
        print(f"  skipping {res.path}: no 'category' column", file=sys.stderr)
        return None
    df = df[df["category"].isin(levels)].copy()
    if df.empty:
        return None
    eval_level = res.eval_level
    jpath = os.path.join(os.path.dirname(res.path), "metrics.json")
    if os.path.exists(jpath):
        try:
            eval_level = json.load(open(jpath)).get("eval_level", eval_level)
        except (json.JSONDecodeError, OSError):
            pass

    df = df.rename(columns={"category": "level"})
    df.insert(0, "run", res.run)
    df.insert(1, "split", res.split)
    df.insert(2, "eval_level", eval_level)
    df.insert(3, "latents", res.latents)
    for c in METRICS:
        if c not in df.columns:
            df[c] = np.nan
    return df[["run", "split", "eval_level", "latents", "level"] + METRICS]

def build_summary(roots, levels, quiet=False):
    frames = []
    for res in find_results(roots, "metrics_summary.csv"):
        if not quiet:
            print(f"found {res.run:10s} {res.split:6s} {res.eval_level}_{res.latents}")
        df = load_summary(res, levels)
        if df is not None:
            frames.append(df)
    if not frames:
        sys.exit("No metrics_summary.csv files found. Check --roots.")

    rank = {"split": SPLIT_ORDER, "level": {lv: i for i, lv in enumerate(levels)}}
    summary = pd.concat(frames, ignore_index=True)
    return summary.sort_values(
        ["split", "eval_level", "latents", "level", "run"],
        key=lambda c: c.map(rank[c.name]) if c.name in rank else c).reset_index(drop=True)


def paper_table(summary, levels=TABLE_LEVELS, latents=TABLE_LATENTS,
                splits=TABLE_SPLITS, evals=TABLE_EVALS):
    """Stacked 'top1 / top5' blocks, one per level, in the published layout.

    Rows are conditions, columns are split x masking level. The first row of
    each block carries the query count and the reachable-class count, which
    differ between LOO and LOSO and so cannot be folded into a single caption.
    """
    d = summary[(summary["latents"] == latents)
                & (summary["eval_level"].isin(evals))]
    if d.empty:
        print(f"  no rows with latents={latents}; skipping {T2_NAME}",
              file=sys.stderr)
        return None

    blocks = []
    for level, level_label in levels.items():
        sub = d[d["level"] == level]
        if sub.empty:
            continue
        sub = sub.assign(Condition=sub["run"].map(lambda r: RUN_LABELS.get(r, r)))
        TOP1, TOP5 = "top1_accuracy_reachable", "top5_accuracy_reachable"
        N, NCLS = "n_eligible", "n_reachable_classes"
        p = sub.pivot_table(index="Condition", columns=["split", "eval_level"],
                            values=[TOP1, TOP5, N, NCLS])
        cols = {f"{s.capitalize()} {EVAL_LABELS.get(e, e)}": (s, e)
                for s in splits for e in evals if (TOP1, s, e) in p.columns}

        known = [c for c in RUN_LABELS.values() if c in p.index]
        conditions = known + [c for c in p.index if c not in known]

        def cell(c, k):
            if pd.isna(p.loc[c, (TOP1, *k)]):
                return ""
            return (f"{p.loc[c, (TOP1, *k)] * 100:.1f} / "
                    f"{p.loc[c, (TOP5, *k)] * 100:.1f} "
                    f"(n={int(p.loc[c, (N, *k)])}, "
                    f"{int(p.loc[c, (NCLS, *k)])} cls)")

        rows = [{"Level": level_label, "Condition": c,
                 **{name: cell(c, k) for name, k in cols.items()}}
                for c in conditions]
        blocks.append(pd.DataFrame(rows))
    return pd.concat(blocks, ignore_index=True) if blocks else None

def build_supplement(roots, outdir, quiet=False):
    """Tables S4 and S5, from the predictions of one condition.

    Every split is included: the correlation between recall and specimen count
    is only interpretable read across train, val and test together.
    """
    per_class, corr = [], []
    for res in find_results(roots, "predictions.csv", eval_level=SUPP_EVAL,
                            latents=SUPP_LATENTS):
        df = normalize_labels(pd.read_csv(res.path), quiet=quiet)
        sup = class_support(df, SUPP_CATEGORY)
        if sup is None or sup.empty:
            continue
        meta = {"run": res.run, "split": res.split}
        per_class.append(sup.assign(**meta))
        corr.extend(correlation_rows(sup, meta))
    if not per_class:
        print(f"  no predictions.csv with eval_level={SUPP_EVAL} "
              f"latents={SUPP_LATENTS}; skipping S4 and S5", file=sys.stderr)
    return write_supplement(per_class, corr, outdir, quiet)

def write_tables(summary, outdir, name, splits, verbose=True):
    """Table 2 and S3."""
    os.makedirs(outdir, exist_ok=True)
    summary.to_csv(os.path.join(outdir, S3_NAME), index=False)
    written = [S3_NAME]

    t2 = paper_table(summary, splits=splits)
    if t2 is not None:
        t2.to_csv(os.path.join(outdir, name), index=False)
        written.append(name)
        if verbose:
            print(f"\n{'=' * 96}\nTABLE 2 - top-1 / top-5 accuracy (%), "
                  f"reachable classes\n{'=' * 96}")
            print(t2.to_string(index=False))
    return written

# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True,
                    help="run directories, e.g. ../run_v72 ../run_v73h ../run_v73c")
    ap.add_argument("--outdir", default=OUTDIR,
                    help=f"where to write the tables (default: {OUTDIR}); S4 and "
                         f"S5 go in a {MAIN_ROOT} subfolder")
    ap.add_argument("--name", default=T2_NAME,
                    help=f"paper table filename (default: {T2_NAME})")
    ap.add_argument("--table-splits", nargs="+", default=list(TABLE_SPLITS),
                    help=f"splits in the paper table (default: {' '.join(TABLE_SPLITS)})")
    ap.add_argument("--no-supplement", action="store_true",
                    help="skip Tables S4 and S5")
    ap.add_argument("--quiet", action="store_true",
                    help="write files without printing")
    args = ap.parse_args()

    summary = build_summary(args.roots, LEVELS, args.quiet)
    written = write_tables(summary, args.outdir, args.name, args.table_splits,
                           not args.quiet)
    print(f"\nWrote {', '.join(written)} to {args.outdir}")

    if args.no_supplement:
        return
    main_root = next((r for r in args.roots
                      if os.path.basename(r.rstrip("/")) == MAIN_ROOT), None)
    if main_root is None:
        sys.exit(f"{MAIN_ROOT} not in --roots; needed for Tables S4 and S5.")
    supp_dir = os.path.join(args.outdir, MAIN_ROOT)
    os.makedirs(supp_dir, exist_ok=True)
    supp = build_supplement([main_root], supp_dir, args.quiet)
    print(f"Wrote {', '.join(supp)} to {supp_dir}")

if __name__ == "__main__":
    main()