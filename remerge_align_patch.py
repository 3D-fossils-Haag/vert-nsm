# Re-merge saved inference summaries with alignment logs (fills in rms_err_pct without rerunning)
import os
import pandas as pd

TRAIN_DIR = "run_v72"                                   # TO DO
OUT_ROOT = "fine_tuning_encoder"                        # TO DO: "fine_tuning" if fast_mode was False
SPLIT = "train"                                         # TO DO
PERTURB_DIR = "perturbed"                               # TO DO: or "perturbed_preprocessed"
STAGES = ["aligned_mw_surf_per_foss_cover", "aligned_mw_lms_per_foss_cover"]   # TO DO: stages to fix
POSE_SUFFIX = "_pose"                                   # TO DO: "" for runs without pose correction, "_pose_low" for LOW_COMPLETE

for stage in STAGES:
    per_foss = "per_foss_cover" in stage
    mw_mode = "lms" if "_lms" in stage else "surf"
    mw_log = f"{PERTURB_DIR}/pose_log_mw{'_per_foss_cover' if per_foss else ''}.csv"
    run_dir = f"{OUT_ROOT}_{PERTURB_DIR}_{stage}_models{POSE_SUFFIX}"
    outdir = os.path.join(TRAIN_DIR, "shape_completion", run_dir, SPLIT)
    run_name = f"{run_dir}_{SPLIT}"

    summary_df = pd.read_csv(os.path.join(outdir, "best_cfg_predictions", "inference_summary.csv"))
    log = pd.read_csv(f"{PERTURB_DIR}/perturb_log.csv")
    mw = pd.read_csv(mw_log)
    mw = mw[mw["mode"] == mw_mode][["fossil", "rms_err_pct"]]
    log = log.drop(columns=["rms_err_pct", "rot_err_deg", "scale_err_pct"]).merge(mw, on="fossil", how="left")
    log["key"] = log.fossil.str.replace("_hollow_cut", "").str.replace(".vtk", "")
    summary_df["key"] = summary_df.mesh.str.replace("_hollow_cut", "").str.replace("_align", "").str.replace(".vtk", "")
    merged = summary_df.merge(log, on="key")
    merged.to_csv(os.path.join(outdir, "best_cfg_predictions", "inference_vs_alignment.csv"), index=False)
    merged.to_csv(os.path.join(TRAIN_DIR, "shape_completion", f"inference_vs_alignment_{run_name}.csv"), index=False)

    cols = ["rms_err_pct", "n_used"]
    corr = pd.DataFrame({"chamfer_intact": merged[cols].corrwith(merged.chamfer_intact, method="spearman"),
                         "chamfer_intact_shape": merged[cols].corrwith(merged.chamfer_intact_shape, method="spearman")})
    print(f"\n{stage}: {merged.rms_err_pct.notna().sum()}/{len(merged)} with rms_err_pct")
    print(f"\033[32m{corr.round(3).to_string()}\033[0m")