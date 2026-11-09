# @File    : marts.py
# @Author  : BoostMetrics
# @Date    : 2026-11-03
"""Build the run-level marts table from the clean dyno data"""

from pathlib import Path
import pandas as pd
import numpy as np

MARTS_DIR = Path("data") / "marts"
CLEAN_DIR = Path("data") / "clean"
AUC_RPM_MIN = 2500
AUC_RPM_MAX = 6000
MIN_STOCK_RUN = 2
MAX_STOCK_SPREAD = 0.10
ROBUST_STOCK_RUNS = 3


def compute_peaks(dyno_runs: pd.DataFrame) -> pd.DataFrame:
    """Return one row per run with power and torque peaks"""
    pmax = dyno_runs.groupby("run")["hp"].idxmax()
    tmax = dyno_runs.groupby("run")["tq"].idxmax()
    tq_peak = dyno_runs.loc[tmax]
    tq_peak = tq_peak[["run", "tq", "rpm"]].rename(
        columns={"tq": "tq_peak", "rpm": "rpm_at_peak_tq"}
    )
    hp_peak = dyno_runs.loc[pmax]
    hp_peak = hp_peak[["run", "hp", "rpm"]].rename(
        columns={"hp": "hp_peak", "rpm": "rpm_at_peak_hp"}
    )

    hp_tq_max = pd.merge(hp_peak, tq_peak, on="run")
    return hp_tq_max


def compute_auc(dyno_runs: pd.DataFrame) -> pd.DataFrame:
    """Return one row per run with power AUC between AUC_RPM_MIN and AUC_RPM_MAX"""
    min_max_rpm = dyno_runs.groupby("run")["rpm"].agg(["min", "max"])
    covers = (min_max_rpm["min"] <= AUC_RPM_MIN) & (min_max_rpm["max"] >= AUC_RPM_MAX)
    rpm_window = dyno_runs[
        (dyno_runs["rpm"] >= AUC_RPM_MIN) & (dyno_runs["rpm"] <= AUC_RPM_MAX)
    ].sort_values(by=["run", "rpm"])
    trap_run = rpm_window.groupby("run").apply(
        lambda run_points: np.trapezoid(run_points["hp"], run_points["rpm"])
    )
    auc: pd.DataFrame = trap_run.where(covers).reset_index(name="powerband_auc")

    return auc


def build_run_summary(
    car_info: pd.DataFrame, peaks: pd.DataFrame, auc: pd.DataFrame
) -> pd.DataFrame:
    """Return one row per run with peaks and AUC"""
    car_peak = car_info.merge(peaks, on="run")
    run_summary = car_peak.merge(auc, on="run")
    return run_summary


def compute_gain_hp(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Return one row per run with stock perf"""
    stock_engine = run_summary[run_summary["tuning_stage"] == 0]
    stock_ref = stock_engine.groupby("vehicle_key")["hp_peak"].agg(
        ["median", "count", "min", "max"]
    )
    stock_valid = stock_ref[
        (stock_ref["count"] >= ROBUST_STOCK_RUNS)
        | (stock_ref["count"] == MIN_STOCK_RUN)
        & (
            (stock_ref["max"] - stock_ref["min"]) / stock_ref["median"]
            <= MAX_STOCK_SPREAD
        )
    ]
    ref = stock_valid["median"].rename("stock_ref_hp").reset_index()
    run_gain = run_summary.merge(ref, on="vehicle_key", how="left")
    run_gain["gain_hp"] = run_gain["hp_peak"] - run_gain["stock_ref_hp"]
    run_gain.loc[run_gain["tuning_stage"] == 0, "gain_hp"] = pd.NA

    return run_gain


def main() -> None:
    """Import files and print shape"""
    car_info = pd.read_parquet(CLEAN_DIR / "car_info.parquet")
    dyno_runs = pd.read_parquet(CLEAN_DIR / "dyno_runs.parquet")
    peaks = compute_peaks(dyno_runs)
    auc = compute_auc(dyno_runs)
    summary = build_run_summary(car_info, peaks, auc)
    run_summary_final = compute_gain_hp(summary)

    print(run_summary_final)

    MARTS_DIR.mkdir(parents=True, exist_ok=True)
    run_summary_final.to_parquet(MARTS_DIR / "run_summary.parquet", index=False)
    print(f"run_summary: {run_summary_final.shape}")


if __name__ == "__main__":

    main()
