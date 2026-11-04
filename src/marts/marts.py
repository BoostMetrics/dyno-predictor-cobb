# @File    : marts.py
# @Author  : BoostMetrics
# @Date    : 2026-11-03
"""Build the run-level marts table from the clean dyno data"""

from pathlib import Path
import pandas as pd
import numpy as np

CLEAN_DIR = Path("data") / "clean"
AUC_RPM_MIN = 2500
AUC_RPM_MAX = 6000


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


def main() -> None:
    """Import files and print shape"""
    car_info = pd.read_parquet(CLEAN_DIR / "car_info.parquet")
    dyno_runs = pd.read_parquet(CLEAN_DIR / "dyno_runs.parquet")
    peaks = compute_peaks(dyno_runs)
    auc = compute_auc(dyno_runs)
    print(f"Peaks: {peaks}")
    print(f"Car_info shape: {car_info.shape}")
    print(f"Dyno runs shape: {dyno_runs.shape}")
    print(f"AUC shape:{auc.shape}")
    print(f"Powerband AUC:{auc["powerband_auc"].notna().sum()}")


if __name__ == "__main__":

    main()
