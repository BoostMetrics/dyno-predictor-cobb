# @File    : clean.py
# @Author  : BoostMetrics
# @Date    : 2026-11-02
"""Clean the raw COBB Tuning Dyno tables and save them into data/clean."""

import hashlib
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

RAW_DIR = Path("data") / "raw"
CLEAN_DIR = Path("data") / "clean"

# Run 5933: 4936 points vs 140-330 for a normal run (atypical recording).
EXCLUDED_RUNS = [5933]


def anonymize(name: str, salt: str) -> str:
    """Return the salted SHA-256 hash of an owner name."""
    return hashlib.sha256((salt + name).encode()).hexdigest()


def clean_car_info(raw: pd.DataFrame, salt: str) -> pd.DataFrame:
    """Parse Car and Specs, anonymize Name and drop the raw text columns."""
    df = raw.copy()
    specs = df["Specs"]

    # Vehicle
    df["vehicle_year"] = df["Car"].str.extract(r"^(\d{4})")[0].astype("Int64")
    df["vehicle_make"] = df["Car"].str.extract(r"^\d{4} (\S+)")[0]
    df["vehicle_model"] = df["Car"].str.extract(r"^\d{4} \S+ (.+)$")[0]
    df["vehicle_key"] = df["vehicle_year"].astype(str) + "|" + df["vehicle_model"]
    df["date_test"] = pd.to_datetime(
        df["Date"], format="%m-%d-%Y %I:%M %p", errors="coerce"
    )

    # Modifications
    df["boost_target"] = (
        specs.str.extract(r"(\d+(?:\.\d+)?)\s*psi", flags=re.IGNORECASE)[0]
    ).astype(float)
    df["fuel_octane"] = (
        specs.str.extract(r"(\d{2,3})\s*oct", flags=re.IGNORECASE)[0]
    ).astype(float)
    df["is_e85"] = specs.str.contains(r"e-?85|flex", case=False)
    df["turbo_up"] = specs.str.contains(r"gt\d{2}|evo ?3|\d{2}g\b", case=False)
    df["has_exhaust_up"] = specs.str.contains(
        r"catback|downpipe|turboback|header|catless|pipe", case=False
    )
    df["has_intake_up"] = specs.str.contains(r"intake|airbox|cai|sri", case=False)

    # Tuning stage: "stage N" in the text, else 0 if stock with no parts and no psi
    df["tuning_stage"] = (
        specs.str.extract(r"stage\s*(\d)", flags=re.IGNORECASE)[0]
    ).astype("Int64")
    is_stock = specs.str.strip().str.lower().str.contains(r"stock|baseline")
    no_parts = ~(df["turbo_up"] | df["has_exhaust_up"] | df["has_intake_up"])
    stock_mask = (
        df["tuning_stage"].isna() & is_stock & no_parts & df["boost_target"].isna()
    )
    df.loc[stock_mask, "tuning_stage"] = 0
    df.loc[df["tuning_stage"] > 5, "tuning_stage"] = pd.NA

    # Owner
    df["owner_key"] = df["Name"].apply(lambda name: anonymize(name, salt))

    return df.drop(columns=["Date", "Car", "Name"]).rename(columns={"Run": "run"})


def clean_dyno_runs(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid points, exclude atypical runs and merge duplicated points."""
    df = raw.drop(columns="Unnamed: 0")
    df = df[(df["RPM"] > 0) & (df["HP"] > 0) & (df["Torque"] > 0)]
    df = df.rename(
        columns={
            "Run": "run",
            "RPM": "rpm",
            "HP": "hp",
            "Torque": "tq",
            "AFR": "afr",
            "Boost": "boost",
        }
    )
    df = df[~df["run"].isin(EXCLUDED_RUNS)]

    # Same (run, rpm) measured several times: average the measures
    return df.groupby(["run", "rpm"], as_index=False).agg(
        hp=("hp", "mean"),
        tq=("tq", "mean"),
        afr=("afr", "mean"),
        boost=("boost", "mean"),
        tec_load_ts=("tec_load_ts", "first"),
    )


def main() -> None:
    """Read data/raw, clean both tables, keep common runs and write data/clean."""
    load_dotenv()
    salt = os.environ["SEL"]

    car_info = clean_car_info(pd.read_parquet(RAW_DIR / "car_info.parquet"), salt)
    dyno_runs = clean_dyno_runs(pd.read_parquet(RAW_DIR / "dyno_runs.parquet"))

    # Keep only runs present in both tables
    common = set(car_info["run"]) & set(dyno_runs["run"])
    car_info = car_info[car_info["run"].isin(common)]
    dyno_runs = dyno_runs[dyno_runs["run"].isin(common)]

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    car_info.to_parquet(CLEAN_DIR / "car_info.parquet", index=False)
    dyno_runs.to_parquet(CLEAN_DIR / "dyno_runs.parquet", index=False)
    print(f"car_info: {len(car_info)} rows | dyno_runs: {len(dyno_runs)} rows")
    print(f"{dyno_runs['run'].nunique()} runs kept")


if __name__ == "__main__":
    main()
