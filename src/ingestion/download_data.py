# @File    : download_data.py
# @Author  : BoostMetrics
# @Date    : 2026-11-02
"""Download the COBB Tuning Dyno dataset from Kaggle into data/raw."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import kagglehub
import pandas as pd

DATASET = "paxtonjacob/cobb-tuning-dyno-data"
FILES = ("car_info.csv", "dyno_runs.csv")
RAW_DIR = Path("data") / "raw"


def download_raw() -> None:
    """Download the Kaggle CSV files and save them as Parquet with tec_load_ts."""
    source = Path(kagglehub.dataset_download(DATASET))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tec_load_ts = datetime.now(ZoneInfo("Europe/Paris"))

    for name in FILES:
        df = pd.read_csv(source / name)
        df["tec_load_ts"] = tec_load_ts
        output = RAW_DIR / name.replace(".csv", ".parquet")
        df.to_parquet(output, index=False)
        print(f"{name}: {len(df)} rows -> {output}")


if __name__ == "__main__":
    download_raw()
