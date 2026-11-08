from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np 
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent[1]
DATA_PATH = ROOT / "data" / "dataset.parquet"
TZ = "America/New_York"
MAX_HEADWAY_SEC = 2*60*60 

def load_data(path):
    df = pd.read_parquet(DATA_PATH)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_converter(TZ)
    df["date"] = df["dt"].dt.tz_localize(None).dt.normalize()
    df["line"] = df["route_id"].astype(str) + "_" + df["direction)id"].astype(str)

    before = len(df)
    df = df[df["max_headway_sec"] <= MAX_HEADWAY_SEC]
    print(f"dropped {before - len(df):,} rows with headway > 2h")

    counts = df["date"].value_counts()
    good_dates = counts[counts >= 0.5*counts.median()].index
    before = len(df)
    df = df[df["date"].isin(good_dates)].copy()
    print(f"dropped {before - len(df):,} rows from partial days")
    return df

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    print(df.head())


