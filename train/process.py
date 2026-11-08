import io, tarfile, requests
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/subway")
RAW.mkdir(parents=True, exist_ok=True)
INTERIM = Path("data/interim")
INTERIM.mkdir(parents=True, exist_ok=True)

def download_day(day: str):
    url = f"https://subwaydata.nyc/data/subwaydatanyc_{day}_csv.tar.xz"
    req = requests.get(url, timeout=120)
    req.raise_for_status() #error
    with tarfile.open(fileobj=io.BytesIO(req.content), mode="r:xz") as tar:
        tar.extractall(RAW)

def process_day(day: str) -> pd.DataFrame:
    trips = pd.read_csv(
        RAW / f"subwaydatanyc_{day}_trips.csv",
        usecols=["trip_uid", "route_id", "direction_id"],
        dtype={
            "route_id": str,
            "trip_uid": str
        }
    )

    st = pd.read_csv(
        RAW / f"subwaydatanyc_{day}_stop_times.csv",
        usecols=[
            "trip_uid",
            "stop_id",
            "arrival_time",
            "departure_time"
        ],
        dtype={
            "trip_uid": str,
            "stop_id": str
        }
    )

    df = st.merge(trips, on="trip_uid", how="inner")

    df["event_time"] = (
        df["arrival_time"].fillna(df["departure_time"])
    )

    df = df.dropna(subset=["event_time"])

    keys = ["route_id", "direction_id", "stop_id"]

    df = df.sort_values(keys + ["event_time"])

    df["headway_sec"] = (
        df.groupby(keys)["event_time"].diff()
    )

    df = df.dropna(subset=["headway_sec"])

    df = df[df["headway_sec"] >= 0].copy()
    df["timestamp"] = pd.to_datetime(
        df["event_time"], unit="s", utc=True).dt.floor("5min")
    out = (
        df.groupby(["timestamp", "route_id"], as_index=False).agg(
            avg_headway_sec = ("headway_sec", "mean"),
            max_headway_sec = ("headway_sec", "max"),
            n_trains = ("trip_uid", "nunique"),
        ).rename(columns = {"route_id": "line"}))
    local = out["timestamp"].dt.tz_convert("America/New_York")
    out["hour"] = local.dt.hour
    out["weekday"] = local.dt.day_of_week
    out.to_parquet(INTERIM / f"bins_{day}.parquet", index=False)
    return out

def build_dataset(days):
    daily = []
    for day in days:
        cached = INTERIM / f"bins_{day}.parquet"
        if cached.exists():
            daily.append(pd.read_parquet(cached))
            continue
        try:
            print(f"Processing {day}")
            download_day(day)
            daily.append(process_day(day))
        except Exception as e:
            print(f"Skipping {day}: {e}")
            continue
        for f in RAW.glob(f"subwaydatanyc_{day}_*.csv"):
            f.unlink()
    dataset = pd.concat(daily, ignore_index=True).sort_values(["timestamp", "line"])
    dataset.to_parquet("data/dataset.parquet", index=False)
    print("\nDataset complete:", dataset.shape)
    print(dataset.dtypes)
    return dataset

if __name__ == "__main__":
    days = pd.date_range("2026-09-1", "2026-11-01").strftime("%Y-%m-%d").tolist()
    dataset = build_dataset(days)
    print(dataset["max_headway_sec"].describe(percentiles=[.5, .9, .99]))
    print(dataset.groupby("line")["avg_headway_sec"].median().sort_values())
