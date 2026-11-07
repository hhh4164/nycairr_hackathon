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

    df["timestamp"] = df["event_time"].astype("int64")

    df["datetime"] = pd.to_datetime(
        df["timestamp"], unit="s", utc=True
    ).dt.tz_convert("America/New_York")

    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday

    out = (
        df.groupby(
            ["timestamp", "route_id", "direction_id", "stop_id"],
            as_index=False
        )
        .agg(
            avg_headway_sec=("headway_sec", "mean"),
            max_headway_sec=("headway_sec", "max"),
            n_trains=("trip_uid", "count"),
            hour=("hour", "first"),
            weekday=("weekday", "first")
        )
    )

    out.to_parquet(
        INTERIM / f"headways_{day}.parquet",
        index=False
    )

    return out

def build_dataset(days):
    daily_data = []

    for day in days:
        print(f"Processing {day}...")

        download_day(day)
        daily_data.append(process_day(day))

    dataset = pd.concat(daily_data, ignore_index=True)

    dataset = dataset.sort_values(
        ["timestamp", "route_id", "stop_id"]
    )

    dataset.to_parquet(
        "data/dataset.parquet",
        index=False
    )

    print("\nDataset complete:")
    print(dataset.shape)
    print(dataset.dtypes)

    return dataset

if __name__ == "__main__":
    days = [
        "2026-11-01",
        "2026-11-02",
        "2026-11-03",
        "2026-11-04",
        "2026-11-05"
    ]

    dataset = build_dataset(days)