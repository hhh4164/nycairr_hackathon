from datetime import datetime, timedelta
from datetime import date as date_type
from zoneinfo import ZoneInfo
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from train.process import EXCLUDED_LINES, download_day, process_day
from train.train import (MODEL_PATH, MAX_HEADWAY_SEC, DISRUPT_RATIO, DISRUPT_EXTRA_SEC, TZ, make_hourly, make_X)

#currently data a day behind
def get_predictions(target_date: date_type | str | None = None) -> pd.DataFrame:
    now = datetime.now(ZoneInfo(TZ))
    if target_date is None:
        prediction_date = pd.Timestamp(now.date())
        day = (now - timedelta(hours=36)).strftime("%Y-%m-%d")
    else:
        prediction_date = pd.Timestamp(target_date).normalize()
        day = (prediction_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    normal = bundle["normal"]
    normal = normal[~normal["line"].isin(EXCLUDED_LINES)].copy()

    download_day(day)
    df = process_day(day)

    df = df[~df["line"].isin(EXCLUDED_LINES)]
    df = df[df["max_headway_sec"] <= MAX_HEADWAY_SEC].copy()
    df["date"] = df["timestamp"].dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()

    hourly = make_hourly(df)
    hourly = hourly[hourly["date"] == pd.Timestamp(day)].copy()
    hourly = hourly.merge(normal, on=["line", "direction_id", "stop_id", "hour", "is_weekend"], how="left")

    valid = hourly["max_headway"].notna() & hourly["normal_max_headway"].notna()
    worse = hourly["max_headway"] >= DISRUPT_RATIO * hourly["normal_max_headway"]
    longer = hourly["max_headway"] >= hourly["normal_max_headway"] + DISRUPT_EXTRA_SEC
    hourly["y_disrupted"] = np.where(valid, (worse & longer).astype(float), np.nan)

    yesterday_features = hourly[["line", "direction_id", "stop_id", "hour", "max_headway", "n_arrivals", "y_disrupted"]].rename(columns={
        "max_headway": "y_max_headway",
        "n_arrivals": "y_n_arrivals"
    })

    daily_features = hourly.groupby(["line", "direction_id", "stop_id"])["y_disrupted"].sum(min_count=1).reset_index(name="y_disrupted_hours")

    combinations = normal[["line", "direction_id", "stop_id"]].drop_duplicates().reset_index(drop=True)
    predictions = combinations.merge(pd.DataFrame({"hour": range(24)}), how="cross")
    predictions["is_weekend"] = int(prediction_date.dayofweek >= 5)

    predictions = predictions.merge(yesterday_features, on=["line", "direction_id", "stop_id", "hour"], how="left")
    predictions = predictions.merge(daily_features, on=["line", "direction_id", "stop_id"], how="left")
    probabilities = model.predict_proba(make_X(predictions))[:, 1]

    stop_names = pd.read_csv(
        Path(__file__).resolve().parent / "stops.txt",
        dtype=str
    ).set_index("stop_id")["stop_name"]
    predictions = pd.DataFrame({
        "line": predictions["line"].values,
        "direction_id": predictions["direction_id"].values,
        "stop_id": predictions["stop_id"].values,
        "stop_name": predictions["stop_id"].astype(str).map(stop_names).fillna(predictions["stop_id"].astype(str)).values,
        "hour": predictions["hour"].values,
        "probability": probabilities,
        "data_day": day,
        "prediction_date": prediction_date
    })

    return predictions.sort_values(["line", "direction_id", "hour", "stop_id"]).reset_index(drop=True)


if __name__ == "__main__":
    predictions = get_predictions()
    print(predictions.to_string(index=False))
