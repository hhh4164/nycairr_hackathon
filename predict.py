from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import joblib
import pandas as pd

from train.process import download_day, process_day
from train.train import (MODEL_PATH, MAX_HEADWAY_SEC, DISRUPT_RATIO, DISRUPT_EXTRA_SEC, TZ, make_hourly, make_X)



#currently data a day behind
def get_predictions() -> pd.DataFrame:
    now = datetime.now(ZoneInfo(TZ))
    day = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    normal = bundle["normal"]

    download_day(day)
    df = process_day(day)

    df = df[df["max_headway_sec"] <= MAX_HEADWAY_SEC].copy()
    df["date"] = (df["timestamp"].dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize())

    hourly = make_hourly(df)
    target_date = pd.Timestamp(day)
    hourly = hourly[hourly["date"] == target_date].copy()
    hourly = hourly.merge(normal, on=["line", "hour", "is_weekend"], how="left")

    worse = (hourly["max_headway"]>=DISRUPT_RATIO * hourly["normal_max_headway"])
    longer = (hourly["max_headway"]>=hourly["normal_max_headway"]+DISRUPT_EXTRA_SEC)

    hourly["y_disrupted"] = (worse & longer).astype(int)

    #total disrupted hours yesterday per line
    hourly["y_disrupted_hours"] = (hourly.groupby("line")["y_disrupted"].transform("sum"))
    #yesterday data for today prediction
    hourly = hourly.rename(columns={"max_headway": "y_max_headway", "n_arrivals": "y_n_arrivals"})

    hourly["is_weekend"] = int(now.weekday()>=5)
    probabilities = model.predict_proba(make_X(hourly))[:,1]

    predictions = pd.DataFrame({
        "line": hourly["line"].values,
        "hour": hourly["hour"].values,
        "risk": probabilities,
        "data_day": day
    })

    return predictions.sort_values(["line", "hour"]).reset_index(drop=True)


if __name__ == "__main__":
    predictions = get_predictions()
    print(predictions.to_string(index=False))

