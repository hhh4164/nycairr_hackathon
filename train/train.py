from pathlib import Path
from train.process import EXCLUDED_LINES
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "dataset.parquet"
TZ = "America/New_York"
MAX_HEADWAY_SEC = 2*60*60 
DISRUPT_RATIO = 2.5
DISRUPT_EXTRA_SEC = 10*60
FEATURES = ["line", "direction_id", "hour", "is_weekend", "y_max_headway", "y_n_arrivals", "y_disrupted", "y_disrupted_hours"]
LABEL = "disrupted"
TEST_DAYS = 14
MODEL_PATH = ROOT/ "models" / "model.joblib"

def load_data(path):
    df = pd.read_parquet(path)
    df = df[~df["line"].isin(EXCLUDED_LINES)].copy()
    df["dt"] = df["timestamp"].dt.tz_convert(TZ)
    df["date"] = df["dt"].dt.tz_localize(None).dt.normalize()

    before = len(df)
    df = df[df["max_headway_sec"] <= MAX_HEADWAY_SEC]
    print(f"dropped {before - len(df):,} rows with headway > 2h")

    counts = df["date"].value_counts()
    good_dates = counts[counts >= 0.5*counts.median()].index
    before = len(df)
    df = df[df["date"].isin(good_dates)].copy()
    print(f"dropped {before - len(df):,} rows from partial days")
    return df

def make_hourly(df):
    hourly = (
        df.groupby(["line", "direction_id", "date", "hour"], as_index=False).agg(
            avg_headway = ("avg_headway_sec", "mean"),
            max_headway = ("max_headway_sec", "max"),
            n_arrivals = ("n_trains", "sum")
        )
    )
    hourly["weekday"] = hourly["date"].dt.day_of_week
    hourly["is_weekend"] = (hourly["weekday"] >= 5).astype(int)
    return hourly

def add_disruption_features(hourly):
    normal = (
        hourly.groupby(["line", "direction_id", "hour", "is_weekend"], as_index=False)["max_headway"]
            .median()
            .rename(columns={"max_headway": "normal_max_headway"})
    )
    hourly = hourly.merge(normal, on=["line", "direction_id", "hour", "is_weekend"], how="left")
    much_worse = hourly["max_headway"] >= (DISRUPT_RATIO * hourly["normal_max_headway"])
    much_longer = hourly["max_headway"] >= hourly["normal_max_headway"] + DISRUPT_EXTRA_SEC
    hourly["disrupted"] = (much_worse & much_longer).astype(int)
    return hourly, normal

def add_yesterday_features(hourly):
    yesterday = hourly[["line", "direction_id", "date", "hour", "max_headway", "n_arrivals", "disrupted"]].copy()
    yesterday["date"] = yesterday["date"] + pd.Timedelta(days=1)
    yesterday = yesterday.rename(columns = {
        "max_headway": "y_max_headway",
        "n_arrivals": "y_n_arrivals",
        "disrupted": "y_disrupted"
    })
    hourly = hourly.merge(yesterday, on=["line", "direction_id", "date", "hour"], how="left")
    daily = (
        hourly.groupby(["line", "direction_id", "date"], as_index=False)["disrupted"].sum()
            .rename(columns={"disrupted": "y_disrupted_hours"})
    )
    daily["date"] = daily["date"] + pd.Timedelta(days=1)
    hourly = hourly.merge(daily, on=["line", "direction_id", "date"], how="left")
    hourly = hourly[hourly["date"] > hourly["date"].min()].copy()
    return hourly

def split_by_date(hourly):
    dates = sorted(hourly["date"].unique())
    test_dates = dates[-TEST_DAYS:]
    train = hourly[~hourly["date"].isin(test_dates)].copy()
    test = hourly[hourly["date"].isin(test_dates)].copy()
    return train, test

def make_X(d):
    X = d[FEATURES].copy()
    X["line"] = X["line"].astype("category")
    X["direction_id"] = X["direction_id"].astype("category")
    return X

def train_model(train):
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        verbose=-1
    )
    model.fit(make_X(train), train[LABEL])
    return model

def evaluate(model, test):
    prob = model.predict_proba(make_X(test))[:,1]
    baseline = test["y_disrupted"].fillna(0)
    print("base rate (random guess):", round(test[LABEL].mean(), 3))
    print("baseline PR-AUC (yesterday):", round(average_precision_score(test[LABEL], baseline), 3))
    print("model PR-AUC:", round(average_precision_score(test[LABEL], prob), 3))

def save_model(model, normal):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model":model, "features": FEATURES, "normal": normal}, MODEL_PATH)
    print("model saved to", MODEL_PATH)



if __name__ == "__main__":
    df = load_data(DATA_PATH)
    hourly = make_hourly(df)
    hourly, normal = add_disruption_features(hourly)
    hourly = add_yesterday_features(hourly)
    train, test = split_by_date(hourly)
    print(f"train: {len(train)} rows, test: {len(test)} rows")

    model = train_model(train)
    evaluate(model, test)

    final_model = train_model(hourly)     # retrain on ALL days before saving
    save_model(final_model, normal)

    print("\nRows per line-direction combination:")
    print(hourly.groupby(["line", "direction_id"]).size().sort_index())

    print("\nHourly row counts:")
    print(hourly.groupby(["line", "direction_id", "date"]).size().value_counts())

