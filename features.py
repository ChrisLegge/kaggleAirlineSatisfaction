"""Shared data loading + feature engineering."""
import numpy as np
import pandas as pd

DATA = "playground-series-s6e10"
SEED = 42
N_FOLDS = 5
TARGET = "satisfaction"

BASE_CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
RATINGS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]


def load():
    train = pd.read_csv(f"{DATA}/train.csv")
    test = pd.read_csv(f"{DATA}/test.csv")
    y = train[TARGET].astype(int).values
    return train.drop(columns=[TARGET]), test, y


def add_features(df):
    df = df.copy()
    r = df[RATINGS]
    # Rating aggregates (0 means "not applicable" in the original survey)
    df["rating_mean"] = r.replace(0, np.nan).mean(axis=1)
    df["rating_min"] = r.replace(0, np.nan).min(axis=1)
    df["rating_std"] = r.std(axis=1)
    df["n_zero"] = (r == 0).sum(axis=1)
    df["n_five"] = (r == 5).sum(axis=1)
    df["n_low"] = r.isin([1, 2]).sum(axis=1)
    # Ratings that matter most in the baseline, combined
    df["boarding_x_wifi"] = df["Online boarding"] * 10 + df["Inflight wifi service"]
    df["digital_mean"] = df[["Inflight wifi service", "Online boarding", "Ease of Online booking"]].mean(axis=1)
    df["comfort_mean"] = df[["Seat comfort", "Inflight entertainment", "Leg room service", "Cleanliness"]].mean(axis=1)

    # Delays
    arr = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])
    dep = df["Departure Delay in Minutes"]
    df["Arrival Delay in Minutes"] = arr
    df["delay_total"] = np.log1p(dep + arr)
    df["delay_gain"] = dep - arr  # time made up in the air
    df["any_delay"] = ((dep + arr) > 0).astype(int)

    # Categorical interactions
    df["class_travel"] = df["Class"] + "_" + df["Type of Travel"]
    df["cust_travel"] = df["Customer Type"] + "_" + df["Type of Travel"]
    return df


CAT_COLS = BASE_CATS + ["class_travel", "cust_travel"]


def load_original():
    orig = pd.read_csv("originalData/data.csv")
    y = orig[TARGET].astype(int).values
    return orig.drop(columns=[TARGET]), y


def build(cat_ratings=False):
    """Returns X, X_test, y, test_ids, cats, X_orig, y_orig.
    If cat_ratings, ratings are also added as string categoricals."""
    train, test, y = load()
    orig, y_orig = load_original()
    ids = test["id"].values
    cols = train.drop(columns=["id"]).columns
    X = add_features(train.drop(columns=["id"]))
    X_test = add_features(test.drop(columns=["id"]))
    X_orig = add_features(orig[cols])
    cats = list(CAT_COLS)
    if cat_ratings:
        for c in RATINGS:
            for df in (X, X_test, X_orig):
                df[c + "_cat"] = df[c].astype(str)
            cats.append(c + "_cat")
    return X, X_test, y, ids, cats, X_orig, y_orig
