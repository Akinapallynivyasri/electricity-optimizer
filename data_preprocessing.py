"""Load, clean and prepare electricity data for the LSTM and the dashboard.

Works with the simulated dataset from data/generate_data.py and also with the
UCI "Household Electric Power Consumption" file (semicolon separated, with
separate Date and Time columns and '?' for missing values).
"""
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_FILE = os.path.join(DATA_DIR, "electricity_data.csv")
DAILY_FILE = os.path.join(DATA_DIR, "daily_clean.csv")
HOURLY_PROFILE_FILE = os.path.join(DATA_DIR, "hourly_profile.csv")


# ── 1. LOAD DATA ─────────────────────────────────────────────
def load_data(filepath=None):
    """Read the raw CSV and index it by timestamp."""
    filepath = filepath or RAW_FILE
    print(f"Loading {os.path.relpath(filepath, BASE_DIR)} ...")

    # The UCI file uses ';' as separator, our own file uses ','.
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline()
    sep = ";" if header.count(";") > header.count(",") else ","

    df = pd.read_csv(filepath, sep=sep, na_values=["?"], low_memory=False)

    if "datetime" in df.columns:
        stamps = pd.to_datetime(df["datetime"])
    elif {"Date", "Time"} <= set(df.columns):          # UCI layout, dd/mm/yyyy
        stamps = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
    else:
        raise ValueError("Need a 'datetime' column, or 'Date' and 'Time' columns.")

    df = df.drop(columns=[c for c in ("datetime", "Date", "Time") if c in df.columns])
    df.index = pd.DatetimeIndex(stamps, name="datetime")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    print(f"Loaded {len(df)} records, {df.index.min()} -> {df.index.max()}")
    return df


# ── 2. CLEAN DATA ─────────────────────────────────────────────
def clean_data(df):
    """Make every column numeric and fill gaps in the power readings."""
    print("Cleaning data ...")
    df = df.replace("?", np.nan).apply(pd.to_numeric, errors="coerce")

    missing = int(df["Global_active_power"].isna().sum())
    print(f"Missing power readings: {missing}")

    # Carry the last known reading forward; bfill covers gaps at the very start.
    df["Global_active_power"] = df["Global_active_power"].ffill().bfill()
    return df


# ── 3. DAILY AGGREGATION ──────────────────────────────────────
def _hours_per_sample(index):
    """Length of one reading in hours (1.0 for hourly data, 1/60 for per-minute)."""
    step = index.to_series().diff().median()
    return step.total_seconds() / 3600


def get_daily_data(df):
    """Convert power readings (kW) to daily energy (kWh).

    Energy = power x time, so each reading is multiplied by how long it lasts.
    (The earlier version divided by 1000, which shrank every value 1000x.)
    """
    print("Aggregating to daily usage ...")
    power = df["Global_active_power"]
    hours = _hours_per_sample(df.index)

    daily_kwh = (power * hours).resample("D").sum()

    # Drop incomplete days at the very start/end of the data. A day with only a
    # few readings looks like a day of near-zero usage and ruins the metrics.
    expected = 24 / hours
    full = (power.resample("D").count() >= 0.9 * expected).to_numpy()
    first, last = 0, len(full)
    while first < last and not full[first]:
        first += 1
    while last > first and not full[last - 1]:
        last -= 1
    dropped = len(full) - (last - first)
    if dropped:
        print(f"Dropped {dropped} incomplete day(s) at the edges of the data")
    daily_kwh = daily_kwh.iloc[first:last]

    daily_df = pd.DataFrame({"usage_kwh": daily_kwh})

    # Keep the voltage if the dataset has it (the dashboard can chart it).
    volt_col = next((c for c in df.columns if c.lower() == "voltage"), None)
    if volt_col:
        daily_df["voltage"] = df[volt_col].resample("D").mean().loc[daily_df.index]

    daily_df = daily_df.rename_axis("date").reset_index()
    daily_df["day_of_week"] = daily_df["date"].dt.dayofweek
    daily_df["month"] = daily_df["date"].dt.month
    daily_df["is_weekend"] = daily_df["day_of_week"].isin([5, 6]).astype(int)

    print(f"Daily records: {len(daily_df)}, average {daily_df['usage_kwh'].mean():.2f} kWh/day")
    return daily_df


def get_hourly_profile(df):
    """Average energy used in each hour of the day (kWh), from 0:00 to 23:00."""
    return df["Global_active_power"].groupby(df.index.hour).mean().rename("avg_kwh")


# ── 4. CALCULATE BILL ─────────────────────────────────────────
def calculate_bill(units, days=30):
    """Estimate a bill from a slab tariff (Maharashtra-style slabs).

    Slab 1: first 100 units         -> Rs 3.46 / unit
    Slab 2: next 200 units (to 300) -> Rs 6.58 / unit
    Slab 3: above 300 units         -> Rs 9.56 / unit

    The slabs are *monthly*. For a shorter period (a day, a week) pass `days`
    and the slab limits are scaled down, so a single day isn't billed as if it
    were a whole month. The default (30 days) is the normal monthly bill.

    This is a simplified estimate: it leaves out fixed charges, duty and fuel
    adjustment, and the rates should be checked against the current tariff order.
    """
    scale = days / 30
    slab1, slab2 = 100 * scale, 300 * scale

    if units <= slab1:
        bill = units * 3.46
    elif units <= slab2:
        bill = slab1 * 3.46 + (units - slab1) * 6.58
    else:
        bill = slab1 * 3.46 + (slab2 - slab1) * 6.58 + (units - slab2) * 9.56
    return round(bill, 2)


# ── 5. PREPARE ML DATA ────────────────────────────────────────
def prepare_ml_data(daily_df, sequence_length=30, fit_scaler_on_train_only=True):
    """Build LSTM sequences: the past `sequence_length` days predict the next day.

    The scaler is fitted on the training period only, so no information from the
    test period leaks into training. Pass fit_scaler_on_train_only=False to scale
    with the whole series instead.
    """
    from sklearn.preprocessing import MinMaxScaler   # only needed for training

    print("Preparing ML sequences ...")
    data = daily_df["usage_kwh"].to_numpy().reshape(-1, 1)

    n_sequences = len(data) - sequence_length
    if n_sequences < 50:
        raise ValueError("Not enough daily data to train (need at least ~80 days).")

    split = int(n_sequences * 0.8)                 # 80% train / 20% test, in time order
    train_rows = sequence_length + split           # raw rows the training part can see

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[:train_rows] if fit_scaler_on_train_only else data)
    data_scaled = scaler.transform(data)

    X, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i - sequence_length:i, 0])
        y.append(data_scaled[i, 0])

    X = np.array(X).reshape(-1, sequence_length, 1)   # [samples, timesteps, features]
    y = np.array(y)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Training samples: {len(X_train)}, testing samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler


# ── 6. DETECT ANOMALIES ───────────────────────────────────────
def detect_anomalies(daily_df):
    """Flag days that use far more than usual (mean + 2 standard deviations).

    This is a simple statistical rule, not a machine-learning detector.
    """
    mean = daily_df["usage_kwh"].mean()
    std = daily_df["usage_kwh"].std()
    threshold = mean + 2 * std

    daily_df["is_anomaly"] = daily_df["usage_kwh"] > threshold
    anomalies = daily_df[daily_df["is_anomaly"]]

    print(f"Anomaly threshold: {threshold:.2f} kWh, anomaly days: {len(anomalies)}")
    return daily_df, anomalies


# ── PIPELINE HELPERS ──────────────────────────────────────────
def build_daily_dataset(raw_path=None):
    """Raw file -> (cleaned readings, daily table with anomaly flags and cost)."""
    df = clean_data(load_data(raw_path))
    daily_df = get_daily_data(df)
    daily_df, _ = detect_anomalies(daily_df)
    daily_df["bill"] = daily_df["usage_kwh"].apply(lambda u: calculate_bill(u, days=1))
    return df, daily_df


def run_preprocessing(raw_path=None):
    """Run the whole preprocessing step and save the files the dashboard reads."""
    df, daily_df = build_daily_dataset(raw_path)

    daily_df.to_csv(DAILY_FILE, index=False)
    get_hourly_profile(df).rename_axis("hour").to_csv(HOURLY_PROFILE_FILE)
    print(f"Saved {os.path.relpath(DAILY_FILE, BASE_DIR)} and "
          f"{os.path.relpath(HOURLY_PROFILE_FILE, BASE_DIR)}")
    return daily_df


if __name__ == "__main__":
    daily = run_preprocessing()
    prepare_ml_data(daily)
    print("Preprocessing complete!")
