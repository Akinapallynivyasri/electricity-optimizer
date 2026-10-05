# ============================================================
# FILE: data_preprocessing.py
# PURPOSE: Load, clean and prepare electricity data for ML
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# ── 1. LOAD DATA ─────────────────────────────────────────────
def load_data(filepath='data/electricity_data.csv'):
    """Load electricity dataset from CSV file"""
    print("📂 Loading dataset...")
    df = pd.read_csv(filepath)

    # Parse datetime column
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    print(f"✅ Loaded {len(df)} records")
    print(f"📅 Date range: {df.index.min()} → {df.index.max()}")
    return df


# ── 2. CLEAN DATA ─────────────────────────────────────────────
def clean_data(df):
    """Remove missing values and fix data types"""
    print("\n🧹 Cleaning data...")

    # Replace '?' with NaN (common in UCI dataset)
    df.replace('?', np.nan, inplace=True)

    # Convert to numeric
    df['Global_active_power'] = pd.to_numeric(
        df['Global_active_power'], errors='coerce'
    )

    # Count missing values
    missing = df['Global_active_power'].isna().sum()
    print(f"⚠️  Missing values found: {missing}")

    # Fill missing with forward fill
    df['Global_active_power'].fillna(method='ffill', inplace=True)
    print("✅ Missing values filled")

    return df


# ── 3. DAILY AGGREGATION ──────────────────────────────────────
def get_daily_data(df):
    """Convert hourly data to daily total usage in kWh"""
    print("\n📊 Aggregating to daily usage...")

    # Sum hourly kW values → daily kWh
    daily = df['Global_active_power'].resample('D').sum() / 1000

    # Add useful time features
    daily_df = daily.reset_index()
    daily_df.columns = ['date', 'usage_kwh']
    daily_df['day_of_week'] = daily_df['date'].dt.dayofweek
    daily_df['month'] = daily_df['date'].dt.month
    daily_df['is_weekend'] = daily_df['day_of_week'].isin([5, 6]).astype(int)

    print(f"✅ Daily records: {len(daily_df)}")
    print(f"📈 Avg daily usage: {daily_df['usage_kwh'].mean():.2f} kWh")
    return daily_df


# ── 4. CALCULATE BILL ─────────────────────────────────────────
def calculate_bill(units):
    """
    Calculate electricity bill using Maharashtra tariff slabs
    Slab 1: 0-100 units  → ₹3.46/unit
    Slab 2: 101-300 units → ₹6.58/unit
    Slab 3: 300+ units   → ₹9.56/unit
    """
    if units <= 100:
        return round(units * 3.46, 2)
    elif units <= 300:
        return round((100 * 3.46) + ((units - 100) * 6.58), 2)
    else:
        return round((100 * 3.46) + (200 * 6.58) + ((units - 300) * 9.56), 2)


# ── 5. PREPARE ML DATA ────────────────────────────────────────
def prepare_ml_data(daily_df, sequence_length=30):
    """
    Prepare sequences for LSTM model
    Uses past 30 days to predict next day
    """
    print("\n🤖 Preparing ML sequences...")

    # Get usage values
    data = daily_df['usage_kwh'].values.reshape(-1, 1)

    # Normalize between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Create input sequences
    X, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i - sequence_length:i, 0])
        y.append(data_scaled[i, 0])

    X = np.array(X)
    y = np.array(y)

    # Reshape for LSTM [samples, timesteps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # 80% train / 20% test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"✅ Training samples: {len(X_train)}")
    print(f"✅ Testing samples:  {len(X_test)}")

    return X_train, X_test, y_train, y_test, scaler


# ── 6. DETECT ANOMALIES ───────────────────────────────────────
def detect_anomalies(daily_df):
    """Flag days with unusually high electricity usage"""
    mean = daily_df['usage_kwh'].mean()
    std  = daily_df['usage_kwh'].std()
    threshold = mean + 2 * std

    daily_df['is_anomaly'] = daily_df['usage_kwh'] > threshold
    anomalies = daily_df[daily_df['is_anomaly']]

    print(f"\n🚨 Anomaly threshold: {threshold:.2f} kWh")
    print(f"⚠️  Anomaly days found: {len(anomalies)}")
    return daily_df, anomalies


# ── MAIN RUNNER ───────────────────────────────────────────────
if __name__ == '__main__':
    df       = load_data()
    df       = clean_data(df)
    daily_df = get_daily_data(df)
    daily_df, anomalies = detect_anomalies(daily_df)

    # Add monthly bill column
    daily_df['bill'] = daily_df['usage_kwh'].apply(calculate_bill)

    # Save cleaned data
    daily_df.to_csv('data/daily_clean.csv', index=False)
    print("\n💾 Saved: data/daily_clean.csv")

    X_train, X_test, y_train, y_test, scaler = prepare_ml_data(daily_df)
    print("\n✅ Preprocessing complete!")
