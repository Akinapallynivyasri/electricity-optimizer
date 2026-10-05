import pandas as pd
import numpy as np

# Generate 2 years of hourly electricity data
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='H')

# Simulate realistic usage patterns
def generate_usage(hour, month):
    # Base load
    base = 0.5
    # Morning peak (6-9 AM)
    if 6 <= hour <= 9:
        base += np.random.uniform(1.0, 2.5)
    # Evening peak (6-10 PM)
    elif 18 <= hour <= 22:
        base += np.random.uniform(1.5, 3.0)
    # Night low usage
    elif 0 <= hour <= 5:
        base += np.random.uniform(0.1, 0.4)
    else:
        base += np.random.uniform(0.5, 1.2)
    # Summer months higher usage (AC)
    if month in [4, 5, 6, 7]:
        base += np.random.uniform(0.5, 1.5)
    return round(base, 3)

usage = [generate_usage(d.hour, d.month) for d in dates]

df = pd.DataFrame({
    'datetime': dates,
    'Global_active_power': usage,
    'voltage': np.random.uniform(230, 245, len(dates)),
    'sub_metering_1': np.random.uniform(0, 0.5, len(dates)),
    'sub_metering_2': np.random.uniform(0, 0.8, len(dates)),
    'sub_metering_3': np.random.uniform(0, 1.2, len(dates)),
})

df.to_csv('electricity_data.csv', index=False)
print(f"Dataset created: {len(df)} records")
print(df.head())
