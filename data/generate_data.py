"""Creates a simulated hourly household electricity dataset (2 years).

This is NOT real meter data. It is a seeded generator with realistic daily
and seasonal patterns, useful for demos and for testing the pipeline.
To use real readings, replace data/electricity_data.csv (see README).

Run:  python data/generate_data.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_usage(hour, month):
    """Power (kW) for one hour, based on time of day and season."""
    base = 0.5                                    # always-on load
    if 6 <= hour <= 9:                            # morning peak
        base += np.random.uniform(1.0, 2.5)
    elif 18 <= hour <= 22:                        # evening peak
        base += np.random.uniform(1.5, 3.0)
    elif 0 <= hour <= 5:                          # night, low usage
        base += np.random.uniform(0.1, 0.4)
    else:                                         # daytime
        base += np.random.uniform(0.5, 1.2)
    if month in [4, 5, 6, 7]:                     # summer months (AC)
        base += np.random.uniform(0.5, 1.5)
    return round(base, 3)


def main():
    np.random.seed(42)

    # Ends at 23:00 so the final day has all 24 hours (a partial last day
    # would look like a day of almost-zero usage).
    dates = pd.date_range(start="2022-01-01", end="2023-12-31 23:00", freq="h")
    usage = [generate_usage(d.hour, d.month) for d in dates]

    df = pd.DataFrame({
        "datetime": dates,
        "Global_active_power": usage,
        "voltage": np.random.uniform(230, 245, len(dates)),
        "sub_metering_1": np.random.uniform(0, 0.5, len(dates)),
        "sub_metering_2": np.random.uniform(0, 0.8, len(dates)),
        "sub_metering_3": np.random.uniform(0, 1.2, len(dates)),
    })

    df.to_csv(os.path.join(HERE, "electricity_data.csv"), index=False)

    # The dashboard shows this note so nobody mistakes demo data for real data.
    with open(os.path.join(HERE, "DATA_NOTE.txt"), "w", encoding="utf-8") as f:
        f.write("Simulated household data (seeded generator, 2022-2023). "
                "Not real meter readings.")

    print(f"Dataset created: {len(df)} hourly records")
    print(df.head())


if __name__ == "__main__":
    main()
