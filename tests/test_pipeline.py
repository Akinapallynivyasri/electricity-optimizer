"""Basic checks on the data pipeline, tariff and forecaster.  Run:  pytest"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_preprocessing as dp  # noqa: E402
from forecast import LSTMForecaster  # noqa: E402


def hourly_frame(start, periods, kw=2.0, freq="h"):
    idx = pd.date_range(start, periods=periods, freq=freq)
    return pd.DataFrame({"Global_active_power": kw}, index=idx.rename("datetime"))


# ── tariff ────────────────────────────────────────────────────
def test_monthly_slabs():
    assert dp.calculate_bill(100) == pytest.approx(346.0)
    assert dp.calculate_bill(300) == pytest.approx(346.0 + 200 * 6.58)
    assert dp.calculate_bill(400) == pytest.approx(346.0 + 200 * 6.58 + 100 * 9.56)


def test_bill_scales_with_period():
    # 3 days of usage billed over 3 days should be 3x one day's usage billed over 1 day
    assert dp.calculate_bill(30, days=3) == pytest.approx(3 * dp.calculate_bill(10, days=1), rel=1e-6)


# ── energy units ──────────────────────────────────────────────
def test_hourly_kw_becomes_kwh_without_scaling():
    daily = dp.get_daily_data(hourly_frame("2023-01-01", 72, kw=2.0))
    assert len(daily) == 3
    assert daily["usage_kwh"].tolist() == pytest.approx([48.0, 48.0, 48.0])   # 2 kW x 24 h


def test_per_minute_data_is_handled():
    daily = dp.get_daily_data(hourly_frame("2023-01-01", 2 * 1440, kw=1.2, freq="min"))
    assert daily["usage_kwh"].tolist() == pytest.approx([28.8, 28.8])          # 1.2 kW x 24 h


def test_partial_days_at_the_edges_are_dropped():
    df = hourly_frame("2023-01-01 10:00", 24 * 3 + 6)       # starts 10:00, ends 04:00 on day 4
    daily = dp.get_daily_data(df)
    assert len(daily) == 2                                    # only the two full days remain
    assert (daily["usage_kwh"] > 40).all()


def test_voltage_is_carried_into_daily_table():
    df = hourly_frame("2023-01-01", 48)
    df["voltage"] = 240.0
    assert dp.get_daily_data(df)["voltage"].tolist() == pytest.approx([240.0, 240.0])


# ── loading ───────────────────────────────────────────────────
def test_uci_format_file_loads(tmp_path):
    rows = ["Date;Time;Global_active_power;Voltage"]
    for minute in range(120):
        t = pd.Timestamp("2007-01-01") + pd.Timedelta(minutes=minute)
        value = "?" if minute == 5 else "1.5"
        rows.append(f"{t:%d/%m/%Y};{t:%H:%M:%S};{value};240.0")
    path = tmp_path / "uci.csv"
    path.write_text("\n".join(rows))

    df = dp.clean_data(dp.load_data(str(path)))
    assert df["Global_active_power"].isna().sum() == 0        # the '?' was filled
    assert df.index[0] == pd.Timestamp("2007-01-01 00:00")


# ── ML preparation ────────────────────────────────────────────
def test_scaler_uses_training_period_only():
    pytest.importorskip("sklearn")
    # 150 days -> 120 sequences -> the training period covers days 0-125.
    usage = np.r_[np.full(126, 10.0), np.full(24, 500.0)]      # a huge spike only in the test period
    daily = pd.DataFrame({"usage_kwh": usage})
    _, _, _, _, scaler = dp.prepare_ml_data(daily, sequence_length=30)
    assert scaler.data_max_[0] == pytest.approx(10.0)


def test_sequences_have_expected_shape():
    pytest.importorskip("sklearn")
    daily = pd.DataFrame({"usage_kwh": np.random.default_rng(0).uniform(5, 15, 200)})
    X_train, X_test, y_train, y_test, _ = dp.prepare_ml_data(daily, sequence_length=30)
    assert X_train.shape[1:] == (30, 1)
    assert len(X_train) + len(X_test) == 200 - 30
    assert len(X_test) == pytest.approx(0.2 * 170, abs=1)


# ── trained model ─────────────────────────────────────────────
def test_numpy_forecaster_gives_sensible_output():
    fc = LSTMForecaster.load()
    if fc is None:
        pytest.skip("models/lstm_weights.npz not found (run python train_model.py)")
    history = np.linspace(35, 45, 60)
    out = fc.forecast(history, steps=7)
    assert out.shape == (7,)
    assert np.all(out >= 0) and np.all(np.isfinite(out))
