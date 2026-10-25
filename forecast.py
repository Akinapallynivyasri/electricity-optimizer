"""Forecasting helpers used by the dashboard.

The LSTM is trained with TensorFlow (see train_model.py), but at prediction time
it is just a few matrix multiplications. This module re-implements that forward
pass in plain NumPy, so the dashboard can use the trained model without
installing TensorFlow (which is large and awkward to host).
"""
import json
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
WEIGHTS_FILE = os.path.join(MODELS_DIR, "lstm_weights.npz")
METRICS_FILE = os.path.join(MODELS_DIR, "metrics.json")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lstm_layer(x, kernel, recurrent, bias, return_sequences):
    """One Keras LSTM layer. x has shape (batch, time, features)."""
    batch, steps, _ = x.shape
    units = recurrent.shape[0]
    h = np.zeros((batch, units))
    c = np.zeros((batch, units))
    outputs = []

    for t in range(steps):
        z = x[:, t, :] @ kernel + h @ recurrent + bias
        i = _sigmoid(z[:, :units])                 # input gate
        f = _sigmoid(z[:, units:2 * units])        # forget gate
        g = np.tanh(z[:, 2 * units:3 * units])     # candidate cell state
        o = _sigmoid(z[:, 3 * units:])             # output gate
        c = f * c + i * g
        h = o * np.tanh(c)
        outputs.append(h)

    return np.stack(outputs, axis=1) if return_sequences else h


class LSTMForecaster:
    """The trained 2-layer LSTM, running on NumPy."""

    def __init__(self, arrays):
        self.w = {k: arrays[k] for k in arrays.files if k not in
                  ("data_min", "data_max", "sequence_length")}
        self.data_min = float(arrays["data_min"])
        self.data_max = float(arrays["data_max"])
        self.sequence_length = int(arrays["sequence_length"])

    @classmethod
    def load(cls, path=WEIGHTS_FILE):
        """Return the forecaster, or None if no trained weights are available."""
        if not os.path.exists(path):
            return None
        try:
            return cls(np.load(path))
        except Exception:
            return None

    def predict_scaled(self, x):
        """x: (batch, sequence_length, 1) in the 0-1 scaled range."""
        w = self.w
        h = _lstm_layer(x, w["lstm1_kernel"], w["lstm1_recurrent"], w["lstm1_bias"], True)
        h = _lstm_layer(h, w["lstm2_kernel"], w["lstm2_recurrent"], w["lstm2_bias"], False)
        h = np.maximum(h @ w["dense1_kernel"] + w["dense1_bias"], 0)      # ReLU
        return h @ w["dense2_kernel"] + w["dense2_bias"]

    def _scale(self, kwh):
        return (np.asarray(kwh, dtype=float) - self.data_min) / (self.data_max - self.data_min)

    def _unscale(self, scaled):
        return scaled * (self.data_max - self.data_min) + self.data_min

    def forecast(self, history_kwh, steps=30):
        """Predict the next `steps` days, feeding each prediction back in.

        The model was trained to predict one day ahead, so a 30-day forecast
        built this way gets less certain the further out it goes.
        """
        history = list(np.asarray(history_kwh, dtype=float))
        if len(history) < self.sequence_length:
            raise ValueError("Not enough history for the LSTM window.")

        predictions = []
        for _ in range(steps):
            window = self._scale(history[-self.sequence_length:]).reshape(1, -1, 1)
            next_kwh = float(self._unscale(self.predict_scaled(window))[0, 0])
            next_kwh = max(next_kwh, 0.0)           # usage can't be negative
            predictions.append(next_kwh)
            history.append(next_kwh)
        return np.array(predictions)


def trend_estimate(usage_kwh):
    """Fallback used when no trained model is available: recent average + trend.

    Returns the estimated total for the next 30 days (kWh).
    """
    usage = np.asarray(usage_kwh, dtype=float)
    last30 = usage[-30:].mean()
    prev30 = usage[-60:-30].mean() if len(usage) >= 60 else last30
    trend = (last30 - prev30) / prev30 if prev30 else 0.0
    return float(last30 * 30 * (1 + trend * 0.5))


def load_metrics(path=METRICS_FILE):
    """Test-set results saved by train_model.py (or None)."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
