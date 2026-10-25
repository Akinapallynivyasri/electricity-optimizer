"""Build, train and save the LSTM that predicts the next day's electricity use.

Run:  python train_model.py        (needs requirements-train.txt)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")                      # save plots to files, don't open windows
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed

from data_preprocessing import build_daily_dataset, prepare_ml_data
from forecast import LSTMForecaster

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SEQUENCE_LENGTH = 30        # days of history the model looks at
SEED = 42                   # makes training repeatable


# ── 1. BUILD LSTM MODEL ───────────────────────────────────────
def build_model(sequence_length=SEQUENCE_LENGTH):
    """Two stacked LSTM layers followed by a small dense head."""
    print("Building LSTM model ...")

    model = Sequential([
        Input(shape=(sequence_length, 1)),
        LSTM(units=64, return_sequences=True),      # learns basic patterns
        Dropout(0.2),                               # helps prevent overfitting
        LSTM(units=32, return_sequences=False),     # learns longer patterns
        Dropout(0.2),
        Dense(units=16, activation="relu"),
        Dense(units=1),                             # next day's usage
    ])

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    model.summary()
    return model


# ── 2. TRAIN MODEL ────────────────────────────────────────────
def train_model(model, X_train, y_train):
    """Train with early stopping and keep the weights from the best epoch."""
    print("Training model ...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    # Saving only the weights works on every TensorFlow/Keras version.
    best_weights = os.path.join(MODELS_DIR, "best.weights.h5")
    checkpoint = ModelCheckpoint(
        best_weights, monitor="val_loss", save_best_only=True,
        save_weights_only=True, verbose=0,
    )

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,       # the last 10% of the training period, in time order
        callbacks=[early_stop, checkpoint],
        verbose=2,
    )
    model.load_weights(best_weights)          # go back to the best epoch
    print("Training complete!")
    return model, history


# ── 3. EVALUATE MODEL ─────────────────────────────────────────
def compute_metrics(actual, predicted):
    """MAE / RMSE in kWh and MAPE as a percentage."""
    actual, predicted = np.ravel(actual), np.ravel(predicted)
    nonzero = actual != 0                    # avoid dividing by zero in MAPE
    mape = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100
    return {
        "mae_kwh": float(mean_absolute_error(actual, predicted)),
        "rmse_kwh": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape_percent": float(mape),
        "accuracy_percent": float(100 - mape),
        "test_days": int(len(actual)),
    }


def evaluate_model(model, X_test, y_test, scaler):
    """Score the model on the held-out test period (original kWh scale)."""
    print("Evaluating model ...")
    y_pred_actual = scaler.inverse_transform(model.predict(X_test, verbose=0))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    m = compute_metrics(y_test_actual, y_pred_actual)
    print("Model performance (1 day ahead, test period):")
    print(f"   MAE  : {m['mae_kwh']:.4f} kWh")
    print(f"   RMSE : {m['rmse_kwh']:.4f} kWh")
    print(f"   MAPE : {m['mape_percent']:.2f}%")
    print(f"   Accuracy (100 - MAPE): {m['accuracy_percent']:.2f}%")
    return y_pred_actual, y_test_actual


def baseline_metrics(X_test, y_train, y_test, scaler):
    """Two trivial forecasts the LSTM should beat to be worth using."""
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    yesterday = scaler.inverse_transform(X_test[:, -1, :]).ravel()          # persistence
    train_mean = np.full_like(actual, scaler.inverse_transform(y_train.reshape(-1, 1)).mean())
    return {
        "persistence": compute_metrics(actual, yesterday),
        "train_mean": compute_metrics(actual, train_mean),
    }


# ── 4. PLOTS ──────────────────────────────────────────────────
def plot_training(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train loss", color="blue")
    plt.plot(history.history["val_loss"], label="Validation loss", color="orange")
    plt.title("Model training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"], label="Train MAE", color="green")
    plt.plot(history.history["val_mae"], label="Validation MAE", color="red")
    plt.title("MAE over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (scaled)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "training_history.png"))
    plt.close()
    print("Saved models/training_history.png")


def plot_predictions(y_test_actual, y_pred_actual):
    plt.figure(figsize=(14, 5))
    plt.plot(y_test_actual[:60], label="Actual", color="blue", linewidth=2)
    plt.plot(y_pred_actual[:60], label="Predicted", color="orange", linewidth=2, linestyle="--")
    plt.title("Actual vs predicted daily usage (first 60 test days)")
    plt.xlabel("Day")
    plt.ylabel("Usage (kWh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "predictions.png"))
    plt.close()
    print("Saved models/predictions.png")


# ── 5. EXPORT FOR THE DASHBOARD ───────────────────────────────
def export_numpy_weights(model, scaler, path=None):
    """Save the trained weights so forecast.py can run the model without TensorFlow."""
    path = path or os.path.join(MODELS_DIR, "lstm_weights.npz")
    w = model.get_weights()
    if len(w) != 10:
        raise RuntimeError("Unexpected model layout; update export_numpy_weights().")

    names = ["lstm1_kernel", "lstm1_recurrent", "lstm1_bias",
             "lstm2_kernel", "lstm2_recurrent", "lstm2_bias",
             "dense1_kernel", "dense1_bias", "dense2_kernel", "dense2_bias"]
    np.savez(path, **dict(zip(names, w)),
             data_min=scaler.data_min_[0], data_max=scaler.data_max_[0],
             sequence_length=model.input_shape[1])
    print(f"Saved {os.path.relpath(path, BASE_DIR)}")


# ── MAIN RUNNER ───────────────────────────────────────────────
def run_training(seed=SEED):
    set_random_seed(seed)
    os.makedirs(MODELS_DIR, exist_ok=True)

    _, daily_df = build_daily_dataset()
    X_train, X_test, y_train, y_test, scaler = prepare_ml_data(daily_df, SEQUENCE_LENGTH)

    model = build_model(SEQUENCE_LENGTH)
    model, history = train_model(model, X_train, y_train)

    y_pred_actual, y_test_actual = evaluate_model(model, X_test, y_test, scaler)
    metrics = compute_metrics(y_test_actual, y_pred_actual)
    metrics["baselines"] = baseline_metrics(X_test, y_train, y_test, scaler)
    metrics["sequence_length"] = SEQUENCE_LENGTH
    metrics["seed"] = seed
    for name, b in metrics["baselines"].items():
        print(f"Baseline {name:11s}: MAE {b['mae_kwh']:.4f} kWh, MAPE {b['mape_percent']:.2f}%")

    plot_training(history)
    plot_predictions(y_test_actual, y_pred_actual)

    model.save(os.path.join(MODELS_DIR, "lstm_model.keras"))
    export_numpy_weights(model, scaler)

    # Double-check that the NumPy copy of the model matches TensorFlow.
    fast = LSTMForecaster.load()
    diff = np.abs(model.predict(X_test, verbose=0) - fast.predict_scaled(X_test)).max()
    print(f"NumPy vs TensorFlow max difference: {diff:.2e}")
    metrics["numpy_max_difference"] = float(diff)

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved models/metrics.json. All done - run the dashboard next:")
    print("   streamlit run app/dashboard.py")


if __name__ == "__main__":
    run_training()
