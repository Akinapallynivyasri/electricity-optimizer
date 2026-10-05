# ============================================================
# FILE: train_model.py
# PURPOSE: Build, train and save the LSTM prediction model
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

from data_preprocessing import (
    load_data, clean_data, get_daily_data,
    prepare_ml_data, calculate_bill
)


# ── 1. BUILD LSTM MODEL ───────────────────────────────────────
def build_model(sequence_length=30):
    """Build LSTM neural network architecture"""
    print("🏗️  Building LSTM model...")

    model = Sequential([
        # First LSTM layer — learns basic patterns
        LSTM(
            units=64,
            return_sequences=True,
            input_shape=(sequence_length, 1)
        ),
        Dropout(0.2),  # Prevent overfitting

        # Second LSTM layer — learns complex patterns
        LSTM(
            units=32,
            return_sequences=False
        ),
        Dropout(0.2),

        # Dense layers — final prediction
        Dense(units=16, activation='relu'),
        Dense(units=1)  # Output: next day usage
    ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )

    model.summary()
    return model


# ── 2. TRAIN MODEL ────────────────────────────────────────────
def train_model(model, X_train, y_train):
    """Train LSTM model with early stopping"""
    print("\n🚀 Training model...")

    # Stop early if no improvement
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Save best model automatically
    os.makedirs('models', exist_ok=True)
    checkpoint = ModelCheckpoint(
        'models/best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    print("✅ Training complete!")
    return model, history


# ── 3. EVALUATE MODEL ─────────────────────────────────────────
def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model accuracy on test data"""
    print("\n📊 Evaluating model...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Convert back to original scale (kWh)
    y_pred_actual = scaler.inverse_transform(y_pred)
    y_test_actual = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    )

    # Calculate metrics
    mae  = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mape = np.mean(
        np.abs((y_test_actual - y_pred_actual) / y_test_actual)
    ) * 100

    print(f"\n📈 Model Performance:")
    print(f"   MAE  (Mean Absolute Error):  {mae:.4f} kWh")
    print(f"   RMSE (Root Mean Sq Error):   {rmse:.4f} kWh")
    print(f"   MAPE (Mean Abs % Error):     {mape:.2f}%")
    print(f"   Accuracy:                    {100 - mape:.2f}%")

    return y_pred_actual, y_test_actual


# ── 4. PLOT TRAINING HISTORY ──────────────────────────────────
def plot_training(history):
    """Plot training vs validation loss"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
    plt.title('Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE', color='green')
    plt.plot(history.history['val_mae'], label='Val MAE', color='red')
    plt.title('Model MAE Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (kWh)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.show()
    print("💾 Saved: models/training_history.png")


# ── 5. PLOT PREDICTIONS ───────────────────────────────────────
def plot_predictions(y_test_actual, y_pred_actual):
    """Plot actual vs predicted electricity usage"""
    plt.figure(figsize=(14, 5))
    plt.plot(y_test_actual[:60], label='Actual Usage',
             color='blue', linewidth=2)
    plt.plot(y_pred_actual[:60], label='Predicted Usage',
             color='orange', linewidth=2, linestyle='--')
    plt.title('Actual vs Predicted Electricity Usage (First 60 Days)')
    plt.xlabel('Days')
    plt.ylabel('Usage (kWh)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('models/predictions.png')
    plt.show()
    print("💾 Saved: models/predictions.png")


# ── MAIN RUNNER ───────────────────────────────────────────────
if __name__ == '__main__':
    # Load and prepare data
    df       = load_data()
    df       = clean_data(df)
    daily_df = get_daily_data(df)

    X_train, X_test, y_train, y_test, scaler = prepare_ml_data(daily_df)

    # Build and train model
    model          = build_model(sequence_length=30)
    model, history = train_model(model, X_train, y_train)

    # Evaluate
    y_pred_actual, y_test_actual = evaluate_model(
        model, X_test, y_test, scaler
    )

    # Save plots
    plot_training(history)
    plot_predictions(y_test_actual, y_pred_actual)

    # Save final model
    model.save('models/lstm_model.h5')
    print("\n💾 Final model saved: models/lstm_model.h5")
    print("✅ All done! Run dashboard.py next.")
