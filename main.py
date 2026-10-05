

import os
import sys

print("=" * 60)
print("⚡  ELECTRICITY CONSUMPTION OPTIMIZER")
print("    AI-Powered Household Energy Management System")
print("=" * 60)


print("\n📊 STEP 1: Generating dataset...")
os.chdir('data')
os.system('python generate_data.py')
os.chdir('..')
print("✅ Dataset ready!")

print("\n🧹 STEP 2: Preprocessing data...")
from data_preprocessing import (
    load_data, clean_data, get_daily_data,
    detect_anomalies, calculate_bill, prepare_ml_data
)

df       = load_data('data/electricity_data.csv')
df       = clean_data(df)
daily_df = get_daily_data(df)
daily_df, anomalies = detect_anomalies(daily_df)
daily_df['bill'] = daily_df['usage_kwh'].apply(calculate_bill)
daily_df.to_csv('data/daily_clean.csv', index=False)
print("✅ Preprocessing complete!")


print("\n🤖 STEP 3: Training LSTM model...")
print("   (This may take 2-5 minutes...)")
from train_model import build_model, train_model, evaluate_model

X_train, X_test, y_train, y_test, scaler = prepare_ml_data(daily_df)
model          = build_model(sequence_length=30)
model, history = train_model(model, X_train, y_train)
y_pred, y_test_actual = evaluate_model(model, X_test, y_test, scaler)
print("✅ Model trained and saved!")

print("\n🚀 STEP 4: Launching dashboard...")
print("   Opening http://localhost:8501")
print("   Press Ctrl+C to stop the dashboard")
print("=" * 60)
os.system('streamlit run app/dashboard.py')
