"""Run the whole project: data -> preprocessing -> training -> dashboard.

    python main.py                  everything
    python main.py --no-dashboard   stop after training
    python main.py --regenerate     overwrite data/electricity_data.csv with fresh simulated data
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(ROOT, "data", "electricity_data.csv")


def main():
    parser = argparse.ArgumentParser(description="Electricity Consumption Optimizer")
    parser.add_argument("--regenerate", action="store_true",
                        help="recreate the simulated dataset even if a data file exists")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="don't launch the Streamlit dashboard at the end")
    args = parser.parse_args()

    print("=" * 60)
    print("ELECTRICITY CONSUMPTION OPTIMIZER")
    print("=" * 60)

    # Step 1: data. An existing file is never overwritten unless asked, so your
    # own meter data is safe.
    print("\nSTEP 1: Dataset")
    if args.regenerate or not os.path.exists(RAW_FILE):
        subprocess.run([sys.executable, os.path.join(ROOT, "data", "generate_data.py")], check=True)
    else:
        print("Using existing data/electricity_data.csv")

    # Step 2: preprocessing
    print("\nSTEP 2: Preprocessing")
    from data_preprocessing import run_preprocessing
    run_preprocessing()

    # Step 3: training
    print("\nSTEP 3: Training the LSTM (a few minutes on a normal laptop)")
    try:
        from train_model import run_training
    except ImportError as err:
        sys.exit(f"{err}\nTraining needs TensorFlow: pip install -r requirements-train.txt")
    run_training()

    # Step 4: dashboard
    if args.no_dashboard:
        print("\nDone. Start the dashboard with: streamlit run app/dashboard.py")
        return
    print("\nSTEP 4: Launching the dashboard at http://localhost:8501 (Ctrl+C to stop)")
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    os.path.join(ROOT, "app", "dashboard.py")], cwd=ROOT)


if __name__ == "__main__":
    main()
