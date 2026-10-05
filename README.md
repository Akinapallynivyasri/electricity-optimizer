# ⚡ Electricity Consumption Optimizer
### AI-Powered Household Energy Management System

---

## 📁 Project Structure

```
electricity_optimizer/
│
├── data/
│   ├── generate_data.py       ← Creates sample dataset
│   ├── electricity_data.csv   ← Raw dataset (auto-generated)
│   └── daily_clean.csv        ← Cleaned dataset (auto-generated)
│
├── models/
│   ├── lstm_model.h5          ← Trained LSTM model (auto-saved)
│   ├── best_model.h5          ← Best checkpoint (auto-saved)
│   ├── training_history.png   ← Training loss graph
│   └── predictions.png        ← Prediction accuracy graph
│
├── app/
│   └── dashboard.py           ← Streamlit web dashboard
│
├── data_preprocessing.py      ← Data loading & cleaning
├── train_model.py             ← LSTM model training
├── main.py                    ← Run everything at once
├── requirements.txt           ← All required libraries
└── README.md                  ← This file
```

---

## 🚀 Setup Instructions (Step by Step)

### Step 1: Install Python
- Download Python 3.10+ from https://python.org
- During install → CHECK "Add Python to PATH"
- Verify: open terminal → type `python --version`

### Step 2: Download This Project
- Save all files in a folder called `electricity_optimizer`

### Step 3: Open Terminal in Project Folder
```bash
cd electricity_optimizer
```

### Step 4: Install Libraries
```bash
pip install -r requirements.txt
```

### Step 5: Run Complete Project
```bash
python main.py
```

### OR Run Each Step Manually:
```bash
# Step A: Generate dataset
cd data && python generate_data.py && cd ..

# Step B: Preprocess data
python data_preprocessing.py

# Step C: Train LSTM model
python train_model.py

# Step D: Launch dashboard
streamlit run app/dashboard.py
```

### Step 6: Open Dashboard
- Browser opens automatically
- If not: go to http://localhost:8501

---

## 📊 Dataset
- Uses UCI Household Electric Power Consumption dataset
- OR uses auto-generated simulation data
- To use real Kaggle dataset:
  1. Download from: https://kaggle.com/datasets/uciml/electric-power-consumption-data-set
  2. Place in `data/` folder
  3. Rename to `electricity_data.csv`

---

## 🧠 Technologies Used
| Library      | Purpose                    |
|-------------|----------------------------|
| Pandas       | Data loading & cleaning    |
| NumPy        | Numerical operations       |
| Scikit-learn | Data scaling               |
| TensorFlow   | LSTM model training        |
| Matplotlib   | Training plots             |
| Plotly       | Interactive charts         |
| Streamlit    | Web dashboard              |

---

## 📈 Expected Results
- MAE: ~0.4-0.6 kWh
- RMSE: ~0.5-0.8 kWh
- Accuracy: ~92-96%
- Dashboard: Real-time charts, predictions, bill calculator

---

## 👨‍💻 Built With Python for Undergraduate Research Project
