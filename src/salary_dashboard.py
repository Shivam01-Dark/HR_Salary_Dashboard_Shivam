# salary_dashboard.py
# Simple student-style analysis for TCS iON RIO-125 project
# Author: Shivam (Intern)
# Date: 2025-11- 20

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Load data ----------
# file path inside repo
DATA_PATH = "data/HRDataset_v14.csv"

print("Loading dataset:", DATA_PATH)
data = pd.read_csv(DATA_PATH)

print("\nColumns found:", list(data.columns)[:20])   # show first 20 cols

# ---------- Basic cleaning ----------
# Trim string columns
for c in data.select_dtypes(include=['object']).columns:
    data[c] = data[c].astype(str).str.strip()

# Try to find Salary column (common names)
possible_salary_cols = [c for c in data.columns if 'salary' in c.lower() or 'income' in c.lower()]
salary_col = possible_salary_cols[0] if possible_salary_cols else None

if salary_col:
    print("Detected salary column:", salary_col)
    # make salary numeric
    data[salary_col] = pd.to_numeric(data[salary_col].astype(str).str.replace(',','').str.replace('₹','').str.replace('Rs.',''), errors='coerce')
else:
    print("No obvious salary column found. Will use last numeric column as target if possible.")

# Save a cleaned sample (useful for GitHub)
data_clean_path = "data/cleaned_HRDataset_v14.csv"
data.to_csv(data_clean_path, index=False)
print("Saved cleaned CSV:", data_clean_path)

# ---------- Quick charts (student style) ----------
# 1) Salary distribution (if salary exists)
if salary_col and data[salary_col].notna().sum() > 0:
    plt.figure(figsize=(6,4))
    data[salary_col].hist(bins=30)
    plt.title("Salary distribution")
    plt.xlabel("Salary")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("visualizations/salary_distribution.png")
    print("Saved plot: visualizations/salary_distribution.png")
else:
    # simple numeric column histogram fallback
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        plt.figure(figsize=(6,4))
        data[numeric_cols[0]].hist(bins=20)
        plt.title(f"Distribution of {numeric_cols[0]}")
        plt.tight_layout()
        plt.savefig("visualizations/distribution_fallback.png")
        print("Saved plot: visualizations/distribution_fallback.png")

# 2) Average salary by Position (if exists)
if 'Position' in data.columns and salary_col:
    pos = data.groupby('Position')[salary_col].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(8,4))
    pos.plot(kind='bar')
    plt.title("Top 10 positions by avg salary")
    plt.tight_layout()
    plt.savefig("visualizations/avg_salary_by_position.png")
    print("Saved plot: visualizations/avg_salary_by_position.png")

# ---------- Simple Linear Regression (student-level) ----------
# We will try to predict salary. If salary detected, use simple numeric features.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Prepare X and y
numeric = data.select_dtypes(include=[np.number]).copy()

# If salary column present, use it as y. Otherwise use last numeric column.
if salary_col and salary_col in numeric.columns:
    y = numeric[salary_col].dropna()
    # choose features: all numeric except target
    X = numeric.drop(columns=[salary_col]).loc[y.index]
else:
    # fallback: take last numeric col as y
    if numeric.shape[1] < 2:
        print("Not enough numeric columns to train a model. Skipping regression.")
        X = None
        y = None
    else:
        y = numeric.iloc[:, -1]
        X = numeric.iloc[:, :-1]

if X is not None and y is not None and len(y) > 10:
    # Drop rows with missing values in X or y
    df_xy = pd.concat([X, y], axis=1).dropna()
    X_clean = df_xy.iloc[:, :-1].values
    y_clean = df_xy.iloc[:, -1].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Eval
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("\n--- Linear Regression Results ---")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")

    # Save a simple scatter plot actual vs predicted
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted (Regression)")
    plt.tight_layout()
    plt.savefig("visualizations/actual_vs_predicted.png")
    print("Saved plot: visualizations/actual_vs_predicted.png")
else:
    print("Regression skipped: insufficient numeric data or target.")
