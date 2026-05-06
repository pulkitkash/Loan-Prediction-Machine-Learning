"""
generate_model.py
─────────────────
Run this script ONCE to create a compatible demo model.pkl
if you don't yet have your real model.pkl.

Usage:
    python generate_model.py

This generates a RandomForestClassifier trained on synthetic data
that matches the exact feature format expected by app.py.
Replace the resulting model.pkl with your real trained model when ready.
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("Generating synthetic training data...")

np.random.seed(42)
n = 5000

annual_income  = np.random.uniform(10000, 1000000, n)
credit_score   = np.random.randint(300, 851, n)
age            = np.random.randint(18, 76, n)
existing_debts = np.random.uniform(0, 300000, n)
loan_amount    = np.random.uniform(1000, 500000, n)
loan_term      = np.random.choice([12, 24, 36, 60, 120], n)

emp_choice     = np.random.choice([0, 1, 2], n, p=[0.65, 0.25, 0.10])
emp_employed      = (emp_choice == 0).astype(int)
emp_self_employed = (emp_choice == 1).astype(int)
emp_unemployed    = (emp_choice == 2).astype(int)

# Feature matrix (9 features — matches app.py feature_order exactly)
X = np.column_stack([
    annual_income,
    credit_score,
    age,
    existing_debts,
    loan_amount,
    loan_term,
    emp_employed,
    emp_self_employed,
    emp_unemployed,
])

# Synthetic label: 1 = Approved, 0 = Rejected
# Approved if good credit, low debt-to-income, employed, sufficient income
dti = (existing_debts + loan_amount) / np.maximum(annual_income, 1)
approved = (
    (credit_score >= 650) &
    (dti < 0.45) &
    (emp_unemployed == 0) &
    (annual_income > 25000)
).astype(int)  # 1=Approved, 0=Rejected

# Add some noise
noise_mask = np.random.rand(n) < 0.05
approved[noise_mask] = 1 - approved[noise_mask]

X_train, X_test, y_train, y_test = train_test_split(X, approved, test_size=0.2, random_state=42)

print("Training RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"Test accuracy: {acc:.3f}")

joblib.dump(model, "model.pkl")
print("✅  model.pkl saved. Place it in your loan_app/ folder.")
