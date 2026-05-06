# LoanSight — AI Loan Eligibility Predictor

A full-stack Flask web application for predicting loan eligibility using your pre-trained ML model.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up your model

**Option A — Use your real model.pkl:**
Copy your `model.pkl` into the `loan_app/` folder (same directory as `app.py`).

**Option B — Generate a demo model:**
```bash
python generate_model.py
```
This creates a compatible RandomForest model. Replace it with your real model anytime.

### 3. Run the app
```bash
python app.py
```
Visit: http://localhost:5000

---

## Features

| Feature | Status |
|---------|--------|
| User Registration & Login | ✅ |
| Protected Routes | ✅ |
| Loan Prediction Form (7 inputs) | ✅ |
| Credit Score Color Indicator | ✅ |
| Result Page with Confidence % | ✅ |
| EMI Calculation (8.5% p.a.) | ✅ |
| Prediction History Table | ✅ |
| Delete Old Predictions | ✅ |
| Export History as CSV | ✅ |
| Dashboard with Stats | ✅ |
| Admin Panel | ✅ |
| Demo Fallback (no model needed) | ✅ |

---

## Model Requirements

The app expects `model.pkl` trained on **9 features** in this order:

```python
[
    'annual_income',       # float
    'credit_score',        # int  (300–850)
    'age',                 # int  (18–75)
    'existing_debts',      # float
    'loan_amount',         # float
    'loan_term',           # int  (12/24/36/60/120)
    'emp_employed',        # 0 or 1
    'emp_self_employed',   # 0 or 1
    'emp_unemployed',      # 0 or 1
]
```

The model should have `predict_proba()` for confidence scores (e.g. RandomForest, GradientBoosting, LogisticRegression). Binary classification: **1 = Approved, 0 = Rejected**.

---

## Making Yourself Admin

After registering, run:
```bash
sqlite3 instance/loan_app.db "UPDATE users SET is_admin=1 WHERE username='your_username';"
```

---

## Project Structure

```
loan_app/
├── app.py               # Main Flask application
├── generate_model.py    # Demo model generator
├── requirements.txt
├── model.pkl            # ← Your ML model goes here
├── instance/
│   └── loan_app.db      # SQLite database (auto-created)
└── templates/
    ├── base.html
    ├── landing.html
    ├── login.html
    ├── register.html
    ├── dashboard.html
    ├── predict.html
    ├── result.html
    ├── history.html
    └── admin.html
```

---

## Configuration

Set environment variables before running in production:

```bash
export SECRET_KEY="your-super-secret-key-here"
```
