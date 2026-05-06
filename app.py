import os
import csv
import pickle
import joblib
import numpy as np
import pandas as pd
from io import StringIO
from datetime import datetime
from functools import wraps

from flask import (Flask, render_template, request, redirect, url_for,
                   flash, jsonify, make_response)
from flask_login import (LoginManager, UserMixin, login_user, logout_user,
                         login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-prod-2024")

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "warning"

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "instance", "loan_app.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT    UNIQUE NOT NULL,
            email     TEXT    UNIQUE NOT NULL,
            password  TEXT    NOT NULL,
            is_admin  INTEGER DEFAULT 0,
            created   TEXT    DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id           INTEGER NOT NULL,
            annual_income     REAL,
            credit_score      INTEGER,
            age               INTEGER,
            existing_debts    REAL,
            loan_amount       REAL,
            loan_term         INTEGER,
            employment_type   TEXT,
            employment_years  INTEGER,
            existing_loans    INTEGER,
            interest_rate     REAL,
            education_level   TEXT,
            gender            TEXT,
            marital_status    TEXT,
            loan_purpose      TEXT,
            home_ownership    TEXT,
            result            TEXT,
            confidence        REAL,
            emi               REAL,
            created           TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()

# ── ML Model ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
_model = None
EDUCATION_MAP = {"High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "PhD": 4}

def load_model():
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        _model = joblib.load(MODEL_PATH)
        print(f"[INFO] Model loaded via joblib: {type(_model).__name__}")
    except Exception:
        try:
            with open(MODEL_PATH, "rb") as f:
                _model = pickle.load(f)
            print(f"[INFO] Model loaded via pickle: {type(_model).__name__}")
        except Exception as e:
            print(f"[ERROR] Could not load model: {e}")
            return None
    return _model

def build_feature_df(annual_income, credit_score, age, existing_debts,
                     loan_amount, loan_term, employment_type, employment_years,
                     existing_loans, interest_rate, education_level,
                     gender, marital_status, loan_purpose, home_ownership):
    """Build the 32-feature DataFrame the XGBoost pipeline expects."""
    now = datetime.now()
    dti            = float(existing_debts) / max(float(annual_income), 1)
    loan_to_income = float(loan_amount)    / max(float(annual_income), 1)
    r = (float(interest_rate) / 100) / 12
    n = int(loan_term)
    est_monthly = (float(loan_amount) * r * (1 + r)**n / ((1 + r)**n - 1)) if r > 0 else float(loan_amount) / n
    edu_enc = EDUCATION_MAP.get(education_level, 2)

    row = {
        "age":                           int(age),
        "credit_score":                  int(credit_score),
        "employment_years":              int(employment_years),
        "debt_to_income_ratio":          round(dti, 6),
        "existing_loans":                int(existing_loans),
        "loan_term_months":              n,
        "interest_rate":                 float(interest_rate),
        "app_year":                      now.year,
        "app_month":                     now.month,
        "app_quarter":                   (now.month - 1) // 3 + 1,
        "app_dow":                       now.weekday(),
        "log_annual_income":             round(np.log1p(float(annual_income)), 6),
        "log_loan_amount":               round(np.log1p(float(loan_amount)), 6),
        "education_level_enc":           edu_enc,
        "gender_Male":                   1 if gender == "Male"          else 0,
        "gender_Other":                  1 if gender == "Other"         else 0,
        "marital_status_Married":        1 if marital_status == "Married"  else 0,
        "marital_status_Single":         1 if marital_status == "Single"   else 0,
        "marital_status_Widowed":        1 if marital_status == "Widowed"  else 0,
        "employment_type_Part-time":     1 if employment_type == "Part-time"     else 0,
        "employment_type_Self-employed": 1 if employment_type == "Self-employed" else 0,
        "employment_type_Unemployed":    1 if employment_type == "Unemployed"    else 0,
        "loan_purpose_Car":              1 if loan_purpose == "Car"       else 0,
        "loan_purpose_Education":        1 if loan_purpose == "Education" else 0,
        "loan_purpose_Home":             1 if loan_purpose == "Home"      else 0,
        "loan_purpose_Medical":          1 if loan_purpose == "Medical"   else 0,
        "loan_purpose_Personal":         1 if loan_purpose == "Personal"  else 0,
        "home_ownership_Other":          1 if home_ownership == "Other"   else 0,
        "home_ownership_Own":            1 if home_ownership == "Own"     else 0,
        "home_ownership_Rent":           1 if home_ownership == "Rent"    else 0,
        "loan_to_income":                round(loan_to_income, 6),
        "est_monthly_payment":           round(est_monthly, 2),
    }
    return pd.DataFrame([row])

def run_prediction(annual_income, credit_score, age, existing_debts,
                   loan_amount, loan_term, employment_type, employment_years,
                   existing_loans, interest_rate, education_level,
                   gender, marital_status, loan_purpose, home_ownership):
    model = load_model()

    if model is None:
        # Rule-based fallback
        dti  = float(existing_debts) / max(float(annual_income), 1)
        lti  = float(loan_amount)    / max(float(annual_income), 1)
        sc   = int(credit_score)
        if sc >= 680 and dti < 0.4 and lti < 0.5 and employment_type != "Unemployed":
            return True,  round(min(0.95, 0.55 + (sc - 680) / 1000) * 100, 1)
        else:
            return False, round(max(0.05, 0.40 - max(0, 650 - sc) / 1000) * 100, 1)

    try:
        df    = build_feature_df(annual_income, credit_score, age, existing_debts,
                                 loan_amount, loan_term, employment_type, employment_years,
                                 existing_loans, interest_rate, education_level,
                                 gender, marital_status, loan_purpose, home_ownership)
        pred  = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
        conf  = round(float(np.max(proba)) * 100, 1)
        return int(pred) == 1, conf   # classes_: [0=Rejected, 1=Approved]
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return False, 50.0

def calculate_emi(principal, annual_rate_pct, months):
    r = (annual_rate_pct / 100) / 12
    if r == 0:
        return round(principal / months, 2)
    return round(principal * r * (1 + r)**months / ((1 + r)**months - 1), 2)

# ── User Model ────────────────────────────────────────────────────────────────
class User(UserMixin):
    def __init__(self, id, username, email, is_admin=False):
        self.id = id; self.username = username
        self.email = email; self.is_admin = is_admin

@login_manager.user_loader
def load_user(user_id):
    conn = get_db()
    row  = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    if row:
        return User(row["id"], row["username"], row["email"], bool(row["is_admin"]))
    return None

# ── Auth Routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("dashboard")) if current_user.is_authenticated else render_template("landing.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email",    "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm",  "")
        if not all([username, email, password]):
            flash("All fields are required.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
        else:
            conn = get_db()
            try:
                conn.execute("INSERT INTO users (username,email,password) VALUES (?,?,?)",
                             (username, email, generate_password_hash(password)))
                conn.commit()
                flash("Account created! Please log in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("Username or email already exists.", "error")
            finally:
                conn.close()
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password   = request.form.get("password", "")
        conn = get_db()
        row  = conn.execute("SELECT * FROM users WHERE username=? OR email=?",
                            (identifier, identifier)).fetchone()
        conn.close()
        if row and check_password_hash(row["password"], password):
            login_user(User(row["id"], row["username"], row["email"], bool(row["is_admin"])), remember=True)
            return redirect(request.args.get("next") or url_for("dashboard"))
        flash("Invalid credentials.", "error")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "success")
    return redirect(url_for("index"))

# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    conn     = get_db()
    total    = conn.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?", (current_user.id,)).fetchone()[0]
    approved = conn.execute("SELECT COUNT(*) FROM predictions WHERE user_id=? AND result='Approved'", (current_user.id,)).fetchone()[0]
    recent   = conn.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY created DESC LIMIT 5", (current_user.id,)).fetchall()
    conn.close()
    return render_template("dashboard.html", total=total, approved=approved,
                           approval_rate=round(approved / total * 100, 1) if total else 0,
                           recent=recent)

# ── Prediction ────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        try:
            annual_income    = float(request.form["annual_income"])
            credit_score     = int(request.form["credit_score"])
            age              = int(request.form["age"])
            existing_debts   = float(request.form["existing_debts"])
            loan_amount      = float(request.form["loan_amount"])
            loan_term        = int(request.form["loan_term"])
            employment_type  = request.form["employment_type"]
            employment_years = int(request.form.get("employment_years", 3))
            existing_loans   = int(request.form.get("existing_loans", 0))
            interest_rate    = float(request.form.get("interest_rate", 8.5))
            education_level  = request.form.get("education_level", "Bachelor")
            gender           = request.form.get("gender", "Male")
            marital_status   = request.form.get("marital_status", "Single")
            loan_purpose     = request.form.get("loan_purpose", "Personal")
            home_ownership   = request.form.get("home_ownership", "Rent")
        except (ValueError, KeyError) as e:
            flash(f"Invalid input: {e}", "error")
            return redirect(url_for("predict"))

        errors = []
        if not (300 <= credit_score <= 850): errors.append("Credit score must be 300–850.")
        if not (18 <= age <= 75):            errors.append("Age must be 18–75.")
        if annual_income <= 0:               errors.append("Annual income must be positive.")
        if loan_amount < 1000:               errors.append("Loan amount must be at least ₹1,000.")
        if not (0 < interest_rate <= 50):    errors.append("Interest rate must be between 0–50%.")
        if errors:
            for e in errors: flash(e, "error")
            return redirect(url_for("predict"))

        approved, confidence = run_prediction(
            annual_income, credit_score, age, existing_debts, loan_amount, loan_term,
            employment_type, employment_years, existing_loans, interest_rate,
            education_level, gender, marital_status, loan_purpose, home_ownership
        )
        result = "Approved" if approved else "Rejected"
        emi    = calculate_emi(loan_amount, interest_rate, loan_term) if approved else None

        conn = get_db()
        conn.execute("""INSERT INTO predictions
            (user_id,annual_income,credit_score,age,existing_debts,loan_amount,loan_term,
             employment_type,employment_years,existing_loans,interest_rate,education_level,
             gender,marital_status,loan_purpose,home_ownership,result,confidence,emi)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (current_user.id, annual_income, credit_score, age, existing_debts,
             loan_amount, loan_term, employment_type, employment_years, existing_loans,
             interest_rate, education_level, gender, marital_status, loan_purpose,
             home_ownership, result, confidence, emi))
        conn.commit()
        pred_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()
        return redirect(url_for("result", pred_id=pred_id))

    return render_template("predict.html", model_loaded=os.path.exists(MODEL_PATH))

@app.route("/result/<int:pred_id>")
@login_required
def result(pred_id):
    conn = get_db()
    pred = conn.execute("SELECT * FROM predictions WHERE id=? AND user_id=?",
                        (pred_id, current_user.id)).fetchone()
    conn.close()
    if not pred:
        flash("Prediction not found.", "error")
        return redirect(url_for("dashboard"))
    return render_template("result.html", pred=pred)

# ── History ───────────────────────────────────────────────────────────────────
@app.route("/history")
@login_required
def history():
    conn  = get_db()
    preds = conn.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY created DESC",
                         (current_user.id,)).fetchall()
    conn.close()
    return render_template("history.html", predictions=preds)

@app.route("/history/delete/<int:pred_id>", methods=["POST"])
@login_required
def delete_prediction(pred_id):
    conn = get_db()
    conn.execute("DELETE FROM predictions WHERE id=? AND user_id=?", (pred_id, current_user.id))
    conn.commit()
    conn.close()
    flash("Prediction deleted.", "success")
    return redirect(url_for("history"))

@app.route("/history/export")
@login_required
def export_csv():
    conn  = get_db()
    preds = conn.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY created DESC",
                         (current_user.id,)).fetchall()
    conn.close()
    si = StringIO()
    w  = csv.writer(si)
    w.writerow(["Date","Annual Income","Credit Score","Age","Existing Debts","Loan Amount",
                "Loan Term","Employment Type","Employment Years","Existing Loans","Interest Rate",
                "Education","Gender","Marital Status","Loan Purpose","Home Ownership",
                "Result","Confidence %","EMI"])
    for p in preds:
        w.writerow([p["created"], p["annual_income"], p["credit_score"], p["age"],
                    p["existing_debts"], p["loan_amount"], p["loan_term"],
                    p["employment_type"], p["employment_years"], p["existing_loans"],
                    p["interest_rate"], p["education_level"], p["gender"],
                    p["marital_status"], p["loan_purpose"], p["home_ownership"],
                    p["result"], p["confidence"], p["emi"] or "N/A"])
    out = make_response(si.getvalue())
    out.headers["Content-Disposition"] = "attachment; filename=loan_predictions.csv"
    out.headers["Content-type"] = "text/csv"
    return out

# ── Admin ─────────────────────────────────────────────────────────────────────
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash("Admin access required.", "error")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated

@app.route("/admin")
@login_required
@admin_required
def admin():
    conn          = get_db()
    users         = conn.execute("SELECT * FROM users ORDER BY created DESC").fetchall()
    total_preds   = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    total_approved= conn.execute("SELECT COUNT(*) FROM predictions WHERE result='Approved'").fetchone()[0]
    conn.close()
    return render_template("admin.html", users=users, total_preds=total_preds,
                           total_approved=total_approved,
                           approval_rate=round(total_approved / total_preds * 100, 1) if total_preds else 0)

@app.route("/api/credit-tier")
def credit_tier():
    score = int(request.args.get("score", 0))
    if score < 580:   tier, color = "Poor",      "#ef4444"
    elif score < 670: tier, color = "Fair",      "#f97316"
    elif score < 740: tier, color = "Good",      "#eab308"
    else:             tier, color = "Excellent", "#22c55e"
    return jsonify({"tier": tier, "color": color})

if __name__ == "__main__":
    init_db()
    load_model()
    app.run(debug=True, port=5000)
