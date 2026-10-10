import sqlite3
import sys
from functools import wraps
from pathlib import Path

import pandas as pd
from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, g)
from werkzeug.security import generate_password_hash, check_password_hash

BASE = Path(__file__).resolve().parent      # dashboard/
ROOT = BASE.parent                          # project root
DB_PATH = BASE / "users.db"

sys.path.append(str(ROOT / "src"))          # so we can import prediction.py
from prediction import load_model, predict_rul, get_status, RUL_CAP

app = Flask(__name__)
app.secret_key = "put-any-long-random-text-here"   # needed for login sessions

# ---------- load everything ONCE when the app starts ----------
model = load_model()
test_df = pd.read_csv(ROOT / "data" / "processed" / "test_FD001_processed.csv")
true_rul = pd.read_csv(ROOT / "data" / "raw" / "RUL_FD001.txt",
                       header=None, sep=r"\s+")[0].clip(upper=RUL_CAP)
metrics = pd.read_csv(ROOT / "results" / "metrics" / "rf_metrics.csv").iloc[0]
engine_ids = sorted(int(e) for e in test_df["engine_id"].unique())


# ---------- database ----------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )""")


# ---------- login protection ----------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


# ---------- routes ----------
@app.route("/")
def home():
    return redirect(url_for("dashboard"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        if not username or not email or len(password) < 6:
            flash("Fill all fields. Password needs at least 6 characters.")
        else:
            try:
                get_db().execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (username, email, generate_password_hash(password)))
                get_db().commit()
                flash("Account created. Please log in.")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("That username or email is already registered.")
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = get_db().execute(
            "SELECT * FROM users WHERE username = ?",
            (request.form["username"].strip(),)).fetchone()

        if user and check_password_hash(user["password_hash"], request.form["password"]):
            session["user"] = user["username"]
            return redirect(url_for("dashboard"))
        flash("Wrong username or password.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    engine_id = request.args.get("engine", default=engine_ids[0], type=int)
    if engine_id not in engine_ids:
        engine_id = engine_ids[0]

    # latest recorded cycle of the chosen engine
    last_row = test_df[test_df["engine_id"] == engine_id].tail(1)
    predicted = float(predict_rul(model, last_row)[0])
    actual = float(true_rul.iloc[engine_id - 1])    # NASA lists engines in order 1..100

    return render_template(
        "dashboard.html",
        user=session["user"],
        engine_ids=engine_ids,
        engine_id=engine_id,
        cycle=int(last_row["cycle"].iloc[0]),
        predicted=round(predicted),
        actual=round(actual),
        status=get_status(predicted),
        mae=round(float(metrics["MAE"]), 1),
        rmse=round(float(metrics["RMSE"]), 1),
        r2=round(float(metrics["R2"]), 2),
    )


if __name__ == "__main__":
    init_db()
    app.run(debug=True)