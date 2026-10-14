import sqlite3
import sys
import threading
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, g)
from werkzeug.security import generate_password_hash, check_password_hash

BASE = Path(__file__).resolve().parent      # dashboard/
ROOT = BASE.parent                          # project root
DB_PATH = BASE / "users.db"

sys.path.append(str(ROOT / "src"))          # so we can import prediction.py
from prediction import (FEATURES, RUL_CAP, get_status, load_predictors,
                        load_test_data, evaluate)
from charts import line_chart, bar_chart

app = Flask(__name__)
app.secret_key = "put-any-long-random-text-here"   # needed for login sessions

SENSORS = [c for c in FEATURES if c.startswith("sensor_")]
DEFAULT_SENSORS = ["sensor_2", "sensor_4", "sensor_7", "sensor_11"]


# ---------- models and data: loaded once, on the first page that needs them ----------
_state = None
_state_lock = threading.Lock()


def build_state():
    test = load_test_data()
    predictors, problems = load_predictors()

    # Predict every test engine once, with every model, for the comparison and planner pages.
    last_row = test["cycle"] == test.groupby("engine_id")["cycle"].transform("max")
    last = test[last_row].set_index("engine_id")
    final, metrics = {}, []
    for name, predict in predictors.items():
        preds = predict(test)[last_row.to_numpy()]
        final[name] = dict(zip(last.index, preds))
        metrics.append({"model": name, **evaluate(last["ref_rul"], preds)})

    best = min(metrics, key=lambda m: m["MAE"]) if metrics else None
    return {
        "test": test,
        "engine_ids": sorted(int(e) for e in test["engine_id"].unique()),
        "predictors": predictors,
        "problems": problems,
        "final": final,
        "metrics": metrics,
        "best": best,
    }


def get_state():
    global _state
    with _state_lock:
        if _state is None:
            _state = build_state()
    return _state


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


@app.context_processor
def inject_user():
    return {"user": session.get("user")}


# ---------- helpers for the dashboard pages ----------
def page(template, state, **context):
    """Render a dashboard page, or the 'no models' page if nothing could be loaded."""
    if not state["predictors"]:
        return render_template("no_models.html", problems=state["problems"])
    return render_template(template, problems=state["problems"], **context)


def selection(state):
    """Engine and model chosen by the user (kept in the session between pages)."""
    ids, models = state["engine_ids"], list(state["predictors"])
    engine = request.args.get("engine", type=int, default=session.get("engine", ids[0]))
    if engine not in ids:
        engine = ids[0]
    default_model = state["best"]["model"] if state["best"] else models[0]
    model = request.args.get("model", default=session.get("model", default_model))
    if model not in models:
        model = default_model
    session["engine"], session["model"] = engine, model
    return engine, model


def engine_rows(state, engine):
    test = state["test"]
    return test[test["engine_id"] == engine]


# ---------- routes: accounts ----------
@app.route("/")
def home():
    return redirect(url_for("overview"))


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
            return redirect(url_for("overview"))
        flash("Wrong username or password.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------- routes: dashboard pages ----------
@app.route("/overview")
@login_required
def overview():
    state = get_state()
    if not state["predictors"]:
        return page("overview.html", state)
    engine, model = selection(state)
    eng = engine_rows(state, engine)

    predicted, error = None, None
    try:
        predicted = float(state["predictors"][model](eng)[-1])
    except Exception as exc:
        error = f"Prediction failed: {exc}"

    return page("overview.html", state,
                engine_ids=state["engine_ids"], models=list(state["predictors"]),
                engine=engine, model=model, cycle=int(eng["cycle"].max()),
                predicted=predicted, status=get_status(predicted) if predicted is not None else None,
                cap=RUL_CAP, best=state["best"], error=error)


@app.route("/explorer")
@login_required
def explorer():
    state = get_state()
    if not state["predictors"]:
        return page("explorer.html", state)
    engine, model = selection(state)
    eng = engine_rows(state, engine)

    curves, breakdown, errors = {}, [], []
    for name, predict in state["predictors"].items():
        try:
            curves[name] = predict(eng)
            breakdown.append({"model": name, "value": float(curves[name][-1]),
                              "selected": name == model})
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    predicted = float(curves[model][-1]) if model in curves else None
    cycles = eng["cycle"].tolist()
    rul_chart = sensor_chart = None
    if model in curves:
        rul_chart = line_chart(
            cycles,
            [{"name": f"{model} (estimated)", "ys": curves[model], "cls": "line-main"},
             {"name": "Reference RUL", "ys": eng["ref_rul"].tolist(), "cls": "line-ref"}],
            y_min=0, y_max=RUL_CAP, y_ticks=5, y_label="RUL (cycles)")
    sensor_chart = line_chart(
        cycles, [{"name": "sensor_11", "ys": eng["sensor_11"].tolist(), "cls": "line-main"}],
        y_label="Sensor 11 reading")

    return page("explorer.html", state,
                engine_ids=state["engine_ids"], models=list(state["predictors"]),
                engine=engine, model=model, cycle=int(eng["cycle"].max()),
                predicted=predicted, status=get_status(predicted) if predicted is not None else None,
                breakdown=breakdown, reference=float(eng["ref_rul"].iloc[-1]),
                timestamp=datetime.now().strftime("%d %b %Y, %H:%M:%S"),
                rul_chart=rul_chart, sensor_chart=sensor_chart,
                error="; ".join(errors) if errors else None)


@app.route("/comparison")
@login_required
def comparison():
    state = get_state()
    if not state["predictors"]:
        return page("comparison.html", state)
    metrics, best = state["metrics"], state["best"]["model"]
    mae_chart = bar_chart([(m["model"], m["MAE"]) for m in metrics], best=best)
    rmse_best = min(metrics, key=lambda m: m["RMSE"])["model"]
    rmse_chart = bar_chart([(m["model"], m["RMSE"]) for m in metrics], best=rmse_best)
    return page("comparison.html", state, metrics=metrics, best=state["best"],
                mae_chart=mae_chart, rmse_chart=rmse_chart,
                n_engines=len(state["engine_ids"]), cap=RUL_CAP)


@app.route("/sensors")
@login_required
def sensors():
    state = get_state()
    if not state["predictors"]:
        return page("sensors.html", state)
    engine, _ = selection(state)
    eng = engine_rows(state, engine)

    chosen = [s for s in request.args.getlist("sensor") if s in SENSORS] or DEFAULT_SENSORS
    charts = [(s, line_chart(eng["cycle"].tolist(),
                             [{"name": s, "ys": eng[s].tolist(), "cls": "line-main"}],
                             height=210))
              for s in chosen]
    return page("sensors.html", state, engine_ids=state["engine_ids"], engine=engine,
                all_sensors=SENSORS, chosen=chosen, charts=charts,
                cycle=int(eng["cycle"].max()))


@app.route("/planner")
@login_required
def planner():
    state = get_state()
    if not state["predictors"]:
        return page("planner.html", state)
    _, model = selection(state)
    threshold = request.args.get("threshold", type=float, default=50.0)

    rows = sorted(state["final"][model].items(), key=lambda kv: kv[1])
    table = [{"engine": int(e), "rul": float(v), "status": get_status(v), "flag": v < threshold}
             for e, v in rows]
    return page("planner.html", state, models=list(state["predictors"]), model=model,
                threshold=threshold, table=table,
                flagged=sum(r["flag"] for r in table))


if __name__ == "__main__":
    init_db()
    # use_reloader=False: the auto-reloader and TensorFlow do not get along,
    # so restart the app by hand (Ctrl+C, run again) after editing a file.
    app.run(debug=True, use_reloader=False)
