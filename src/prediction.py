from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

RUL_CAP = 125
WINDOW_SIZE = 30          # must match the window used in 04_lstm.ipynb

FEATURES = ["op_setting_1", "op_setting_2",
            "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
            "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
            "sensor_17", "sensor_20", "sensor_21"]

# Models that are saved as a single joblib file. Display name -> file in /models
TABULAR_MODELS = {
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

# The LSTM notebook renames op_setting_N -> setting_N before training
LSTM_RENAME = {"op_setting_1": "setting_1",
               "op_setting_2": "setting_2",
               "op_setting_3": "setting_3"}


def load_model(name="random_forest"):
    return joblib.load(MODELS_DIR / f"{name}.pkl")


def predict_rul(model, df):
    """Predict RUL for every row of df (Random Forest / XGBoost).

    Each model remembers the columns it was trained on, so we use those.
    (XGBoost was trained with 'cycle' and 'op_setting_3'; Random Forest was not.)
    """
    cols = list(getattr(model, "feature_names_in_", FEATURES))
    return np.clip(model.predict(df[cols]), 0, RUL_CAP)


def get_status(rul):
    """Demo thresholds defined by our project, not real aviation limits."""
    if rul > 100:
        return "Healthy"
    elif rul >= 50:
        return "Monitor"
    elif rul >= 20:
        return "Warning"
    else:
        return "Critical"


# ---------------------------------------------------------------- LSTM
def load_lstm():
    """Load the LSTM, its scaler and its feature list (needs TensorFlow)."""
    from tensorflow.keras.models import load_model as keras_load_model
    return {
        "model": keras_load_model(MODELS_DIR / "lstm.keras"),
        "scaler": joblib.load(MODELS_DIR / "lstm_scaler.pkl"),
        "features": joblib.load(MODELS_DIR / "lstm_features.pkl"),
    }


def lstm_predict(bundle, df, window=WINDOW_SIZE):
    """Predict RUL for every row of df with the LSTM.

    df holds one or more engines (processed test format). For each cycle the
    model sees the last `window` cycles of that engine; the start of an engine
    is padded by repeating its first row, exactly as in 04_lstm.ipynb.
    The result is in the same row order as df.
    """
    d = df.reset_index(drop=True).rename(columns=LSTM_RENAME)
    d["_row"] = np.arange(len(d))
    feats = bundle["features"]

    sequences, rows = [], []
    for _, eng in d.sort_values(["engine_id", "cycle"]).groupby("engine_id", sort=False):
        values = bundle["scaler"].transform(eng[feats].astype(float))
        padded = np.concatenate([np.repeat(values[:1], window - 1, axis=0), values])
        sequences.extend(padded[i:i + window] for i in range(len(values)))
        rows.extend(eng["_row"].tolist())

    X = np.asarray(sequences, dtype="float32")
    pred = bundle["model"].predict(X, verbose=0, batch_size=512).flatten()

    out = np.empty(len(d))
    out[rows] = pred
    return np.clip(out, 0, RUL_CAP)


# ---------------------------------------------------------------- all models together
def load_predictors():
    """Return ({model name: function(df) -> predicted RUL per row}, {model name: why it failed})."""
    predictors, problems = {}, {}

    for name, filename in TABULAR_MODELS.items():
        try:
            model = joblib.load(MODELS_DIR / filename)
            predictors[name] = lambda df, m=model: predict_rul(m, df)
        except Exception as exc:
            problems[name] = f"{type(exc).__name__}: {exc}"

    try:
        bundle = load_lstm()
        predictors["LSTM"] = lambda df, b=bundle: lstm_predict(b, df)
    except Exception as exc:
        problems["LSTM"] = f"{type(exc).__name__}: {exc}"

    return predictors, problems


# ---------------------------------------------------------------- data and evaluation
def load_test_data():
    """Processed test set plus 'ref_rul': the reference RUL at every observed cycle.

    NASA's RUL_FD001.txt gives the true RUL at the LAST observed cycle of each
    test engine (engine 1 first, engine 2 second, ...). Earlier cycles are that
    number plus the cycles still to come. All values are capped at RUL_CAP.
    """
    test = pd.read_csv(ROOT / "data" / "processed" / "test_FD001_processed.csv")
    test = test.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    final_rul = pd.read_csv(ROOT / "data" / "raw" / "RUL_FD001.txt",
                            header=None, sep=r"\s+")[0].to_numpy()
    engine_ids = sorted(test["engine_id"].unique())
    if len(engine_ids) != len(final_rul):
        raise ValueError("Number of test engines and RUL labels do not match.")

    final_by_engine = dict(zip(engine_ids, final_rul))
    last_cycle = test.groupby("engine_id")["cycle"].transform("max")
    test["ref_rul"] = (test["engine_id"].map(final_by_engine)
                       + (last_cycle - test["cycle"])).clip(upper=RUL_CAP)
    return test


def evaluate(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

