from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

RUL_CAP = 125

FEATURES = ["op_setting_1", "op_setting_2",
            "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
            "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
            "sensor_17", "sensor_20", "sensor_21"]


def load_model(name="random_forest"):
    return joblib.load(MODELS_DIR / f"{name}.pkl")


def predict_rul(model, df):
    """Predict RUL for every row of df."""
    return np.clip(model.predict(df[FEATURES]), 0, RUL_CAP)


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