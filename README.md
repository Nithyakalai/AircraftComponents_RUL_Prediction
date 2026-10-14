# AeroLife: Aircraft Engine Remaining Useful Life Predictor

AeroLife estimates the remaining useful life (RUL) of aircraft engines using the NASA C-MAPSS FD001 dataset. It includes a Flask dashboard and Jupyter notebooks for data preprocessing and model experiments with Random Forest, XGBoost, and LSTM models.

## Requirements

- Python 3.10 or newer
- The packages listed in [`requirements.txt`](requirements.txt)

## Setup on Windows

Open PowerShell in the project root and create/activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation scripts, activate the environment in Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

## Run the dashboard

From the project root, start the Flask app:

```powershell
python dashboard\app.py
```

Open <http://127.0.0.1:5000> in a browser, create an account using **Sign up**, and then log in. The app creates or updates `dashboard/users.db` on startup. The first dashboard page may take a little longer while the saved models generate predictions and evaluation metrics.

The dashboard expects the processed test data and RUL labels under `data\processed` and `data\raw`, and the saved predictors under `models`. These files are included in the project. Random Forest and XGBoost predictions use their saved `.pkl` models; the LSTM also requires TensorFlow and its saved `.keras` model, scaler, and feature list. If a model cannot be loaded, the dashboard reports it and can continue with other available models.

Stop the development server with **Ctrl+C**.

## Run the notebooks

Start Jupyter from the project root:

```powershell
jupyter notebook
```

Open the notebook you want in the browser:

- `data\preprocessing.ipynb` — prepare the raw FD001 files and write processed CSVs to `data\processed`.
- `notebooks\random_forest.ipynb` — train/evaluate Random Forest and save its model and results.
- `notebooks\xgboost.ipynb` — run XGBoost experiments and save its model and results.
- `notebooks\lstm.ipynb` — train/evaluate the LSTM and save its model, scaler, features, and results.

Run notebook cells in order. Model-training notebooks can take a while and may overwrite files in `models` and `results`. The dashboard itself does not require rerunning training when the included processed data and model files are present.

## Project folders

| Folder | Contents |
| --- | --- |
| `dashboard` | Flask application, templates, styles, and SQLite user database |
| `data\raw` | Raw FD001 training/test data and final RUL labels |
| `data\processed` | Processed training and test CSV files used by the models |
| `models` | Saved Random Forest, XGBoost, and LSTM artifacts |
| `notebooks` | Model training and evaluation notebooks |
| `results\figures` | Generated model plots |
| `results\metrics` | Generated evaluation metrics |
| `src` | Shared prediction and evaluation helpers |

