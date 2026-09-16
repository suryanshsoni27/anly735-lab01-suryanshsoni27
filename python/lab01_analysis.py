"""
ANLY 735 — Research Seminar in Predictive AI
Replication Laboratory #1 | Evaluation Design Matters
Student Python Analysis

Laboratory question:
Does changing the evaluation design change estimated predictive
performance—and does it change which model appears stronger?

This script intentionally produces the analytical evidence.
Interpretation, the FAIR audit, replication verdict, and claim boundaries
belong in your replication-lab.qmd.

Dataset:
UCI Bike Sharing Dataset
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
"""

from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. REPRODUCIBILITY SETTINGS
# ============================================================

RANDOM_SEED = 735
TEST_SIZE = 0.20

np.random.seed(RANDOM_SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ANALYSIS_DIR = ROOT / "analysis"

DATA_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR.mkdir(exist_ok=True)

ZIP_PATH = DATA_DIR / "Bike-Sharing-Dataset.zip"
CSV_PATH = DATA_DIR / "hour.csv"

UCI_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00275/"
    "Bike-Sharing-Dataset.zip"
)


# ============================================================
# 2. ACQUIRE THE DATA REPRODUCIBLY
# ============================================================

if not CSV_PATH.exists():
    print("Downloading the UCI Bike Sharing Dataset...")

    try:
        urllib.request.urlretrieve(UCI_URL, ZIP_PATH)

        with zipfile.ZipFile(ZIP_PATH, "r") as archive:
            archive.extract("hour.csv", DATA_DIR)

    except Exception as exc:
        raise RuntimeError(
            "\nThe dataset could not be downloaded automatically.\n"
            "Check your internet connection and try again.\n"
            f"Original error: {exc}"
        ) from exc

print(f"Using data: {CSV_PATH}")


# ============================================================
# 3. LOAD DATA AND ESTABLISH CHRONOLOGY
# ============================================================

df = pd.read_csv(CSV_PATH)

df["datetime"] = pd.to_datetime(
    df["dteday"].astype(str)
    + " "
    + df["hr"].astype(str).str.zfill(2)
    + ":00:00"
)

df = df.sort_values("datetime").reset_index(drop=True)

print("\nDataset dimensions:")
print(df.shape)

print("\nObservation period:")
print(f"{df['datetime'].min()} to {df['datetime'].max()}")


# ============================================================
# 4. DEFINE THE PREDICTION PROBLEM
# ============================================================

TARGET = "cnt"

CATEGORICAL_FEATURES = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
]

NUMERIC_FEATURES = [
    "temp",
    "atemp",
    "hum",
    "windspeed",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# IMPORTANT:
# "casual" and "registered" are deliberately excluded.
# Their sum defines the target variable "cnt".
#
# "instant" is an identifier.
# "dteday" establishes chronology but is not used directly as a predictor.

X = df[FEATURES]
y = df[TARGET]


# ============================================================
# 5. DEFINE PREPROCESSING
# ============================================================

def make_preprocessor():
    """
    Create a fresh preprocessing pipeline.

    Numeric predictors:
      - median imputation
      - standardization

    Categorical predictors:
      - most-frequent imputation
      - one-hot encoding

    Preprocessing is fitted only on the training data because it is
    contained inside each scikit-learn Pipeline.
    """

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


# ============================================================
# 6. DEFINE THE TWO COMPETING MODELS
# ============================================================

def make_ridge():
    """Return a fresh Ridge Regression pipeline."""

    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def make_random_forest():
    """Return a fresh Random Forest pipeline."""

    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    evaluation_design,
    model_name,
):
    """
    Fit a model on training data and evaluate it on untouched test data.

    Returns RMSE and MAE. Lower values indicate smaller prediction errors.
    """

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    return {
        "Evaluation Design": evaluation_design,
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
    }


results = []


# ============================================================
# 8. EXPERIMENT 1 — RANDOM EVALUATION
# ============================================================

X_train_random, X_test_random, y_train_random, y_test_random = (
    train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )
)

print("\n" + "=" * 60)
print("EXPERIMENT 1 — RANDOM EVALUATION")
print("=" * 60)

print(f"Training observations: {len(X_train_random)}")
print(f"Testing observations : {len(X_test_random)}")

results.append(
    evaluate_model(
        make_ridge(),
        X_train_random,
        y_train_random,
        X_test_random,
        y_test_random,
        "Random",
        "Ridge Regression",
    )
)

results.append(
    evaluate_model(
        make_random_forest(),
        X_train_random,
        y_train_random,
        X_test_random,
        y_test_random,
        "Random",
        "Random Forest",
    )
)


# ============================================================
# 9. EXPERIMENT 2 — TEMPORAL EVALUATION
# ============================================================

split_index = int(len(df) * (1 - TEST_SIZE))

train_temporal = df.iloc[:split_index]
test_temporal = df.iloc[split_index:]

X_train_temporal = train_temporal[FEATURES]
y_train_temporal = train_temporal[TARGET]

X_test_temporal = test_temporal[FEATURES]
y_test_temporal = test_temporal[TARGET]

print("\n" + "=" * 60)
print("EXPERIMENT 2 — TEMPORAL EVALUATION")
print("=" * 60)

print("\nTraining period:")
print(
    f"{train_temporal['datetime'].min()} "
    f"to {train_temporal['datetime'].max()}"
)

print("\nTesting period:")
print(
    f"{test_temporal['datetime'].min()} "
    f"to {test_temporal['datetime'].max()}"
)

print(f"\nTraining observations: {len(train_temporal)}")
print(f"Testing observations : {len(test_temporal)}")

results.append(
    evaluate_model(
        make_ridge(),
        X_train_temporal,
        y_train_temporal,
        X_test_temporal,
        y_test_temporal,
        "Temporal",
        "Ridge Regression",
    )
)

results.append(
    evaluate_model(
        make_random_forest(),
        X_train_temporal,
        y_train_temporal,
        X_test_temporal,
        y_test_temporal,
        "Temporal",
        "Random Forest",
    )
)


# ============================================================
# 10. DISPLAY THE EVIDENCE
# ============================================================

results_df = pd.DataFrame(results)

display_results = results_df.copy()
display_results["RMSE"] = display_results["RMSE"].round(2)
display_results["MAE"] = display_results["MAE"].round(2)

print("\n" + "=" * 60)
print("ANLY 735 — EVALUATION RESULTS")
print("=" * 60 + "\n")

print(display_results.to_string(index=False))


# ============================================================
# 11. DISPLAY MODEL RANKINGS
# ============================================================

print("\n" + "=" * 60)
print("MODEL RANKING BY RMSE")
print("=" * 60)

for design in ["Random", "Temporal"]:

    subset = (
        display_results[
            display_results["Evaluation Design"] == design
        ]
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    print(f"\n{design} evaluation:")

    for position, row in subset.iterrows():
        print(
            f"{position + 1}. {row['Model']} "
            f"(RMSE = {row['RMSE']}, MAE = {row['MAE']})"
        )


# ============================================================
# 12. SAVE REPRODUCIBLE RESULTS
# ============================================================

output_path = ANALYSIS_DIR / "lab01_results.csv"

display_results.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("RESULTS SAVED")
print("=" * 60)

print(output_path)


# ============================================================
# 13. RESEARCHER HANDOFF
# ============================================================

print("\n" + "=" * 60)
print("YOUR JOB AS THE RESEARCHER")
print("=" * 60)

print(
    """
The script has produced evidence. It has not interpreted that evidence for you.

Return to replication-lab.qmd and investigate:

1. How did RMSE and MAE change across evaluation designs?
2. Did the model ranking change?
3. Which conclusions remained stable?
4. Which conclusions require qualification?
5. Which evaluation design better represents the intended deployment scenario?
6. What does the evidence permit you to claim?
7. What does the evidence NOT permit you to claim?

Do not chase identical numbers. Investigate reproducibility.
"""
)
