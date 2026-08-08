"""
Feature engineering and preprocessing pipeline for the bank term-deposit
subscription dataset (UCI Bank Marketing).

The raw dataset already has 16 columns, but several of the numeric fields
(age, balance, pdays, campaign) are more informative to a linear/distance
model once turned into bands or derived flags. That engineering lives here
so both train_models.py and app.py apply identical transforms.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "deposit"

RAW_CATEGORICAL_COLUMNS = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "poutcome",
]

RAW_NUMERIC_COLUMNS = [
    "age", "balance", "day", "duration", "campaign", "pdays", "previous",
]

ENGINEERED_CATEGORICAL_COLUMNS = ["age_band", "balance_tier"]
ENGINEERED_NUMERIC_COLUMNS = ["was_contacted_before", "campaign_intensity"]

CATEGORICAL_COLUMNS = RAW_CATEGORICAL_COLUMNS + ENGINEERED_CATEGORICAL_COLUMNS
NUMERIC_COLUMNS = RAW_NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS

AGE_BAND_BINS = [0, 25, 35, 45, 55, 65, 120]
AGE_BAND_LABELS = ["under_25", "25_34", "35_44", "45_54", "55_64", "65_plus"]

BALANCE_TIER_LABELS = ["very_low", "low", "medium", "high", "very_high"]

# Fixed quantile edges for balance (computed once from the full training
# dataset's 0/20/40/60/80/100th percentiles). Using fixed edges instead of a
# per-call pd.qcut means single-row inference (live predictions in the
# Streamlit app) buckets consistently with how the models were trained,
# instead of qcut failing/behaving differently on a 1-row input.
BALANCE_BIN_EDGES = [-float("inf"), 62.0, 337.0, 862.6, 2223.0, float("inf")]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns on top of the raw bank marketing columns.

    - age_band: life-stage bucket instead of raw age, so tree/linear models
      pick up on nonlinear age effects without needing polynomial terms.
    - balance_tier: quantile bucket of account balance (balance is heavily
      right-skewed with negative values, so quantile bins beat fixed cutoffs).
    - was_contacted_before: pdays == -1 means "never contacted previously" in
      the raw encoding; making this an explicit flag avoids models treating
      -1 as if it were a small numeric distance.
    - campaign_intensity: campaign count clipped and scaled to flag unusually
      high-contact customers (a proxy for annoyance / diminishing returns).
    """
    out = df.copy()

    out["age_band"] = pd.cut(
        out["age"], bins=AGE_BAND_BINS, labels=AGE_BAND_LABELS, right=False
    ).astype(str)

    out["balance_tier"] = pd.cut(
        out["balance"], bins=BALANCE_BIN_EDGES, labels=BALANCE_TIER_LABELS
    ).astype(str)

    out["was_contacted_before"] = (out["pdays"] != -1).astype(int)

    out["campaign_intensity"] = out["campaign"].clip(upper=10)

    return out


def load_and_engineer(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return engineer_features(df)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("num", StandardScaler(), NUMERIC_COLUMNS),
        ]
    )


def feature_columns() -> list:
    return CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def split_X_y(df: pd.DataFrame):
    X = df[feature_columns()]
    y = (df[TARGET_COLUMN].astype(str).str.lower() == "yes").astype(int)
    return X, y
