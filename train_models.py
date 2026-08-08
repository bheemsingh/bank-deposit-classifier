"""
Trains all 6 required classification models on the bank term-deposit
dataset, evaluates them with the 6 required metrics, and persists:
  - model/<model_name>.pkl   (fitted estimator)
  - model/preprocessor.pkl   (fitted ColumnTransformer, shared by all models)
  - model/metrics.json       (metrics table consumed by app.py)
  - test_data.csv            (raw held-out rows, for grading reproducibility)

Run with: python train_models.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from preprocessing import build_preprocessor, load_and_engineer, split_X_y

DATA_PATH = os.path.join("data", "bank.csv")
MODEL_DIR = "model"
TEST_DATA_PATH = "test_data.csv"
RANDOM_STATE = 42

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "decision_tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "knn": KNeighborsClassifier(n_neighbors=15),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
    ),
}

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "kNN",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest (Ensemble)",
}


def evaluate(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_and_engineer(DATA_PATH)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["deposit"]
    )
    test_df.to_csv(TEST_DATA_PATH, index=False)

    X_train, y_train = split_X_y(train_df)
    X_test, y_test = split_X_y(test_df)

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # GaussianNB needs dense arrays; the one-hot output from ColumnTransformer
    # is sparse, so densify once and reuse for every model (small dataset, fine).
    X_train_t = np.asarray(X_train_t.todense()) if hasattr(X_train_t, "todense") else X_train_t
    X_test_t = np.asarray(X_test_t.todense()) if hasattr(X_test_t, "todense") else X_test_t

    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))

    metrics = {}
    for key, model in MODELS.items():
        model.fit(X_train_t, y_train)
        y_pred = model.predict(X_test_t)
        y_prob = model.predict_proba(X_test_t)[:, 1]

        metrics[key] = {
            "display_name": DISPLAY_NAMES[key],
            **evaluate(y_test, y_pred, y_prob),
        }

        joblib.dump(model, os.path.join(MODEL_DIR, f"{key}.pkl"))
        print(f"{DISPLAY_NAMES[key]:30s} -> {metrics[key]}")

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved {len(MODELS)} models, preprocessor, and metrics.json to '{MODEL_DIR}/'")
    print(f"Saved held-out test split ({len(test_df)} rows) to '{TEST_DATA_PATH}'")


if __name__ == "__main__":
    main()
