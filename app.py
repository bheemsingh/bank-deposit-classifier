"""
Streamlit app: Will this customer subscribe to a term deposit?

Three tabs:
  1. Overview & EDA        - dataset description + a handful of narrative charts
  2. Predict a Customer    - pick a model, fill in a customer profile, get a live prediction
  3. Model Comparison      - the 5-model x 6-metric comparison table + chart

Data & models are produced by train_models.py (run that first).
"""

import json
import os
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix

from preprocessing import (
    RAW_CATEGORICAL_COLUMNS,
    RAW_NUMERIC_COLUMNS,
    engineer_features,
    feature_columns,
    load_and_engineer,
)
from train_models import evaluate

RAW_INPUT_COLUMNS = RAW_CATEGORICAL_COLUMNS + RAW_NUMERIC_COLUMNS

MODEL_DIR = "model"
DATA_PATH = os.path.join("data", "bank.csv")
README_PATH = "README.md"
MAX_UPLOAD_ROWS = 2233  # size of test_data.csv; keeps inference light on Streamlit free tier

st.set_page_config(page_title="Term Deposit Subscription Predictor", layout="wide")


@st.cache_data
def get_data():
    return load_and_engineer(DATA_PATH)


@st.cache_data
def get_metrics():
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        return json.load(f)


@st.cache_data
def get_observations():
    """Parse the '### Observations' markdown table out of README.md into
    {display_name: observation_text}, plus the overall-winner text."""
    with open(README_PATH, encoding="utf-8") as f:
        content = f.read()

    section = re.search(r"### Observations\n(.*?)(?:\n##|\Z)", content, re.S)
    if not section:
        return {}, None

    rows = [
        line for line in section.group(1).strip().splitlines()
        if line.strip().startswith("|") and not re.fullmatch(r"[\s|:-]+", line.strip())
    ][1:]  # drop header row, keep data rows

    observations, winner = {}, None
    for row in rows:
        cells = [re.sub(r"\*+", "", c).strip() for c in row.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        name, text = cells[0], cells[1]
        if "overall winner" in name.lower():
            winner = text
        else:
            observations[name] = text

    return observations, winner


@st.cache_resource
def get_preprocessor():
    return joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))


@st.cache_resource
def get_model(model_key: str):
    return joblib.load(os.path.join(MODEL_DIR, f"{model_key}.pkl"))


def predict(model_key: str, raw_df: pd.DataFrame):
    """Engineer features, preprocess, and run predict/predict_proba for a batch of raw rows."""
    engineered = engineer_features(raw_df)
    X = engineered[feature_columns()]

    X_t = get_preprocessor().transform(X)
    if hasattr(X_t, "todense"):
        X_t = X_t.todense()

    model = get_model(model_key)
    return model.predict(X_t), model.predict_proba(X_t)[:, 1]


def compare_models_on(raw_df: pd.DataFrame, y_true: pd.Series) -> pd.DataFrame:
    """Run every model on raw_df and score each against y_true. Same shape as
    the training-time table in the Model Comparison tab, but computed live
    from whatever file was uploaded."""
    rows = {}
    for model_key, info in get_metrics().items():
        preds, probs = predict(model_key, raw_df)
        m = evaluate(y_true, preds, probs)
        rows[info["display_name"]] = {
            "Accuracy": m["accuracy"], "AUC": m["auc"], "Precision": m["precision"],
            "Recall": m["recall"], "F1": m["f1"], "MCC": m["mcc"],
        }
    return pd.DataFrame(rows).T


st.title("Will this customer subscribe to a term deposit?")
st.caption(
    "Bank Marketing dataset (UCI) — a Portuguese bank's phone-campaign records, "
    "used here to predict whether a customer subscribes to a term deposit."
)

tab_eda, tab_predict, tab_compare = st.tabs(
    ["Overview & EDA", "Predict a Customer", "Model Comparison"]
)

# ---------------------------------------------------------------- Tab 1: EDA
with tab_eda:
    df = get_data()

    st.subheader("Problem statement")
    st.markdown(
        "Predict whether a bank customer will subscribe to a term deposit "
        "(`deposit` = yes/no) after being contacted in a phone marketing "
        "campaign, using customer demographics, account info, and details "
        "of the current and previous campaign contacts. This is a **binary "
        "classification** problem."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(df))
    col2.metric("Raw features", len(RAW_CATEGORICAL_COLUMNS) + len(RAW_NUMERIC_COLUMNS))
    col3.metric("Subscribed rate", f"{(df['deposit'].str.lower() == 'yes').mean():.1%}")

    st.markdown("### Who subscribes, and when?")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=df, x="deposit", y="duration", ax=ax)
        ax.set_title("Call duration vs. subscription outcome")
        st.pyplot(fig)
        st.caption("Longer calls strongly associate with 'yes' — engaged prospects talk longer.")

    with c2:
        fig, ax = plt.subplots(figsize=(5, 4))
        month_order = ["jan", "feb", "mar", "apr", "may", "jun",
                        "jul", "aug", "sep", "oct", "nov", "dec"]
        rate_by_month = (
            df.assign(subscribed=df["deposit"].str.lower() == "yes")
            .groupby("month")["subscribed"].mean()
            .reindex(month_order)
        )
        rate_by_month.plot(kind="bar", ax=ax, color="#4C72B0")
        ax.set_ylabel("Subscription rate")
        ax.set_title("Seasonality of contact month")
        st.pyplot(fig)
        st.caption("Some months convert far better than others — a seasonality signal for campaign timing.")

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots(figsize=(5, 4))
        rate_by_poutcome = (
            df.assign(subscribed=df["deposit"].str.lower() == "yes")
            .groupby("poutcome")["subscribed"].mean()
            .sort_values()
        )
        rate_by_poutcome.plot(kind="barh", ax=ax, color="#55A868")
        ax.set_xlabel("Subscription rate")
        ax.set_title("Effect of previous campaign outcome")
        st.pyplot(fig)
        st.caption("Customers who previously said 'yes' convert far more often on this campaign too.")

    with c4:
        fig, ax = plt.subplots(figsize=(5, 4))
        rate_by_age = (
            df.assign(subscribed=df["deposit"].str.lower() == "yes")
            .groupby("age_band")["subscribed"].mean()
            .reindex(["under_25", "25_34", "35_44", "45_54", "55_64", "65_plus"])
        )
        rate_by_age.plot(kind="bar", ax=ax, color="#C44E52")
        ax.set_ylabel("Subscription rate")
        ax.set_title("Subscription rate by life-stage (age band)")
        st.pyplot(fig)
        st.caption("Younger and retirement-age customers subscribe at noticeably higher rates than the middle bands.")

    with st.expander("Raw sample rows"):
        st.dataframe(df.head(20))

# ---------------------------------------------------------- Tab 3: Compare
with tab_compare:
    st.subheader("Model comparison")

    upload_table = st.session_state.get("upload_compare_table")

    if upload_table is not None:
        table = upload_table
        st.caption(
            f"Based on the uploaded file **{st.session_state['upload_compare_filename']}** "
            "— go to 'Predict a Customer' and upload a different file to refresh this."
        )
    else:
        st.caption(
            "Upload and predict a labeled CSV on the 'Predict a Customer' tab to see "
            "a comparison based on that file. Showing the training-time held-out split "
            "for now."
        )
        metrics = get_metrics()
        table = pd.DataFrame(
            {
                m["display_name"]: {
                    "Accuracy": m["accuracy"],
                    "AUC": m["auc"],
                    "Precision": m["precision"],
                    "Recall": m["recall"],
                    "F1": m["f1"],
                    "MCC": m["mcc"],
                }
                for m in metrics.values()
            }
        ).T

    st.dataframe(table.style.format("{:.4f}").background_gradient(cmap="Blues", axis=0))

    st.markdown("### Metric comparison chart")
    fig, ax = plt.subplots(figsize=(10, 5))
    table.plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", ncol=3, fontsize=8)
    ax.set_title("All 6 metrics across all 5 models")
    st.pyplot(fig)

    st.markdown("### Observations")
    st.caption("Analyst write-up from README.md (based on the training-time held-out split).")
    observations, winner = get_observations()
    for display_name in table.index:
        text = observations.get(display_name)
        if text:
            with st.expander(display_name):
                st.markdown(text)
    if winner:
        st.success(f"**Overall winner:** {winner}")

# ---------------------------------------------------------- Tab 2: Predict
with tab_predict:
    st.subheader("Predict a customer")

    model_key = st.selectbox(
        "Choose a model",
        options=list(get_metrics().keys()),
        format_func=lambda k: get_metrics()[k]["display_name"],
    )

    df_ref = get_data()

    st.subheader("Upload Dataset from a CSV")
    st.caption(
        f"Upload a CSV of raw customer rows (same columns as `data/bank.csv` / "
        f"`test_data.csv` — a `deposit` column, if present, is ignored). Capped at "
        f"{MAX_UPLOAD_ROWS} rows to keep inference light on Streamlit's free tier."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_predict_csv")

    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read that file as CSV: {e}")
            st.stop()

        missing_cols = [c for c in RAW_INPUT_COLUMNS if c not in upload_df.columns]
        if missing_cols:
            st.error(f"Missing required column(s): {', '.join(missing_cols)}")
            st.stop()

        if len(upload_df) > MAX_UPLOAD_ROWS:
            st.warning(
                f"File has {len(upload_df)} rows; only the first {MAX_UPLOAD_ROWS} "
                "will be scored."
            )
            upload_df = upload_df.head(MAX_UPLOAD_ROWS)

        preds, probs = predict(model_key, upload_df)

        results = upload_df.copy()
        results["predicted_deposit"] = np.where(preds == 1, "yes", "no")
        results["subscribe_probability"] = probs

        st.success(f"Scored {len(results)} rows with {get_metrics()[model_key]['display_name']}.")
        st.dataframe(results)

        st.download_button(
            "Download predictions as CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

        if "deposit" in upload_df.columns:
            st.markdown("### Metrics + confusion matrix")
            st.caption(
                "The uploaded CSV includes a `deposit` column, so predictions "
                "can be scored against it."
            )

            y_true = (upload_df["deposit"].astype(str).str.lower() == "yes").astype(int)
            batch_metrics = evaluate(y_true, preds, probs)

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Accuracy", f"{batch_metrics['accuracy']:.4f}")
            m2.metric("AUC", f"{batch_metrics['auc']:.4f}")
            m3.metric("Precision", f"{batch_metrics['precision']:.4f}")
            m4.metric("Recall", f"{batch_metrics['recall']:.4f}")
            m5.metric("F1", f"{batch_metrics['f1']:.4f}")
            m6.metric("MCC", f"{batch_metrics['mcc']:.4f}")

            cm = confusion_matrix(y_true, preds)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["no", "yes"], yticklabels=["no", "yes"], ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion matrix")
            st.pyplot(fig)

            st.session_state["upload_compare_table"] = compare_models_on(upload_df, y_true)
            st.session_state["upload_compare_filename"] = uploaded_file.name
            st.info("Model comparison for this file is now shown on the 'Model Comparison' tab.")