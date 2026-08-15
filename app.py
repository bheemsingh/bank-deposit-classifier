"""
Streamlit app: Will this customer subscribe to a term deposit?

Three tabs:
  1. Overview & EDA        - dataset description + a handful of narrative charts
  2. Model Comparison      - the 5-model x 6-metric comparison table + chart
  3. Predict a Customer    - pick a model, fill in a customer profile, get a live prediction

Data & models are produced by train_models.py (run that first).
"""

import json
import os

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
MAX_UPLOAD_ROWS = 2233  # size of test_data.csv; keeps inference light on Streamlit free tier

st.set_page_config(page_title="Term Deposit Subscription Predictor", layout="wide")


@st.cache_data
def get_data():
    return load_and_engineer(DATA_PATH)


@st.cache_data
def get_metrics():
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        return json.load(f)


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


st.title("Will this customer subscribe to a term deposit?")
st.caption(
    "Bank Marketing dataset (UCI) — a Portuguese bank's phone-campaign records, "
    "used here to predict whether a customer subscribes to a term deposit."
)

tab_eda, tab_compare, tab_predict = st.tabs(
    ["Overview & EDA", "Model Comparison", "Predict a Customer"]
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

# ---------------------------------------------------------- Tab 2: Compare
with tab_compare:
    st.subheader("Model comparison")
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
    st.info(
        "See README.md 'Observations' table for the written analysis of each "
        "model's performance on this dataset (filled in from the metrics above)."
    )

# ---------------------------------------------------------- Tab 3: Predict
with tab_predict:
    st.subheader("Predict a customer")

    model_key = st.selectbox(
        "Choose a model",
        options=list(get_metrics().keys()),
        format_func=lambda k: get_metrics()[k]["display_name"],
    )

    df_ref = get_data()

    # with st.form("customer_form"):
    #     left, right = st.columns(2)
    #
    #     with left:
    #         age = st.slider("Age", 18, 95, 40)
    #         job = st.selectbox("Job", sorted(df_ref["job"].unique()))
    #         marital = st.selectbox("Marital status", sorted(df_ref["marital"].unique()))
    #         education = st.selectbox("Education", sorted(df_ref["education"].unique()))
    #         default = st.selectbox("Has credit in default?", sorted(df_ref["default"].unique()))
    #         balance = st.number_input("Account balance (EUR)", value=1000, step=100)
    #         housing = st.selectbox("Has housing loan?", sorted(df_ref["housing"].unique()))
    #         loan = st.selectbox("Has personal loan?", sorted(df_ref["loan"].unique()))
    #
    #     with right:
    #         contact = st.selectbox("Contact type", sorted(df_ref["contact"].unique()))
    #         day = st.slider("Day of month contacted", 1, 31, 15)
    #         month = st.selectbox("Month contacted", sorted(df_ref["month"].unique()))
    #         duration = st.slider("Last call duration (seconds)", 0, 3000, 180)
    #         campaign = st.slider("Contacts during this campaign", 1, 50, 2)
    #         pdays = st.number_input("Days since last contact (-1 = never)", value=-1, step=1)
    #         previous = st.slider("Previous contacts before this campaign", 0, 50, 0)
    #         poutcome = st.selectbox("Previous campaign outcome", sorted(df_ref["poutcome"].unique()))
    #
    #     submitted = st.form_submit_button("Predict")
    #
    # if submitted:
    #     raw_row = pd.DataFrame([{
    #         "age": age, "job": job, "marital": marital, "education": education,
    #         "default": default, "balance": balance, "housing": housing, "loan": loan,
    #         "contact": contact, "day": day, "month": month, "duration": duration,
    #         "campaign": campaign, "pdays": pdays, "previous": previous,
    #         "poutcome": poutcome, "deposit": "no",  # placeholder, unused for inference
    #     }])
    #
    #     preds, probs = predict(model_key, raw_row)
    #     pred, prob = preds[0], probs[0]
    #
    #     if pred == 1:
    #         st.success(f"Prediction: **Subscribes** (probability {prob:.1%})")
    #     else:
    #         st.warning(f"Prediction: **Does not subscribe** (probability of yes: {prob:.1%})")

    #st.divider()
    st.subheader("Dataset upload from a CSV")
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
