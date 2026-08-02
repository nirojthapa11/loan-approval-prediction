"""
app.py

Streamlit application for the Loan Approval Prediction project.
Collects raw applicant details, runs them through the same pipeline
used in training (src/predict.py), and displays the prediction with
confidence and a brief explanation of the key drivers.

Run with: streamlit run app.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd

from predict import load_inference_artifacts, predict_loan_status, build_feature_row

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered",
)


@st.cache_resource
def get_artifacts():
    """Load model/scaler/feature_columns once and cache across reruns."""
    return load_inference_artifacts()


def main():
    st.title("🏦 Loan Approval Predictor")
    st.write(
        "Enter applicant details in the sidebar, then click **Predict** to see "
        "whether this application would be approved, based on a model trained "
        "on historical loan approval data."
    )

    # --- Load artifacts, with a clear error if training hasn't been run ---
    try:
        model, scaler, feature_columns = get_artifacts()
    except FileNotFoundError as e:
        st.error(
            f"Model artifacts not found. {e}\n\n"
            "Run `python src/train.py` from the project root first, then restart this app."
        )
        st.stop()

    # --- Sidebar inputs ---
    st.sidebar.header("Applicant Details")

    no_of_dependents = st.sidebar.slider("Number of Dependents", 0, 5, 2)
    education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.sidebar.selectbox("Self Employed", ["No", "Yes"])

    st.sidebar.header("Financial Details")
    income_annum = st.sidebar.number_input(
        "Annual Income (₹)", min_value=0, value=5_000_000, step=100_000
    )
    loan_amount = st.sidebar.number_input(
        "Loan Amount Requested (₹)", min_value=0, value=15_000_000, step=100_000
    )
    loan_term = st.sidebar.slider("Loan Term (years)", 2, 20, 10)
    cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, 700)

    st.sidebar.header("Asset Values")
    residential_assets_value = st.sidebar.number_input(
        "Residential Assets (₹)", min_value=0, value=3_000_000, step=100_000
    )
    commercial_assets_value = st.sidebar.number_input(
        "Commercial Assets (₹)", min_value=0, value=2_000_000, step=100_000
    )
    luxury_assets_value = st.sidebar.number_input(
        "Luxury Assets (₹)", min_value=0, value=5_000_000, step=100_000
    )
    bank_asset_value = st.sidebar.number_input(
        "Bank Assets (₹)", min_value=0, value=3_000_000, step=100_000
    )

    raw_input = {
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
    }

    # --- Prediction ---
    if st.sidebar.button("Predict", type="primary"):
        try:
            result = predict_loan_status(raw_input, model, scaler, feature_columns)
        except Exception as e:
            st.error(f"Something went wrong while predicting: {e}")
            st.stop()

        st.header("Result")

        if result["prediction"] == "Approved":
            st.success(f"✅ **Loan Approved**")
        else:
            st.error(f"❌ **Loan Rejected**")

        col1, col2 = st.columns(2)
        col1.metric("Approval Probability", f"{result['approved_probability']*100:.1f}%")
        col2.metric("Rejection Probability", f"{result['rejected_probability']*100:.1f}%")

        st.progress(result["approved_probability"])

        # --- Explanation, grounded in the actual model's learned feature importances ---
        st.subheader("Why this result?")

        engineered = build_feature_row(raw_input)
        asset_to_loan_ratio = float(engineered["asset_to_loan_ratio"].iloc[0])
        total_assets_value = float(engineered["total_assets_value"].iloc[0])

        st.write(
            f"""
This model was trained on historical data where **CIBIL score is by far the
strongest predictor** of approval (roughly 81% of the model's decision
weight), followed by the loan-to-income ratio, loan term, and how well an
applicant's total assets cover the requested loan amount.

**This application's key numbers:**
- CIBIL Score: **{cibil_score}** {"(above the dataset's approved-applicant average of ~703)" if cibil_score >= 703 else "(below the dataset's approved-applicant average of ~703)"}
- Total Assets: ₹{total_assets_value:,.0f}
- Asset-to-Loan Ratio: **{asset_to_loan_ratio:.2f}** (dataset average ≈ 2.23 — a higher ratio means assets more comfortably cover the requested loan)
            """
        )

        with st.expander("See exact values sent to the model"):
            st.dataframe(engineered.T.rename(columns={0: "Value"}))

    else:
        st.info("Fill in the applicant details on the left, then click **Predict**.")

    st.markdown("---")

    st.caption(
        "Developed by Niroj Thapa | "
        "Loan Approval Prediction Project"
    )

if __name__ == "__main__":
    main()
