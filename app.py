import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd

from predict import load_inference_artifacts, predict_loan_status

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

    st.subheader("Prediction")

    if st.button("🔍 Predict Loan Status"):

        with st.spinner("Analyzing application..."):

            result = predict_loan_status(
                raw_input,
                model,
                scaler,
                feature_columns,
            )

        if result["prediction"] == "Approved":
            st.success("✅ Loan Approved")

        else:
            st.error("❌ Loan Rejected")
            
   
    st.subheader("Current Input Values")

    st.dataframe(
        pd.DataFrame([raw_input])
        )
    
    st.markdown("---")
    st.caption(
        "Loan Approval Prediction System | Streamlit Prototype"
    )
    
    
if __name__ == "__main__":
    main()
