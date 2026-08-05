"""
app.py

Streamlit application for the Loan Approval Prediction project.
Collects raw applicant details, runs them through the same pipeline
used in training (src/predict.py), and displays the prediction with
confidence, risk level, a personalized explanation, and a downloadable
report. Model info / about / example applicants live in a second tab
in the main content area, keeping the sidebar dedicated to inputs.

Run with: streamlit run app.py
"""

import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd

from predict import load_inference_artifacts, predict_loan_status, build_feature_row

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered",
)

# Real metrics (see MODEL_COMPARISON.md) -- kept as constants
# here rather than recomputed, since retraining is a deliberate, separate
# step (python src/train.py), not something a page load should trigger.
MODEL_NAME = "Decision Tree Classifier"
DATASET_NAME = "Loan Approval Prediction Dataset (Kaggle, Archit Sharma)"
NUM_FEATURES = 10
TEST_ACCURACY = 1.0000
CV_F1_SCORE = 0.9998


@st.cache_resource
def get_artifacts():
    """Load model/scaler/feature_columns once and cache across reruns."""
    return load_inference_artifacts()


def get_risk_level(approved_probability: float):
    """
    Map an approval probability to a business-style risk label.

    Thresholds (0.85 / 0.60) are a reasonable interpretive convention
    for this kind of tool, not a value derived from the model itself --
    worth stating plainly rather than implying false precision.
    """
    if approved_probability > 0.85:
        return "LOW", "success"
    elif approved_probability > 0.60:
        return "MODERATE", "warning"
    else:
        return "HIGH", "error"


def build_personalized_reasons(raw_input: dict, engineered: pd.DataFrame) -> list:
    """
    Generate plain-language reasons for THIS specific applicant, grounded
    in the actual feature importances learned by the model (see
    MODEL_COMPARISON.md): cibil_score dominates, followed by
    loan_to_income_ratio, loan term, and asset_to_loan_ratio.

    This is a rule-of-thumb explanation for readability, not a formal
    SHAP/feature-attribution value -- worth being upfront about that
    distinction if asked during a certification demo.
    """
    reasons = []

    cibil_score = raw_input["cibil_score"]
    asset_to_loan_ratio = float(engineered["asset_to_loan_ratio"].iloc[0])
    income_annum = raw_input["income_annum"]
    loan_amount = raw_input["loan_amount"]

    if cibil_score >= 700:
        reasons.append(f"CIBIL score of {cibil_score} is strong — this is the model's single biggest positive factor.")
    elif cibil_score < 500:
        reasons.append(f"CIBIL score of {cibil_score} is low — this is the model's single biggest negative factor.")
    else:
        reasons.append(f"CIBIL score of {cibil_score} is moderate — a borderline factor rather than a clear positive or negative.")

    if asset_to_loan_ratio >= 2.0:
        reasons.append(f"Total assets comfortably cover the requested loan (ratio {asset_to_loan_ratio:.2f}, above the dataset average of ~2.2).")
    elif asset_to_loan_ratio < 1.0:
        reasons.append(f"Total assets are low relative to the requested loan (ratio {asset_to_loan_ratio:.2f}, below the dataset average of ~2.2).")

    if income_annum > loan_amount:
        reasons.append("Annual income exceeds the requested loan amount, which supports repayment capacity.")

    return reasons


def render_predict_tab(model, scaler, feature_columns):
    """Everything related to collecting inputs and showing a prediction."""
    st.write(
        "Enter applicant details in the sidebar, then click **Predict** to see "
        "whether this application would be approved, based on a model trained "
        "on historical loan approval data."
    )

    # --- Sidebar inputs (this tab's inputs; sidebar is shared globally by Streamlit,
    # but only this tab reads/uses these widgets) ---
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

    if not st.sidebar.button("Predict", type="primary"):
        st.info("Fill in the applicant details on the left, then click **Predict**.")
        return

    try:
        start_time = time.time()
        result = predict_loan_status(raw_input, model, scaler, feature_columns)
        elapsed_ms = (time.time() - start_time) * 1000
    except Exception as e:
        st.error(f"Something went wrong while predicting: {e}")
        st.stop()

    st.header("Result")

    if result["prediction"] == "Approved":
        st.success("✅ **Loan Approved**")
    else:
        st.error("❌ **Loan Rejected**")

    st.caption(f"Prediction completed in {elapsed_ms:.2f} ms")

    col1, col2 = st.columns(2)
    col1.metric("Approval Probability", f"{result['approved_probability']*100:.1f}%")
    col2.metric("Rejection Probability", f"{result['rejected_probability']*100:.1f}%")

    st.progress(result["approved_probability"])

    confidence = max(result["approved_probability"], result["rejected_probability"])
    st.metric("Model Confidence", f"{confidence*100:.2f}%")

    risk_label, risk_style = get_risk_level(result["approved_probability"])
    risk_message = f"Risk Level: {risk_label}"
    if risk_style == "success":
        st.success(risk_message)
    elif risk_style == "warning":
        st.warning(risk_message)
    else:
        st.error(risk_message)

    st.subheader("Applicant Summary")
    summary = pd.DataFrame(raw_input.items(), columns=["Field", "Value"])
    st.table(summary)

    st.subheader("Why this result?")
    engineered = build_feature_row(raw_input)

    st.write(
        "This model was trained on historical data where **CIBIL score is by "
        "far the strongest predictor** of approval, followed by the "
        "loan-to-income ratio, loan term, and how well total assets cover "
        "the requested loan (see the **About & Model Info** tab for the full "
        "feature importance breakdown)."
    )
    for reason in build_personalized_reasons(raw_input, engineered):
        st.write(f"• {reason}")

    with st.expander("See exact values sent to the model"):
        st.dataframe(engineered.T.rename(columns={0: "Value"}))

    report = pd.DataFrame({
        "prediction": [result["prediction"]],
        "approved_probability": [result["approved_probability"]],
        "rejected_probability": [result["rejected_probability"]],
        "risk_level": [risk_label],
    })

    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_filename = f"prediction_{timestamp}.csv"
    report_path = os.path.join(reports_dir, report_filename)
    report.to_csv(report_path, index=False)
    st.caption(f"Report saved to `reports/{report_filename}`")

    st.download_button(
        "⬇️ Download Prediction Report",
        report.to_csv(index=False),
        file_name=report_filename,
        mime="text/csv",
    )


def render_about_tab():
    """Model info, about, and example applicants -- centered, not sidebar."""
    st.header("Model Information")
    st.markdown(
        f"""
| Field | Value |
|---|---|
| Model | {MODEL_NAME} |
| Dataset | {DATASET_NAME} |
| Target | Loan Status (Approved / Rejected) |
| Features Used | {NUM_FEATURES} |
| Test Accuracy | {TEST_ACCURACY*100:.1f}% |
| Cross-Validated F1 (5-fold) | {CV_F1_SCORE*100:.1f}% |
"""
    )
    st.caption("See `MODEL_COMPARISON.md` in the repository for the full comparison across 7 models.")

    st.header("About")
    st.write(
        """
This is an end-to-end certification project covering the full ML
pipeline: data cleaning, feature engineering, exploratory analysis,
multi-model comparison and tuning, and deployment as an interactive
Streamlit app.

**Built with:** Streamlit, scikit-learn, pandas, numpy, matplotlib, seaborn.
"""
    )

    st.header("Example Applicants to Try")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Likely Approved")
        st.write(
            """
- Education: Graduate
- Annual Income: ₹8,000,000
- Loan Amount: ₹12,000,000
- CIBIL Score: 780
- Assets: comparable to or above the loan amount
"""
        )
    with col2:
        st.subheader("❌ Likely Rejected")
        st.write(
            """
- CIBIL Score: below 500
- Total assets well below the requested loan amount
"""
        )


def main():
    st.title("🏦 Loan Approval Predictor")

    try:
        model, scaler, feature_columns = get_artifacts()
    except FileNotFoundError as e:
        st.error(
            f"Model artifacts not found. {e}\n\n"
            "Run `python src/train.py` from the project root first, then restart this app."
        )
        st.stop()

    tab_predict, tab_about = st.tabs(["🔮 Predict", "ℹ️ About & Model Info"])

    with tab_predict:
        render_predict_tab(model, scaler, feature_columns)

    with tab_about:
        render_about_tab()

    st.markdown("---")
    st.caption("Loan Approval Prediction | Developed by Niroj Thapa")


if __name__ == "__main__":
    main()
