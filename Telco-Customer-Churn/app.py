
import streamlit as st
import pandas as pd
import joblib

# Load the saved machine learning pipeline
model = joblib.load("churn_model.pkl")

THRESHOLD = 0.4

st.set_page_config(page_title="Churn Predictor", page_icon="📊")

st.title("📊 Telco Customer Churn Predictor")
st.write("Enter customer details to estimate the likelihood of churn.")

with st.form("customer_form"):
    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=72, value=12
        )
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )
        backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )

    with col2:
        protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )
        support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )
        tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly = st.number_input(
            "Monthly Charges", min_value=0.0, value=70.0
        )
        total = st.number_input(
            "Total Charges", min_value=0.0, value=840.0
        )

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    probabilities = model.predict_proba(customer)[0]
    classes = list(model.classes_)
    churn_probability = probabilities[classes.index("Yes")]

    prediction = (
        "Yes" if churn_probability >= THRESHOLD else "No"
    )

    st.subheader("Prediction Results")
    st.metric(
        "Estimated Churn Probability",
        f"{churn_probability:.1%}"
    )

    if prediction == "Yes":
        st.warning("The model flags this customer as likely to churn.")
    else:
        st.success("The model does not flag this customer as likely to churn.")

    st.caption(
        f"Decision threshold: {THRESHOLD:.0%}. "
        "This is a model estimate, not a guarantee."
    )
