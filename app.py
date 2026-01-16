import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("churn_model.pkl")        # XGBoost
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")


st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("📉 Customer Churn Prediction App")
st.write(
    "Enter customer details below to predict churn probability."
)


st.subheader("Customer Information")

senior_citizen = st.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

tenure = st.slider(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)


# Start with ALL features set to 0
input_dict = dict.fromkeys(feature_names, 0)

# Numerical features (MUST match training)
input_dict["SeniorCitizen"] = 1 if senior_citizen == "Yes" else 0
input_dict["tenure"] = tenure
input_dict["MonthlyCharges"] = monthly_charges
input_dict["TotalCharges"] = total_charges

# Categorical features (handle drop_first=True safely)

# Contract (Month-to-month is baseline → DO NOTHING)
if contract != "Month-to-month":
    col = f"Contract_{contract}"
    if col in input_dict:
        input_dict[col] = 1

# Payment Method
pay_col = f"PaymentMethod_{payment_method}"
if pay_col in input_dict:
    input_dict[pay_col] = 1

# Paperless Billing
if paperless_billing == "Yes":
    if "PaperlessBilling_Yes" in input_dict:
        input_dict["PaperlessBilling_Yes"] = 1


input_df = pd.DataFrame([input_dict])
input_df = input_df[feature_names]   # enforce order


num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
input_df[num_cols] = scaler.transform(input_df[num_cols])


if st.button("Predict Churn"):
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    st.write(f"### Churn Probability: **{prob:.2%}**")

    if prob >= 0.70:
        st.error("High Risk of Churn")
        st.write("Immediate retention action recommended.")
    elif prob >= 0.40:
        st.warning("Medium Risk of Churn")
        st.write("Monitor and offer targeted incentives.")
    else:
        st.success("Low Risk of Churn")
        st.write("Customer is likely to stay.")


st.markdown("---")
st.caption("Model-based churn prediction | Streamlit Deployment")
