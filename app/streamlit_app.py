import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("model.joblib")

# Define feature names in exact order used during training
feature_order = [
    'Gender', 'Married', 'Dependents', 'Education',
    'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
    'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

# Title
st.title("🏦 Loan Approval Predictor")

# Sidebar for user input
st.sidebar.header("📝 Enter Applicant Details")

# User inputs (top features from SHAP)
credit_history = st.sidebar.selectbox("Credit History", [1.0, 0.0])
married = st.sidebar.selectbox("Married", ['Yes', 'No'])
education = st.sidebar.selectbox("Education", ['Graduate', 'Not Graduate'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
loan_amount = st.sidebar.slider("Loan Amount (in thousands)", 10, 500, 100)

# Default (pre-filled) values for the rest
defaults = {
    'Dependents': 0,
    'CoapplicantIncome': 0.0,
    'Loan_Amount_Term': 360.0,
    'ApplicantIncome': 5000.0,
    'Self_Employed': 0,
    'Property_Area': 1
}

# Mapping categorical inputs to numerical
input_data = {
    'Gender': 1 if gender == 'Male' else 0,
    'Married': 1 if married == 'Yes' else 0,
    'Dependents': defaults['Dependents'],
    'Education': 1 if education == 'Graduate' else 0,
    'Self_Employed': defaults['Self_Employed'],
    'ApplicantIncome': defaults['ApplicantIncome'],
    'CoapplicantIncome': defaults['CoapplicantIncome'],
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': defaults['Loan_Amount_Term'],
    'Credit_History': credit_history,
    'Property_Area': defaults['Property_Area']
}

# Prepare input for prediction
input_df = pd.DataFrame([[input_data[feature] for feature in feature_order]], columns=feature_order)

# Predict button
if st.button("🔍 Predict Loan Approval"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]  # Probability for approval

    st.subheader("📊 Prediction Result")
    st.write(f"### {'✅ Loan Approved' if prediction == 1 else '❌ Loan Not Approved'}")
    st.write(f"Confidence: **{proba*100:.2f}%**")
