import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("attrition_model.pkl", "rb"))

st.title("Employee Attrition Prediction System")
st.write("Fill the employee details below to predict whether the employee may leave the company.")

# Input form
age = st.slider("Age", 18, 60, 25)
distance = st.slider("Distance From Home (KM)", 1, 50, 10)
education = st.selectbox("Education Level", [1,2,3,4,5])
environment = st.selectbox("Environment Satisfaction", [1,2,3,4])
job_satisfaction = st.selectbox("Job Satisfaction", [1,2,3,4])
monthly_income = st.number_input("Monthly Income", 1000, 50000, 5000)
work_life_balance = st.selectbox("Work Life Balance", [1,2,3,4])
years_at_company = st.slider("Years at Company", 0, 40, 3)
overtime = st.selectbox("OverTime", ["Yes", "No"])

# Convert overtime to numeric
overtime = 1 if overtime == "Yes" else 0

# Prepare input
input_data = pd.DataFrame({
    "Age":[age],
    "DistanceFromHome":[distance],
    "Education":[education],
    "EnvironmentSatisfaction":[environment],
    "JobSatisfaction":[job_satisfaction],
    "MonthlyIncome":[monthly_income],
    "WorkLifeBalance":[work_life_balance],
    "YearsAtCompany":[years_at_company],
    "OverTime":[overtime],
})

# Prediction
if st.button("Predict"):
    result = model.predict(input_data)[0]
    if result == 1:
        st.error("⚠ Employee is likely to leave the company (Attrition: YES)")
    else:
        st.success("✔ Employee is NOT likely to leave the company (Attrition: NO)")
