
import pandas as pd
import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Customer Churn Predication")

st.divider()

st.write("PLease enter the values and hit the predict button for getting a predication")

st.divider()

Age = st.number_input("Enter Age:",min_value=10,max_value=100,value=30)

Gender = st.selectbox("Enter the Gender:",["Male","Female"])

if Gender:
    Gender_selected = 1 if Gender == "Female" else 0

Tenure = st.number_input("Enter Tenure:",min_value=0,max_value=130,value=10)

MonthlyCharges = st.number_input("Enter MonthlyCharges:",min_value=30,max_value=150)

st.divider()

Predictbutton = st.button("Predict!")

if Predictbutton:
    Gender_selected = 1 if Gender == "Female" else 0

    X = [Age, Gender_selected, Tenure, MonthlyCharges]

    X1 = np.array(X)

    X_array = scaler.transform([X1])

    Predication = model.predict(X_array)[0]
    
    Predicted = "Yes" if Predication == 1 else "No"

    st.balloons()

    st.write(f"Predicted : {Predicted}")


else:
    st.write("Please enter the values and use Predict botton")