import streamlit as st
import numpy as np
import pandas as pd
import keras
import joblib

model = keras.models.load_model("fraud_detection_model.keras")
scaler = joblib.load("robust_scaler.pkl")
selector = joblib.load("feature_selector.pkl")

st.title("Credit Card Fraud Detection System")

st.write("Enter the transaction details below:")

input_data = {}
for i in range(1, 29):
    input_data[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

input_data["Amount"] = st.number_input("Amount", value=0.0)

if st.button("Predict"):
    df_input = pd.DataFrame([input_data])

    scaled_input = scaler.transform(df_input)
    df_scaled = pd.DataFrame(scaled_input, columns=df_input.columns)

    selected_input = selector.transform(df_scaled)

    prediction = model.predict(selected_input)

    if prediction[0][0] > 0.5:
        st.error("Fraudulent Transaction Detected!")
    else:
        st.success("Legitimate Transaction")
