import os

os.environ["KERAS_BACKEND"] = "torch"

import streamlit as st
import numpy as np
import pandas as pd
import keras
import joblib

# Load the production objects from the deployment folder
model = keras.models.load_model("deployment/fraud_detection_model.keras")
scaler = joblib.load("deployment/robust_scaler.pkl")
selector = joblib.load("deployment/feature_selector.pkl")

st.title("Credit Card Fraud Detection System")
st.write("Enter the transaction details below:")

# Input fields for V1 to V28 and Amount
input_data = {}
for i in range(1, 29):
    input_data[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

input_data["Amount"] = st.number_input("Amount", value=0.0)

if st.button("Predict"):
    # Convert input to DataFrame
    df_input = pd.DataFrame([input_data])

    # Preprocess: Scale and then Select Features
    scaled_input = scaler.transform(df_input)
    df_scaled = pd.DataFrame(scaled_input, columns=df_input.columns)
    selected_input = selector.transform(df_scaled)

    # Predict using the ANN model
    prediction = model.predict(selected_input)

    if prediction[0][0] > 0.5:
        st.error("Fraudulent Transaction Detected!")
    else:
        st.success("Legitimate Transaction")
