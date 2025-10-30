# (D:\Udemy\Complete_DSMLDLNLP_Bootcamp\UPractice1\venv) D:\Udemy\Complete_DSMLDLNLP_Bootcamp\UPractice2\1-Simple_Linear_Reg>streamlit run 2-.py

import streamlit as st
import pickle
import pandas as pd

with open('model.pkl', 'rb') as file:
    linear_model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("Height Prediction 🧍↕📏")
st.text("Welcome!! 🤝 This is an Height Prediction application. You just have to enter your weight to get your height predicted.")

# Test prediction
new_weight = st.text_input("Please enter your weight (in kg)")

if new_weight:
    new_weight = float(new_weight)
    new_weight_df = pd.DataFrame([new_weight], columns=['Weight'])
    new_weight_df = scaler.transform(new_weight_df)

    predicted_height = linear_model.predict(new_weight_df)
    st.success(f"Predicted Height: {predicted_height[0][0]:.2f} cm")
    #st.write(predicted_height[0][0])
