import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import base64

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="House Rent Prediction",
    page_icon="🏠",
    layout="centered"
)

# -----------------------------
# Background Image Function
# -----------------------------
# Absolute base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def add_bg_from_local(image_name):
    image_path = os.path.join(BASE_DIR, image_name)

    with open(image_path, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Call background safely
add_bg_from_local("house.jpg")



# -----------------------------
# Load Model, Scaler & Features
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RF_model = pickle.load(open(os.path.join(BASE_DIR, "RF_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
feature_columns = pickle.load(open(os.path.join(BASE_DIR, "feature_columns.pkl"), "rb"))

# -----------------------------
# App Title
# -----------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:white; text-shadow:2px 2px 5px black;'>
    🏠 House Rent Prediction
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align:center; color:white;'>Predict Hyderabad House Rent using Machine Learning</h4>",
    unsafe_allow_html=True
)

st.markdown("---")

# -----------------------------
# User Inputs
# -----------------------------
st.markdown("<h3 style='color:white;'>Enter House Details</h3>", unsafe_allow_html=True)

area = st.number_input("Area (sq ft)", min_value=200, max_value=10000, value=1400)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
washrooms = st.number_input("Washrooms", min_value=1, max_value=10, value=2)
tennants = st.number_input("Tennants", min_value=1, max_value=10, value=4)

# -----------------------------
# Create Input DataFrame
# -----------------------------
input_data = pd.DataFrame({
    "Area": [area],
    "Bedrooms": [bedrooms],
    "Washrooms": [washrooms],
    "Tennants": [tennants]
})

# -----------------------------
# One-Hot Encoding & Alignment
# -----------------------------
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=feature_columns, fill_value=0)

# -----------------------------
# Scaling
# -----------------------------
input_scaled = scaler.transform(input_data)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔮 Predict Rent"):
    prediction = RF_model.predict(input_scaled)[0]

    st.markdown(
        f"""
        <div style="
            background: rgba(0,0,0,0.7);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-top: 30px;">
            <h1 style="color:#00ffcc;">💰 ₹ {int(prediction):,}</h1>
            <h3 style="color:white;">Estimated Monthly House Rent</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:white;'>Built using Random Forest Regression & Streamlit 🚀</p>",
    unsafe_allow_html=True
)
