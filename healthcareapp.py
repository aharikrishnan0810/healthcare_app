import streamlit as st
import joblib
import numpy as np
import base64
import os

# Function to load and encode the local image to base64
def load_image(image_file):
    with open(image_file, "rb") as image:
        return base64.b64encode(image.read()).decode()

# Load your local background image
bg_image_path = r'C:/Users/ahari/OneDrive/Desktop/DL/db1.jpg'  # Updated with your image path
if os.path.exists(bg_image_path):
    bg_image = load_image(bg_image_path)
else:
    st.error("Background image not found. Please check the file path.")
    bg_image = None

# Set the custom CSS for the background image (if found)
if bg_image:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bg_image}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load the trained Random Forest model and scaler
model_filename = 'random_forest_diabetes_model.pkl'
scaler_filename = 'scaler.pkl'
model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# Streamlit interface content
st.title("Diabetes Prediction App")

# Create input fields for the features with bold labels
pregnancies = st.number_input("**Number of Pregnancies**", min_value=0, max_value=20)
glucose = st.number_input("**Glucose Level**", min_value=0.0, max_value=200.0)
blood_pressure = st.number_input("**Blood Pressure (mm Hg)**", min_value=0.0, max_value=200.0)
skin_thickness = st.number_input("**Skin Thickness (mm)**", min_value=0.0, max_value=100.0)
insulin = st.number_input("**Insulin Level**", min_value=0.0, max_value=1000.0)
bmi = st.number_input("**Body Mass Index (BMI)**", min_value=0.0, max_value=50.0)
diabetes_pedigree = st.number_input("**Diabetes Pedigree Function**", min_value=0.0, max_value=3.0)
age = st.number_input("**Age**", min_value=0, max_value=120)

# Button for prediction
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)

    if prediction[0] == 1:
        st.success("The model predicts: **Diabetes**")
    else:
        st.success("The model predicts: **No Diabetes**")
