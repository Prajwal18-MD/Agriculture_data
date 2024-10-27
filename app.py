
# Import necessary libraries
import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Load the model and encoders
model = joblib.load('knn_model.pkl')

# Initialize encoders and scaler
le_type = LabelEncoder()
le_type.fit(['Fruit', 'Vegetable'])  # Manually fit categories
scaler = StandardScaler()

# App title and description
st.set_page_config(page_title="Quality Predictor", layout="centered", initial_sidebar_state="collapsed")
st.title("Agricultural Quality Predictor")
st.markdown("### Determine the quality of agricultural products with ease and accuracy.")

# Input fields for features
st.markdown("#### Input the characteristics of the product:")

product_type = st.selectbox("Product Type", options=['Fruit', 'Vegetable'])
nutrient_content = st.slider("Nutrient Content", min_value=0, max_value=10, value=5, step=1)
residual_content = st.number_input("Residual Content", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
shelf_life = st.slider("Shelf Life (days)", min_value=1, max_value=100, value=30, step=1)
batch_harvest = st.slider("Batch of Harvesting", min_value=1, max_value=10, value=5, step=1)

# Encoding and scaling input
encoded_type = le_type.transform([product_type])[0]
features = np.array([[encoded_type, nutrient_content, residual_content, shelf_life, batch_harvest]])
scaled_features = scaler.fit_transform(features)  # Scale the features

# Prediction
if st.button("Predict Quality"):
    prediction = model.predict(scaled_features)
    quality_label = "High" if prediction[0] == 1 else "Medium" if prediction[0] == 2 else "Low"
    
    # Display results
    st.success(f"The predicted quality is **{quality_label}**.")

# Footer
st.markdown("---")
st.markdown("Designed for optimal usability on both mobile and desktop screens.")
