# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd  # Import pandas to use DataFrame

# Load the encoder and model
encoder = joblib.load('product_type_encoder.pkl')
model = joblib.load('voting_model.pkl')

st.title("Agricultural Quality Prediction")
st.write("Enter the product features to predict quality.")

# Product type selection
product_type = st.selectbox("Product Type", options=encoder.categories_[0])

# Initialize type_encoded
type_encoded = np.array([])

# Encode the product type using a DataFrame
try:
    # Create a DataFrame to pass to the encoder
    product_type_df = pd.DataFrame([[product_type]], columns=["Product Type"])
    type_encoded = encoder.transform(product_type_df).flatten()
except Exception as e:
    st.error(f"Error transforming product type: {e}")

# Gather numerical inputs
nutrient_content = st.slider("Nutrient Content", 0, 10, 5)
residual_content = st.number_input("Residual Content", 0.0, 1.0, 0.5)
shelf_life = st.slider("Shelf Life (days)", 1, 100, 30)
batch_harvest = st.slider("Batch Harvest", 1, 10, 5)

# Ensure type_encoded is valid before concatenation
if type_encoded.size > 0:
    # Combine inputs into a single array
    input_data = np.concatenate((type_encoded, [nutrient_content, residual_content, shelf_life, batch_harvest])).reshape(1, -1)

    # Prediction button
    if st.button("Predict Quality"):
        prediction = model.predict(input_data)
        st.success(f"Predicted Quality: {'High' if prediction == 1 else 'Medium' if prediction == 2 else 'Low'}")
else:
    st.warning("Please select a valid product type.")
