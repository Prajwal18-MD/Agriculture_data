import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the ensemble model and feature names
ensemble_model = joblib.load('ensemble_model.pkl')
feature_names = pd.read_csv('preprocessed_data.csv').drop('Overall Quality', axis=1).columns

# Streamlit app design
st.title("Agricultural Product Quality Prediction")
st.write("Enter the details below to predict the quality of your agricultural product.")

# User inputs
def user_input_features():
    nutrient_content = st.number_input("Nutrient Content", min_value=0.0, max_value=10.0, step=0.1)
    residual_content = st.number_input("Residual Content", min_value=0.0, max_value=1.0, step=0.1)
    shelf_life = st.number_input("Shelf Life (days)", min_value=1, max_value=60, step=1)
    batch_of_harvest = st.number_input("Batch of Harvesting", min_value=1, max_value=10, step=1)
    
    # Product and Type selection (OneHotEncoded as necessary)
    product = st.selectbox("Select Product", ["Apple", "Banana", "Carrot", "Spinach", "Tomato", 
                                              "Grapes", "Broccoli", "Potato", "Strawberry", 
                                              "Pineapple", "Lettuce", "Mango", "Cucumber", 
                                              "Peach", "Zucchini", "Avocado", "Orange", 
                                              "Pepper", "Eggplant", "Cherry", "Watermelon", 
                                              "Cabbage", "Blueberry"])
    prod_type = st.selectbox("Type", ["Fruit", "Vegetable"])

    # Create data structure matching model's encoding
    input_data = [nutrient_content, residual_content, shelf_life, batch_of_harvest]
    for column in feature_names:
        if column == f"Product_{product}":
            input_data.append(1)
        elif column == f"Type_{prod_type}":
            input_data.append(1)
        elif column.startswith("Product_") or column.startswith("Type_"):
            input_data.append(0)

    input_df = pd.DataFrame([input_data], columns=feature_names)
    return input_df

# Make predictions
input_df = user_input_features()

if st.button("Predict Quality"):
    prediction = ensemble_model.predict(input_df)
    st.write(f"Predicted Quality: {'High Quality' if prediction[0] == 1 else 'Low Quality'}")
