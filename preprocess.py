import pandas as pd
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
import joblib

def preprocess_data(path):
    # Load data
    data = pd.read_csv(path)
    print("Initial columns:", data.columns)  # Debugging step to check columns

    # Separate target variable
    target = data['Overall Quality']
    data = data.drop('Overall Quality', axis=1)
    
    # Encode categorical columns (excluding target variable)
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    categorical_cols = data.select_dtypes(include=['object']).columns  # Identify categorical columns
    
    # Apply OneHotEncoder to categorical columns
    encoded_data = encoder.fit_transform(data[categorical_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Save the encoder
    joblib.dump(encoder, 'product_type_encoder.pkl')
    
    # Drop original categorical columns and append encoded data
    data = data.drop(categorical_cols, axis=1)
    data = pd.concat([data, encoded_df], axis=1)
    print("Columns after encoding:", data.columns)  # Debugging step to check columns

    # Generate polynomial features for numerical data
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(data)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=5)
    reduced_features = pca.fit_transform(poly_features)
    processed_data = pd.DataFrame(reduced_features, columns=[f'pca_{i}' for i in range(5)])
    
    # Add 'Overall Quality' back to processed data as target
    processed_data['Overall Quality'] = target.values
    
    # Save processed data to a CSV
    processed_data.to_csv('processed_data.csv', index=False)
    return processed_data

# Run the function
processed_data = preprocess_data('agriculture_quality_expanded_with_reordered_batch.csv')
