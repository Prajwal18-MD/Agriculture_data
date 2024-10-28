import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load data
data = pd.read_csv('agriculture_quality_expanded_with_reordered_batch.csv')

# Initial columns
print("Initial columns:", data.columns)

# OneHotEncode 'Product' and 'Type' columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_data = encoder.fit_transform(data[['Product', 'Type']])

# Convert encoded features to DataFrame and concatenate with original data
encoded_columns = encoder.get_feature_names_out(['Product', 'Type'])
encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)
processed_data = pd.concat([data.drop(['Product', 'Type'], axis=1), encoded_df], axis=1)

# Columns after encoding
print("Columns after encoding:", processed_data.columns)

# Save the preprocessed data
processed_data.to_csv('preprocessed_data.csv', index=False)
