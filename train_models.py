import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load processed data
data = pd.read_csv('preprocessed_data.csv')
X = data.drop('Overall Quality', axis=1)
y = data['Overall Quality']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'GradientBoosting': GradientBoostingClassifier(),
    'SVM': SVC(probability=True)
}

# Train and save models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f'{model_name.lower()}_model.pkl')
    print(f"{model_name} trained and saved.")

# Check individual model accuracy
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
