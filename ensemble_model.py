import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

# Load models
gradient_boosting = joblib.load('gradientboosting_model.pkl')
svm = joblib.load('svm_model.pkl')

# Ensemble model setup
ensemble = VotingClassifier(
    estimators=[
        ('GradientBoosting', gradient_boosting),
        ('SVM', svm)
    ],
    voting='soft'
)

# Load test data
data = pd.read_csv('preprocessed_data.csv')
X = data.drop('Overall Quality', axis=1)
y = data['Overall Quality']

# Fit ensemble model and evaluate accuracy
ensemble.fit(X, y)
y_pred = ensemble.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Ensemble model accuracy: {accuracy:.4f}")

# Save ensemble model
joblib.dump(ensemble, 'ensemble_model.pkl')
print("Ensemble model saved.")
