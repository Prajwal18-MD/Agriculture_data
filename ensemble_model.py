import joblib
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Load processed data
data = pd.read_csv('processed_data.csv')
X = data.drop('Overall Quality', axis=1)
y = data['Overall Quality']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load trained models
rf = joblib.load('rf_model.pkl')
gb = joblib.load('gb_model.pkl')
svm = joblib.load('svm_model.pkl')

# Stacking Classifier
stacked_model = StackingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
    final_estimator=LogisticRegression()
)
stacked_model.fit(X_train, y_train)
joblib.dump(stacked_model, 'stacked_model.pkl')

# Voting Classifier Ensemble
voting_model = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('svm', svm), ('stacked', stacked_model)],
    voting='soft'  # Use 'soft' voting if all models support probability prediction
)
voting_model.fit(X_train, y_train)
joblib.dump(voting_model, 'voting_model.pkl')
