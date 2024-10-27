# train_models.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import joblib

def train_rf(X_train, y_train):
    rf = RandomForestClassifier()
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    search = RandomizedSearchCV(rf, param_grid, n_iter=10, cv=3, n_jobs=-1)
    search.fit(X_train, y_train)
    best_rf = search.best_estimator_
    joblib.dump(best_rf, 'rf_model.pkl')
    return best_rf

def train_gb(X_train, y_train):
    gb = GradientBoostingClassifier()
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    }
    search = RandomizedSearchCV(gb, param_grid, n_iter=10, cv=3, n_jobs=-1)
    search.fit(X_train, y_train)
    best_gb = search.best_estimator_
    joblib.dump(best_gb, 'gb_model.pkl')
    return best_gb

def train_svm(X_train, y_train):
    svm = SVC(probability=True)
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    search = RandomizedSearchCV(svm, param_grid, n_iter=5, cv=3, n_jobs=-1)
    search.fit(X_train, y_train)
    best_svm = search.best_estimator_
    joblib.dump(best_svm, 'svm_model.pkl')
    return best_svm

data = pd.read_csv('processed_data.csv')
X = data.drop('Overall Quality', axis=1)
y = data['Overall Quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_rf = train_rf(X_train, y_train)
best_gb = train_gb(X_train, y_train)
best_svm = train_svm(X_train, y_train)
