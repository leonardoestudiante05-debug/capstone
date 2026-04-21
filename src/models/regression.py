import joblib
import os
from sklearn.ensemble import RandomForestRegressor

class RegressionModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=200, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path="models/regression.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path="models/regression.pkl"):
        self.model = joblib.load(path)
