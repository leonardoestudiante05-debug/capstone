import joblib
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

class SemanticModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.embeddings = None
        self.texts = None

    def fit(self, texts):
        print("Vectorizing texts (TF-IDF)...")

        X = self.vectorizer.fit_transform(texts)
        self.embeddings = normalize(X.toarray())
        self.texts = texts

    def search(self, query, top_k=5):
        query_vec = self.vectorizer.transform([query])
        query_vec = normalize(query_vec.toarray())

        sims = np.dot(self.embeddings, query_vec.T).flatten()
        top_idx = np.argsort(sims)[-top_k:][::-1]

        return [(self.texts[i], sims[i]) for i in top_idx]

    def save(self, path="models/semantic"):
        os.makedirs("models", exist_ok=True)

        joblib.dump(self.vectorizer, f"{path}_vectorizer.pkl")
        np.save(f"{path}_embeddings.npy", self.embeddings)
        joblib.dump(self.texts, f"{path}_texts.pkl")

    def load(self, path="models/semantic"):
        self.vectorizer = joblib.load(f"{path}_vectorizer.pkl")
        self.embeddings = np.load(f"{path}_embeddings.npy")
        self.texts = joblib.load(f"{path}_texts.pkl")
        