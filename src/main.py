from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os

from src.models.semantic import SemanticModel
from src.models.regression import RegressionModel
from src.pipeline.train import run as train_models


if not os.path.exists("models/semantic_embeddings.npy") or not os.path.exists("models/regression.pkl"):
    print("Modelos no encontrados, entrenando...")
    train_models()

print("Cargando modeloos...")

semantic_model = SemanticModel()
semantic_model.load()

regression_model = RegressionModel()
regression_model.load()


app = FastAPI(title="Capstone ML API")

class SearchRequest(BaseModel):
    query: str

class PredictRequest(BaseModel):
    ViewCount: int
    AnswerCount: int
    CommentCount: int
    FavoriteCount: int
    title_length: int
    body_length: int
    word_count: int
    num_tags: int

# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"message": "API funcionando 🚀"}

@app.post("/search")
def search(req: SearchRequest):
    results = semantic_model.search(req.query)

    return {
        "results": [
            {
                "text": text[:200],
                "score": float(score)
            }
            for text, score in results
        ]
    }

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([req.dict()])
    pred = regression_model.predict(df)

    return {
        "predicted_score": float(pred[0])
    }
