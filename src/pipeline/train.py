import os
from src.preprocess import load_data
from src.models.semantic import SemanticModel
from src.models.regression import RegressionModel

def run():
    print("Cargando datos...")

    os.makedirs("models", exist_ok=True)

    df = load_data("data/Posts.xml", limit=2000)

    print("Entrenando modelo semántico...")
    texts = df["BodyText"].tolist()

    sem = SemanticModel()
    sem.fit(texts)
    sem.save()

    print("Entrenando modelo de regresión...")

    X = df.drop(["Score", "BodyText"], axis=1)
    y = df["Score"]

    reg = RegressionModel()
    reg.fit(X, y)
    reg.save()
