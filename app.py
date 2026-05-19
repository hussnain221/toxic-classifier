import json
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import joblib

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

MODEL_PATH = Path(__file__).parent / "model" / "classifier.pkl"


class MockPredictor:
    def predict_proba(self, texts):
        probs = []
        for text in texts:
            seed = abs(hash(text)) % (2**32)
            rng = np.random.default_rng(seed)
            probs.append(rng.random(len(LABELS)))
        return np.vstack(probs)


def load_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH), True
        except Exception:
            pass
    return MockPredictor(), False


app = Flask(__name__)
CORS(app)

model, model_loaded = load_model()


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": bool(model_loaded)})


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify(
        {
            "model_type": "Logistic Regression (OneVsRest)",
            "vectorizer": "TF-IDF (max_features=5000)",
            "labels": LABELS,
            "training_samples": "~21000",
            "dataset": "Jigsaw Toxic Comment Classification Challenge (Kaggle 2018)",
        }
    )


@app.route("/api/classify", methods=["POST"])
def classify():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing or empty 'text' field."}), 400

    proba = np.asarray(model.predict_proba([text]))
    if proba.ndim != 2 or proba.shape[1] != len(LABELS):
        return jsonify({"error": "Model returned unexpected shape."}), 500

    scores = [float(p) for p in proba[0]]
    predictions = dict(zip(LABELS, scores))
    is_toxic = any(score >= 0.5 for score in scores)
    top_label = max(predictions, key=predictions.get)

    return jsonify(
        {
            "predictions": predictions,
            "is_toxic": bool(is_toxic),
            "top_label": top_label,
        }
    )


@app.route("/", methods=["GET"])
def index():
    template_path = Path(app.root_path) / "templates" / "index.html"
    if template_path.exists():
        return send_from_directory(template_path.parent, "index.html")
    return (
        "<h1>Toxic Classifier API</h1><p>Index template missing.</p>",
        200,
        {"Content-Type": "text/html"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
