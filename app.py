from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import joblib

from model.text_processing import build_preprocessor

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

MODEL_PATH = Path(__file__).parent / "model" / "classifier.pkl"


def load_model():
    if not MODEL_PATH.exists():
        return None, f"Model file not found at {MODEL_PATH}"
    try:
        return joblib.load(MODEL_PATH), None
    except Exception as exc:
        return None, f"Failed to load model: {exc}"


app = Flask(__name__)
CORS(app)

model, model_error = load_model()
preprocess = build_preprocessor()


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok" if model is not None else "degraded",
            "model_loaded": model is not None,
            "model_error": model_error,
        }
    )


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
    if model is None:
        return jsonify({"error": "Model is unavailable.", "detail": model_error}), 503

    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing or empty 'text' field."}), 400

    clean_text = preprocess(text)
    proba = np.asarray(model.predict_proba([clean_text]))
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
