import argparse
import re
import string
from pathlib import Path
from pprint import pprint

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def download_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def build_preprocessor():
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    url_re = re.compile(r"http[s]?://\S+|www\.\S+")
    punct_re = re.compile(rf"[{re.escape(string.punctuation)}]")

    def preprocess(text):
        if not isinstance(text, str):
            text = ""
        text = text.lower()
        text = url_re.sub(" ", text)
        text = punct_re.sub(" ", text)
        tokens = [tok for tok in text.split() if tok and tok not in stop_words]
        tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
        return " ".join(tokens)

    return preprocess


def load_and_sample(data_path):
    df = pd.read_csv(data_path)
    if "comment_text" not in df.columns:
        raise ValueError("Expected 'comment_text' column in dataset.")
    for label in LABELS:
        if label not in df.columns:
            raise ValueError(f"Expected '{label}' column in dataset.")

    label_matrix = df[LABELS]
    is_toxic = label_matrix.sum(axis=1) > 0
    toxic_rows = df[is_toxic]
    non_toxic_rows = df[~is_toxic]

    non_toxic_sample = non_toxic_rows.sample(
        n=min(5000, len(non_toxic_rows)), random_state=42
    )
    sampled = pd.concat([toxic_rows, non_toxic_sample], axis=0)
    sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)

    return sampled


def train(data_path, output_path):
    download_nltk()
    preprocess = build_preprocessor()
    df = load_and_sample(data_path)

    df["clean_text"] = df["comment_text"].apply(preprocess)
    X = df["clean_text"]
    y = df[LABELS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tfidf = TfidfVectorizer(max_features=5000)
    model = OneVsRestClassifier(
        LogisticRegression(class_weight="balanced", max_iter=1000)
    )
    pipeline = Pipeline([("tfidf", tfidf), ("clf", model)])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=LABELS, zero_division=0))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )

    metrics = {}
    for idx, label in enumerate(LABELS):
        metrics[label] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "accuracy": float(accuracy_score(y_test[label], y_pred[:, idx])),
        }

    pprint(metrics)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"Saved model to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train toxic comment classifier")
    parser.add_argument(
        "--data", default="train.csv", help="Path to training CSV file"
    )
    parser.add_argument(
        "--output", default="model/classifier.pkl", help="Path to save model"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(Path(args.data), Path(args.output))
