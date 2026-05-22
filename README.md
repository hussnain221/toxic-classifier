# Toxic Comment Classifier

Brutalist Flask backend + single-page frontend for scanning toxic comments.

This project classifies user-submitted text across six toxicity labels using a
TF-IDF + One-vs-Rest Logistic Regression model trained on the Jigsaw Toxic
Comment Classification dataset.

## Tech Stack

| Layer | Tech |
| --- | --- |
| Backend | Flask, scikit-learn, NLTK |
| Frontend | Vanilla JS, CSS3 |
| Model | TF-IDF vectorizer + OneVsRest Logistic Regression |
| Dataset | Kaggle Jigsaw Toxic Comment Classification Challenge |

## Features

- Multi-label toxicity prediction for six classes.
- Animated confidence bars and clean/toxic verdict.
- Local scan history in the browser.
- Health endpoint that reports whether the real model is loaded.
- Shared text preprocessing between training and inference.

## Quick Start

Windows:
Right-click setup.ps1 -> "Run with PowerShell"
OR in terminal: .\setup.ps1

Mac/Linux:
chmod +x setup.sh && ./setup.sh

Manual run:

```bash
python -m venv venv
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000

## Training

```bash
python model/train.py --data path/to/train.csv
```

The training script samples all toxic comments and up to 5,000 non-toxic
comments, applies the same text cleaner used by the API, trains the model, and
saves it to `model/classifier.pkl`.

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | /api/classify | Classify text. Body: `{"text":"..."}` |
| GET | /api/health | Service health and model load status |
| GET | /api/stats | Model metadata and dataset info |

Example:

```bash
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"You are kind and helpful.\"}"
```

## Project Structure

```text
app.py                   Flask API and frontend server
model/train.py           Training script
model/text_processing.py Shared text cleaner
model/classifier.pkl     Saved classifier
templates/index.html     Single-page app
static/style.css         UI styling
tests/                   API regression tests
```

## Dataset Credit

Kaggle Jigsaw Toxic Comment Classification Challenge (2018).

## License

MIT
