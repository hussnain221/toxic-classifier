# Toxic Comment Classifier

Brutalist Flask backend + single-page frontend for scanning toxic comments.

## Screenshot
Add a screenshot of the UI here.

## Tech Stack

| Layer | Tech |
| --- | --- |
| Backend | Flask, scikit-learn, NLTK |
| Frontend | Vanilla JS, CSS3 |

## Quick Start

Windows:
Right-click setup.ps1 → "Run with PowerShell"
OR in terminal: .\setup.ps1

Mac/Linux:
chmod +x setup.sh && ./setup.sh

## Training

```bash
python model/train.py --data path/to/train.csv
```

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | /api/classify | Classify text. Body: {"text":"..."} |
| GET | /api/health | Service health and model load status |
| GET | /api/stats | Model metadata and dataset info |

## Dataset Credit
Kaggle Jigsaw Toxic Comment Classification Challenge (2018).

## License
MIT
