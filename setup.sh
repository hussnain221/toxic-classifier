#!/usr/bin/env bash
set -euo pipefail

RED="\033[0;31m"
YELLOW="\033[0;33m"
NC="\033[0m"

fail() {
  echo -e "${RED}$1${NC}"
  exit 1
}

trap 'fail "Unexpected error on line $LINENO. Check the output above."' ERR

echo "[1/8] Checking for Python..."
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python not found. Install from https://python.org"
  exit 1
fi

echo "[2/8] Creating virtual environment..."
python3 -m venv venv || fail "Failed to create virtual environment. Try deleting the venv folder and rerun."

echo "[3/8] Activating virtual environment..."
source venv/bin/activate || fail "Failed to activate virtual environment. Try running: source venv/bin/activate"

echo "[4/8] Installing dependencies..."
pip install -r requirements.txt || fail "Dependency installation failed. Check requirements.txt and your internet connection."
pip install flask-cors || fail "Failed to install flask-cors. Try running: pip install flask-cors"

echo "[5/8] Downloading NLTK data..."
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('punkt_tab'); nltk.download('omw-1.4')" || fail "NLTK download failed. Check your internet connection and try again."

echo "[6/8] Checking for train.csv..."
if [ -f "train.csv" ]; then
  echo "[4/6] Training model... this takes 3-8 minutes, please wait"
  python model/train.py --data train.csv || fail "Model training failed. Verify train.csv format and try again."
else
  echo -e "${YELLOW}WARNING: train.csv not found. Skipping model training. App will use mock predictions. Download train.csv from https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data and re-run this script.${NC}"
fi

echo "[7/8] Checking port 5000..."
PIDS=$(lsof -ti:5000 2>/dev/null || true)
if [ -n "$PIDS" ]; then
  echo "Port 5000 in use. Stopping process..."
  echo "$PIDS" | xargs kill -9 || fail "Failed to stop process on port 5000."
fi

echo "[8/8] Starting Flask app..."
python app.py &
APP_PID=$!
if ! kill -0 "$APP_PID" 2>/dev/null; then
  fail "Failed to start Flask app. Check app.py output for details."
fi
if command -v open >/dev/null 2>&1; then
  open "http://localhost:5000"
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "http://localhost:5000"
fi

echo "App running at http://localhost:5000"
echo "SETUP COMPLETE - Open http://localhost:5000 in your browser"
