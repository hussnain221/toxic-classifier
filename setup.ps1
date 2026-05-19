$ErrorActionPreference = "Stop"

function Fail($message) {
  Write-Host $message -ForegroundColor Red
  exit 1
}

try {
  Write-Host "[1/8] Checking for Python..."
  $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
  if (-not $pythonCmd) {
    Write-Host "Python not found. Install from https://python.org"
    exit 1
  }

  Write-Host "[2/8] Creating virtual environment..."
  python -m venv venv
  if ($LASTEXITCODE -ne 0) {
    Fail "Failed to create virtual environment. Try deleting the venv folder and rerun."
  }

  Write-Host "[3/8] Activating virtual environment..."
  try {
    . .\venv\Scripts\Activate.ps1
  } catch {
    Write-Host "Activation failed. Adjusting execution policy..." -ForegroundColor Yellow
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    try {
      . .\venv\Scripts\Activate.ps1
    } catch {
      Fail "Failed to activate virtual environment. Open PowerShell as CurrentUser and try again."
    }
  }

  Write-Host "[4/8] Installing dependencies..."
  pip install -r requirements.txt
  if ($LASTEXITCODE -ne 0) {
    Fail "Dependency installation failed. Check requirements.txt and your internet connection."
  }
  pip install flask-cors
  if ($LASTEXITCODE -ne 0) {
    Fail "Failed to install flask-cors. Try running 'pip install flask-cors' manually."
  }

  Write-Host "[5/8] Downloading NLTK data..."
  python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('punkt_tab'); nltk.download('omw-1.4')"
  if ($LASTEXITCODE -ne 0) {
    Fail "NLTK download failed. Check your internet connection and try again."
  }

  Write-Host "[6/8] Checking for train.csv..."
  if (Test-Path ".\train.csv") {
    Write-Host "[4/6] Training model... this takes 3-8 minutes, please wait"
    python model/train.py --data train.csv
    if ($LASTEXITCODE -ne 0) {
      Fail "Model training failed. Verify train.csv format and try again."
    }
  } else {
    Write-Host "WARNING: train.csv not found. Skipping model training. App will use mock predictions. Download train.csv from https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data and re-run this script." -ForegroundColor Yellow
  }

  Write-Host "[7/8] Checking port 5000..."
  $line = netstat -ano | Select-String ":5000" | Select-Object -First 1
  if ($line) {
    $pid = $line.ToString().Trim().Split()[-1]
    if ($pid -and $pid -ne "0") {
      Stop-Process -Id $pid -Force
    }
  }

  Write-Host "[8/8] Starting Flask app..."
  $appProcess = Start-Process -FilePath "python" -ArgumentList "app.py" -WorkingDirectory $PSScriptRoot -PassThru
  if (-not $appProcess) {
    Fail "Failed to start Flask app. Check app.py output for details."
  }
  Start-Process "http://localhost:5000"
  Write-Host "App running at http://localhost:5000"
  Write-Host ([char]0x2705 + " SETUP COMPLETE - Open http://localhost:5000 in your browser")
} catch {
  Fail "Setup failed: $($_.Exception.Message)"
}
