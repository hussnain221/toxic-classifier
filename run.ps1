$ErrorActionPreference = "Stop"

try {
  . .\venv\Scripts\Activate.ps1
  python app.py
} catch {
  Write-Host "Run failed: $($_.Exception.Message)" -ForegroundColor Red
  exit 1
}
