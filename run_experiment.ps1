# Run the canonical compare_runs experiment in a reproducible way (PowerShell)
# Usage: From repo root, run: .\run_experiment.ps1

# Activate venv
if (Test-Path .\.venv\Scripts\Activate.ps1) {
    Write-Host "Activating virtual environment..."
    . .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "No .venv detected. Create one with: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

# Ensure requirements are installed
pip install -r requirements.txt

# Run experiment
python experiments/compare_runs.py

# Show saved results
$path = Join-Path (Get-Location) "results\compare_runs.png"
if (Test-Path $path) {
    Write-Host "Saved plot at: $path" -ForegroundColor Green
} else {
    Write-Host "No plot was saved. Check experiments/compare_runs.py output." -ForegroundColor Yellow
}
