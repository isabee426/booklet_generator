# Run ARC Puzzle Creator - Flask Version
# PowerShell script

Write-Host "🧩 Starting ARC Puzzle Creator (Flask)..." -ForegroundColor Cyan
Write-Host ""

# Check if Flask is installed
$flaskInstalled = pip list | Select-String "Flask"

if (-not $flaskInstalled) {
    Write-Host "⚠️  Flask not found. Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements_flask.txt
    Write-Host ""
}

# Check if ARC data exists
$arcDataPath = "..\saturn-arc\ARC-AGI-2\ARC-AGI-1\data\training"
if (-not (Test-Path $arcDataPath)) {
    Write-Host "⚠️  Warning: ARC data not found at expected location:" -ForegroundColor Yellow
    Write-Host "   $arcDataPath" -ForegroundColor Yellow
    Write-Host "   The app may not work properly without the data." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "🚀 Launching Flask web server..." -ForegroundColor Green
Write-Host "   The app will open at http://localhost:5000" -ForegroundColor Gray
Write-Host "   Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

# Run Flask app
python app.py

