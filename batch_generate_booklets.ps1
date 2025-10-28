# Batch Booklet Generator Script
# Generates booklets for easy puzzles

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "ARC Booklet Generator - Batch Mode" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Check for API key
if (-not $env:OPENAI_API_KEY) {
    Write-Host "ERROR: OPENAI_API_KEY not set!" -ForegroundColor Red
    Write-Host "Set it with: `$env:OPENAI_API_KEY='your-key'" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ API Key found" -ForegroundColor Green
Write-Host ""

# Check we're in right directory
if (-not (Test-Path "arc-booklet-generator.py")) {
    Write-Host "ERROR: Must run from booklets_ARCAGI directory!" -ForegroundColor Red
    Write-Host "Current directory: $PWD" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ In correct directory" -ForegroundColor Green
Write-Host ""

# Run batch generation
Write-Host "Starting batch generation..." -ForegroundColor Cyan
Write-Host ""

python batch_generate_booklets.py

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "DONE! View booklets with:" -ForegroundColor Green
Write-Host "  streamlit run streamlit_booklet_viewer.py" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Cyan

