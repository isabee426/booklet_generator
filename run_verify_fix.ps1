# Specific puzzle runner to verify fixes
$env:PYTHONUNBUFFERED = "1"
& "c:/Users/Isabe/New folder (3)/saturn-arc/arc-solver-env/Scripts/Activate.ps1"
cd "C:\Users\Isabe\New folder (3)\saturn-arc"

$outputDir = "..\booklets_ARCAGI\traces\batch_verify_fixes_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$puzzles = @("ddf7fa4f", "2753e76c")
$trainingDir = "ARC-AGI-2\ARC-AGI-2\data\training"

foreach ($puzzleId in $puzzles) {
    $filePath = Join-Path $trainingDir "$puzzleId.json"
    Write-Host "`nRunning puzzle: $puzzleId"
    python arc_comprehensive_solver_v10.py "$filePath" --output_dir "$outputDir"
    Start-Sleep -Seconds 2
}

