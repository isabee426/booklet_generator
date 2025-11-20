# Batch runner for ARC Comprehensive Solver V10
# Runs multiple puzzles end-to-end

$env:PYTHONUNBUFFERED = "1"

# Activate virtual environment
& "c:/Users/Isabe/New folder (3)/saturn-arc/arc-solver-env/Scripts/Activate.ps1"

# Change to solver directory
cd "C:\Users\Isabe\New folder (3)\saturn-arc"

# Output directory for results
$outputDir = "..\booklets_ARCAGI\traces\batch_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

# Get list of training files (exclude already run ones)
$alreadyRun = @("00d62c1b", "5b526a93", "f341894c", "00576224", "007bbfb7", "009d5c81", "00dbd492")
$trainingDir = "ARC-AGI-2\ARC-AGI-2\data\training"
$allFiles = Get-ChildItem -Path $trainingDir -Filter "*.json"

# Filter out already run puzzles
$availableFiles = $allFiles | Where-Object {
    $puzzleId = $_.BaseName
    $notRun = $true
    foreach ($runId in $alreadyRun) {
        if ($puzzleId -like "*$runId*") {
            $notRun = $false
            break
        }
    }
    $notRun
}

# Randomly select 5 puzzles
$random = New-Object System.Random
$filesToRun = $availableFiles | Sort-Object { $random.Next() } | Select-Object -First 5

Write-Host "Found $($filesToRun.Count) puzzles to run"
Write-Host "Output directory: $outputDir"

$successCount = 0
$failCount = 0

foreach ($file in $filesToRun) {
    $puzzleId = $file.BaseName
    $filePath = $file.FullName
    $outputFile = Join-Path $outputDir "$puzzleId`_v10_analysis.json"
    
    Write-Host "`n========================================"
    Write-Host "Running puzzle: $puzzleId"
    Write-Host "File: $filePath"
    Write-Host "Output: $outputFile"
    Write-Host "========================================`n"
    
    try {
        python arc_comprehensive_solver_v10.py "`"$filePath`"" --output_dir "`"$outputDir`""
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Success: $puzzleId" -ForegroundColor Green
            $successCount++
        } else {
            Write-Host "❌ Failed: $puzzleId (exit code: $LASTEXITCODE)" -ForegroundColor Red
            $failCount++
        }
    } catch {
        Write-Host "❌ Error running $puzzleId : $_" -ForegroundColor Red
        $failCount++
    }
    
    Write-Host "`nWaiting 2 seconds before next puzzle...`n"
    Start-Sleep -Seconds 2
}

Write-Host "`n========================================"
Write-Host "Batch run complete!"
Write-Host "Success: $successCount"
Write-Host "Failed: $failCount"
Write-Host "Total: $($successCount + $failCount)"
Write-Host "Output directory: $outputDir"
Write-Host "========================================"

