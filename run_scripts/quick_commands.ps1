# Quick Commands for Visual Step Generator
# Simple shortcuts for common operations

param(
    [Parameter(Mandatory=$true, Position=0)]
    [ValidateSet('train', 'test', 'view', 'mark', 'list', 'full', 'train-v2', 'test-v2')]
    [string]$Command,
    
    [Parameter(Mandatory=$false, Position=1)]
    [string]$PuzzleId,
    
    [Parameter(Mandatory=$false)]
    [int]$TestNum = 1,
    
    [Parameter(Mandatory=$false)]
    [int]$TrainingNum
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

function Show-Usage {
    Write-Host ""
    Write-Host "Quick Commands for Visual Step Generator" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Usage: .\quick_commands.ps1 <command> [puzzle-id] [options]" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  train <puzzle-id>              " -NoNewline -ForegroundColor White
    Write-Host "Train on all examples" -ForegroundColor Gray
    Write-Host "  test <puzzle-id>               " -NoNewline -ForegroundColor White
    Write-Host "Solve test example" -ForegroundColor Gray
    Write-Host "  view                           " -NoNewline -ForegroundColor White
    Write-Host "Open Streamlit viewer" -ForegroundColor Gray
    Write-Host "  mark <puzzle-id>               " -NoNewline -ForegroundColor White
    Write-Host "Mark run as successful" -ForegroundColor Gray
    Write-Host "  list                           " -NoNewline -ForegroundColor White
    Write-Host "List all successful runs" -ForegroundColor Gray
    Write-Host "  train-v2 <puzzle-id>          " -NoNewline -ForegroundColor White
    Write-Host "Train on all examples (V2 with expanded actions)" -ForegroundColor Gray
    Write-Host "  test-v2 <puzzle-id>           " -NoNewline -ForegroundColor White
    Write-Host "Solve test example (V2 with expanded actions)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Cyan
    Write-Host "  -TestNum N                     " -NoNewline -ForegroundColor White
    Write-Host "Test example number (default: 1)" -ForegroundColor Gray
    Write-Host "  -TrainingNum N                 " -NoNewline -ForegroundColor White
    Write-Host "Specific training example only" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\quick_commands.ps1 train 05f2a901" -ForegroundColor White
    Write-Host "  .\quick_commands.ps1 test 05f2a901" -ForegroundColor White
    Write-Host "  .\quick_commands.ps1 test 05f2a901 -TestNum 2" -ForegroundColor White
    Write-Host "  .\quick_commands.ps1 view" -ForegroundColor White
    Write-Host "  .\quick_commands.ps1 mark 05f2a901" -ForegroundColor White
    Write-Host "  .\quick_commands.ps1 list" -ForegroundColor White
    Write-Host "  .\quick_commands.ps1 train-v2 05f2a901" -ForegroundColor White
    Write-Host "  .\quick_commands.ps1 test-v2 05f2a901" -ForegroundColor White
    Write-Host ""
}

switch ($Command) {
    'train' {
        if (-not $PuzzleId) {
            Write-Host "Error: Puzzle ID required" -ForegroundColor Red
            Show-Usage
            exit 1
        }
        
        Write-Host ""
        Write-Host "Training on puzzle: $PuzzleId" -ForegroundColor Yellow
        Write-Host "⏸️  Phase 1 generation paused (--skip-phase1)" -ForegroundColor Cyan
        Write-Host ""
        
        if ($TrainingNum) {
            python scripts/visual_step_generator.py --puzzle $PuzzleId --training $TrainingNum --skip-phase1
        } else {
            python scripts/visual_step_generator.py --puzzle $PuzzleId --all --skip-phase1
        }
    }
    
    'test' {
        if (-not $PuzzleId) {
            Write-Host "Error: Puzzle ID required" -ForegroundColor Red
            Show-Usage
            exit 1
        }
        
        Write-Host ""
        Write-Host "Solving test example: $PuzzleId (test $TestNum)" -ForegroundColor Yellow
        Write-Host ""
        
        python scripts/test_step_generator.py --puzzle $PuzzleId --test-num $TestNum
    }
    
    'view' {
        Write-Host ""
        Write-Host "Launching Streamlit viewer..." -ForegroundColor Yellow
        Write-Host "Browser will open at http://localhost:8501" -ForegroundColor Cyan
        Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
        Write-Host ""
        
        streamlit run scripts/view_visual_steps.py
    }
    
    'mark' {
        if (-not $PuzzleId) {
            Write-Host "Error: Puzzle ID required" -ForegroundColor Red
            Show-Usage
            exit 1
        }
        
        Write-Host ""
        Write-Host "Marking $PuzzleId as successful..." -ForegroundColor Yellow
        Write-Host ""
        
        python scripts/mark_successful.py --puzzle $PuzzleId
    }
    
    'list' {
        Write-Host ""
        python scripts/mark_successful.py --list
    }
    
    'full' {
        if (-not $PuzzleId) {
            Write-Host "Error: Puzzle ID required" -ForegroundColor Red
            Show-Usage
            exit 1
        }
        
        Write-Host ""
        Write-Host "Running complete workflow for: $PuzzleId" -ForegroundColor Yellow
        Write-Host ""
        
        & "$scriptDir\run_complete_workflow.ps1" -PuzzleId $PuzzleId -TestNum $TestNum
    }
    
    'train-v2' {
        if (-not $PuzzleId) {
            Write-Host "Error: Puzzle ID required" -ForegroundColor Red
            Show-Usage
            exit 1
        }
        
        Write-Host ""
        Write-Host "Training on puzzle (V2): $PuzzleId" -ForegroundColor Yellow
        Write-Host "⏸️  Phase 1 generation paused (--skip-phase1)" -ForegroundColor Cyan
        Write-Host ""
        
        if ($TrainingNum) {
            python scripts/visual_step_generator_v2.py --puzzle $PuzzleId --training $TrainingNum --skip-phase1
        } else {
            python scripts/visual_step_generator_v2.py --puzzle $PuzzleId --all --skip-phase1
        }
    }
    
    'test-v2' {
        if (-not $PuzzleId) {
            Write-Host "Error: Puzzle ID required" -ForegroundColor Red
            Show-Usage
            exit 1
        }
        
        Write-Host ""
        Write-Host "Solving test example (V2): $PuzzleId (test $TestNum)" -ForegroundColor Yellow
        Write-Host ""
        
        python scripts/test_step_generator.py --puzzle $PuzzleId --test-num $TestNum --v2
    }
    
    default {
        Show-Usage
    }
}


