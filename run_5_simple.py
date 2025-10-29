#!/usr/bin/env python3
"""
Simple batch runner - Just run 1 puzzle to see if it works
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Just ONE simple puzzle to start
    puzzle = "137f0df0"  # Random puzzle
    
    task_file = Path("..") / "saturn-arc" / "ARC-AGI-2" / "ARC-AGI-2" / "data" / "training" / f"{puzzle}.json"
    
    if not task_file.exists():
        print(f"❌ Puzzle {puzzle} not found at {task_file}")
        sys.exit(1)
    
    print("="*80)
    print(f"RUNNING BATCH VISUAL GENERATOR ON: {puzzle}")
    print("="*80)
    print("\nThis will:")
    print("1. Show AI all training examples")
    print("2. Generate universal steps")
    print("3. Create visual booklets")
    print("4. Test on test cases")
    print("\nWatch the output to see what's happening...\n")
    print("="*80)
    
    # Run the generator
    result = subprocess.run([
        sys.executable,
        "arc-booklet-batch-visual.py",
        str(task_file)
    ])
    
    if result.returncode == 0:
        print("\n✅ DONE!")
        print("\nView booklet in Streamlit:")
        print("  streamlit run streamlit_booklet_viewer.py")
        print(f"\nBooklet saved to: batch_visual_booklets/{puzzle}_batch/")
    else:
        print(f"\n❌ Failed with exit code {result.returncode}")

if __name__ == "__main__":
    main()

