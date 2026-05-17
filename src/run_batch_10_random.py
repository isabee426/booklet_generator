#!/usr/bin/env python3
"""
Batch runner - Run 10 randomly selected ARC-AGI-2 training tasks
"""

import subprocess
import sys
from pathlib import Path
import random
import json

def main():
    # Path to ARC-AGI-2 training data
    training_dir = Path("..") / "saturn-arc" / "ARC-AGI-2" / "ARC-AGI-2" / "data" / "training"
    
    if not training_dir.exists():
        print(f"❌ Training directory not found: {training_dir}")
        sys.exit(1)
    
    # Get all task files
    all_tasks = list(training_dir.glob("*.json"))
    
    if len(all_tasks) == 0:
        print(f"❌ No task files found in {training_dir}")
        sys.exit(1)
    
    print(f"Found {len(all_tasks)} total training tasks")
    
    # Randomly select 15
    selected_tasks = random.sample(all_tasks, min(15, len(all_tasks)))
    
    print("="*80)
    print(f"RUNNING BATCH VISUAL GENERATOR ON 15 RANDOM TASKS")
    print("="*80)
    print("\nSelected tasks:")
    for i, task_file in enumerate(selected_tasks, 1):
        print(f"  {i}. {task_file.stem}")
    print("\n" + "="*80 + "\n")
    
    # Run each task
    successful = 0
    failed = 0
    
    for i, task_file in enumerate(selected_tasks, 1):
        print(f"\n{'='*80}")
        print(f"TASK {i}/15: {task_file.stem}")
        print(f"{'='*80}\n")
        
        # Run the generator
        result = subprocess.run([
            sys.executable,
            "arc-booklet-batch-visual.py",
            str(task_file)
        ])
        
        if result.returncode == 0:
            print(f"\n✅ {task_file.stem} - SUCCESS")
            successful += 1
        else:
            print(f"\n❌ {task_file.stem} - FAILED (exit code {result.returncode})")
            failed += 1
        
        print(f"\nProgress: {i}/15 complete ({successful} successful, {failed} failed)")
    
    # Final summary
    print("\n" + "="*80)
    print("BATCH COMPLETE!")
    print("="*80)
    print(f"Total: {len(selected_tasks)} tasks")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {successful/len(selected_tasks)*100:.1f}%")
    print("\nView booklets in Streamlit:")
    print("  streamlit run streamlit_comprehensive_viewer.py")
    print(f"\nBooklets saved to: batch_visual_booklets/")

if __name__ == "__main__":
    main()

