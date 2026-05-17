#!/usr/bin/env python3
"""
Batch Booklet Generator
Generate booklets for multiple puzzles
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Generate booklets for easy puzzles"""
    
    # Output directory for all booklets
    output_dir = "sample_booklets"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Puzzle file paths - Mix of ARC-AGI-1 and ARC-AGI-2 training samples
    puzzles = [
        # Training puzzles from ARC-AGI-1
        "../saturn-arc/ARC-AGI-2/ARC-AGI-1/data/training/007bbfb7.json",
        "../saturn-arc/ARC-AGI-2/ARC-AGI-1/data/training/025d127b.json",
        
        # Training puzzles from ARC-AGI-2
        "../saturn-arc/ARC-AGI-2/ARC-AGI-2/data/training/00576224.json",
        "../saturn-arc/ARC-AGI-2/ARC-AGI-2/data/training/009d5c81.json",
        
        # Specific booklet to redo
        "../saturn-arc/ARC-AGI-2/ARC-AGI-1/data/training/0a2355a6.json",
    ]
    
    print("="*80)
    print("BATCH BOOKLET GENERATOR")
    print("="*80)
    print(f"Puzzles: {len(puzzles)}")
    print(f"Output: {output_dir}/")
    print("="*80)
    print()
    
    # Generate booklets
    for i, puzzle_path in enumerate(puzzles, 1):
        puzzle_name = Path(puzzle_path).stem
        
        print(f"\n[{i}/{len(puzzles)}] Generating booklet for {puzzle_name}...")
        print("-" * 80)
        
        try:
            # Run the booklet generator
            result = subprocess.run(
                [sys.executable, "arc-booklet-generator.py", puzzle_path, output_dir],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ Successfully generated booklet for {puzzle_name}")
            else:
                print(f"❌ Failed to generate booklet for {puzzle_name}")
                print(f"Error: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Exception for {puzzle_name}: {e}")
    
    print("\n" + "="*80)
    print("BATCH GENERATION COMPLETE")
    print("="*80)
    print(f"\nAll booklets saved to: {output_dir}/")
    print("\nView with Streamlit:")
    print("  streamlit run streamlit_booklet_viewer.py")
    print("="*80)


if __name__ == "__main__":
    main()

