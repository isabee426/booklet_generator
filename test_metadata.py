#!/usr/bin/env python3
"""Test metadata reading and verification"""
from arc_visual_solver import ARCVisualSolver
import json

solver = ARCVisualSolver()

print("\n" + "="*80)
print("METADATA FEATURES")
print("="*80)
print("""
After the next solver run, each visualization will have:

1. description: What this step represents
2. visualization_type: "hypothetical" or "transform"
3. grid_dimensions: Size like "5x5"
4. timestamp: When created
5. grid_data: Actual grid as JSON (for verification)
6. step_details: iteration, training_example, task

VERIFICATION: Compares hypotheticals to expected output
and shows which one was closest!
""")

# Read metadata from a hypothetical
print("\n" + "="*80)
print("OLD FILE METADATA (from previous run)")
print("="*80)
metadata = solver.get_visualization_metadata('visualizations/1ae2feb7/training_1/02_hypotheticals/h_01.png')
for key, value in metadata.items():
    print(f"\n{key}:")
    if isinstance(value, dict):
        print(json.dumps(value, indent=2))
    else:
        print(f"  {value}")

# Read metadata from a transform
print("\n" + "="*80)
print("TRANSFORM METADATA")
print("="*80)
metadata = solver.get_visualization_metadata('visualizations/1ae2feb7/training_1/02_transform.png')
for key, value in metadata.items():
    print(f"\n{key}:")
    if isinstance(value, dict):
        print(json.dumps(value, indent=2))
    else:
        print(f"  {value}")

# List all visualizations in order
print("\n" + "="*80)
print("ALL VISUALIZATIONS IN ORDER")
print("="*80)
vizs = solver.list_visualizations('1ae2feb7', training_example=0)
for i, viz in enumerate(vizs, 1):
    desc = viz['metadata'].get('description', 'N/A')
    viz_type = viz['metadata'].get('visualization_type', 'N/A')
    print(f"{i:2d}. {viz['filename']:40s} | {viz_type:12s} | {desc[:50]}")
