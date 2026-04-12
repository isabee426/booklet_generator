#!/usr/bin/env python3
"""
Quick test of V2 solver to verify it works
"""

import sys
import os

# Set a simple puzzle path for testing
test_puzzle = r"..\saturn-arc\ARC-AGI-2\ARC-AGI-2\data\training\00576224.json"

print("="*80)
print("QUICK V2 TEST")
print("="*80)
print(f"\nTesting V2 solver on: {test_puzzle}")
print("\nThis will:")
print("  1. Generate THE RULE from example 1")
print("  2. Break into operations")
print("  3. Execute operations step-by-step")
print("  4. Test on example 2 and refine if needed")
print("  5. Apply to test input")
print("\n" + "="*80 + "\n")

# Import and run
from arc_booklets_solver_v2_stepwise import ARCBookletSolverV2

try:
    solver = ARCBookletSolverV2()
    booklet = solver.solve(test_puzzle)
    
    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)
    print(f"\nSteps generated: {len(booklet.steps)}")
    print(f"Accuracy: {booklet.accuracy*100:.1f}%")
    print(f"Prediction shape: {booklet.final_prediction and [len(booklet.final_prediction), len(booklet.final_prediction[0])]}")
    print(f"Expected shape: {booklet.actual_output and [len(booklet.actual_output), len(booklet.actual_output[0])]}")
    
    if booklet.accuracy == 1.0:
        print("\n✅ V2 WORKS PERFECTLY!")
    elif booklet.accuracy > 0:
        print(f"\n⚠️ V2 partially works ({booklet.accuracy*100:.1f}% accuracy)")
    else:
        print("\n❌ V2 failed")
    
    print(f"\nBooklet saved to: test/{booklet.task_name}_booklet_v2.json")
    print("View in Streamlit: streamlit run streamlit_app.py")
    print("\n" + "="*80 + "\n")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

