# ARC-AGI Task 007bbfb7 - Instructional Booklet

Generated on: 2025-10-27 14:48:19

## Files

- input.png: Original puzzle input
- target_output.png: Expected solution
- step_NNN.png: Model output for each step
- step_NNN_expected.png: Expected/correct grid for that step (if different from model output)
- metadata.json: Complete step data

## Steps

Total steps: 6
Steps reaching target: 0

Step 0: 1. Recognize the input is a 3×3 pattern (N = 3) with two values: background (0) ...
Step 1: 2. Partition the output into a 3×3 array of 3×3 blocks. Block coordinates: block...
Step 2: 3. For each input cell at (r,c):
Step 3: - If the input cell is background (0), fill the corresponding 3×3 output block w...
Step 4: - If the input cell is foreground (7), copy the entire original 3×3 input patter...
Step 5: 4. Repeat for all 9 input cells. Do not alter or overlap blocks; use the exact c...
