# ARC-AGI Task 3e6067c3 - Instructional Booklet

## Task Description
This booklet demonstrates the step-by-step solution process for ARC-AGI task 3e6067c3.

## Solution Strategy

The puzzle involves connecting colored blocks using bridges:

1. **Identify the instruction row**: The bottom row contains a sequence of colors that act as instructions

2. **Find adjacent color pairs**: For each pair of adjacent colors in the instruction row:
   - Locate the blocks with those colors in the grid
   - These blocks will be next to each other, separated by a border

3. **Draw bridges**: Connect the blocks using bridges:
   - Use the color of the FIRST block in the pair
   - Bridge width matches the inner block width
   - Never draw over borders
   - Bridges should be centered

4. **Repeat**: Continue for all color pairs in the sequence

## Steps in This Booklet

- **step_000.png**: Initial input with instruction row highlighted
- **step_001.png**: First bridge (color 2 to color 3)
- **step_002.png**: Second bridge (color 3 to color 9)
- **step_003.png**: Third bridge (color 9 to color 4)
- **step_004.png**: Fourth bridge (color 4 to color 2)
- **step_005.png**: Fifth bridge (color 2 to color 6)
- **step_006.png**: Sixth bridge (color 6 to color 7)
- **step_007.png**: Final bridge (color 7 to color 5)

## Key Rules

1. **Never overwrite borders**: Bridges only go through empty spaces (color 8)
2. **Bridge width**: Match the width of the colored blocks being connected
3. **Bridge color**: Use the color of the FIRST block in each pair
4. **Centering**: Bridges should be centered on the blocks they connect

## Files

- `input.png`: Original puzzle input
- `target_output.png`: Expected solution
- `step_NNN.png`: Step-by-step solution images with metadata

Each step image contains metadata with:
- Step number
- English description of the action
- Grid data in JSON format
- Timestamp

Generated on: 2025-10-16 11:18:20
