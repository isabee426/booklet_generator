# Metadata & Verification Features

## Overview
The solver now embeds rich metadata in each visualization PNG file and automatically verifies hypotheticals against expected outputs.

## Metadata Fields

Each visualization image contains:

### 1. `description`
- What the transformation/step represents
- Example: "Rotating each 2x2 block clockwise"
- Helps understand what the model was thinking

### 2. `visualization_type`
- Either `"hypothetical"` or `"transform"`
- **hypothetical**: Different approaches being explored
- **transform**: The chosen approach for this step

### 3. `grid_dimensions`
- Size of the grid (e.g., "5x5", "3x7")
- Quick reference without loading full grid

### 4. `timestamp`
- ISO format timestamp of when created
- Example: "2025-10-12T14:30:45.123456"

### 5. `grid_data`
- The actual grid as JSON array
- Enables automatic verification
- Example: `[[0,1,2],[3,4,5]]`

### 6. `step_details`
- `iteration`: Which API call iteration
- `training_example`: Which training example (0-indexed)
- `task`: Task ID (e.g., "1ae2feb7")

## Verification Feature

### How It Works
1. Model creates multiple hypotheticals
2. Model picks one as transform
3. **NEW**: System compares ALL hypotheticals to expected output
4. **NEW**: Reports which hypothetical was closest
5. **NEW**: Model gets feedback on accuracy

### Example Output
```
🔍 Verifying hypotheticals against expected output:
  h_01.png: 3 cells different - Rotating pattern clockwise
  h_02.png: 0 cells different - Flipping pattern horizontally
  h_03.png: 12 cells different - Scaling pattern by 2
✅ Best match: 0 cells different
```

### Benefits
- Model learns which approaches work best
- Helps correct course before final answer
- Provides clear feedback on reasoning quality
- Can identify "almost correct" approaches

## File Organization

Files are saved in this order:
```
visualizations/
  task_id/
    training_1/
      01_input.png
      02_hypotheticals/
        h_01.png  ← Has metadata
        h_02.png  ← Has metadata
        h_03.png  ← Has metadata
      02_transform.png  ← Has metadata
      03_hypotheticals/
        h_01.png  ← Has metadata
      03_transform.png  ← Has metadata
      XX_model_output.png
      YY_actual_output.png
```

## Accessing Metadata

### In Python
```python
from arc_visual_solver import ARCVisualSolver

solver = ARCVisualSolver()

# Read metadata from any image
metadata = solver.get_visualization_metadata('path/to/image.png')

print(metadata['description'])
print(metadata['visualization_type'])
print(metadata['grid_data'])  # The actual grid
```

### List All Visualizations
```python
vizs = solver.list_visualizations('task_id', training_example=0)
for viz in vizs:
    print(f"{viz['filename']}: {viz['metadata']['description']}")
```

## Technical Implementation

### PNG Metadata Storage
- Uses PNG text chunks via `PngImagePlugin.PngInfo`
- Metadata travels with the file
- No separate metadata files needed
- Survives file copying/moving

### Verification Algorithm
1. Load expected output grid
2. For each hypothetical directory:
   - Load each hypothetical's `grid_data` from metadata
   - Calculate cell-by-cell difference
   - Track the closest match
3. Report results to model

### Grid Difference Calculation
- Compares dimensions first (must match)
- Counts number of different cells
- Returns infinity if dimensions don't match

## Next Run

The next time you run the solver:
1. All new images will have complete metadata
2. Verification will run automatically
3. Model will see which hypotheticals were closest
4. Console output will show verification results

Try it:
```bash
python arc_visual_solver.py "path/to/task.json"
```

Then check metadata:
```bash
python test_metadata.py
```
