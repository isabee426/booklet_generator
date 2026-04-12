# ARC Comprehensive Solver

A complete analysis and booklet generation system for ARC puzzles that follows a comprehensive analysis workflow.

## Features

### 1. Puzzle Type Identification
- Automatically identifies puzzle type based on input/output size relationships:
  - **Input > Output**: Pattern Extraction/Cropping
  - **Input = Output**: Same-Size Transformation
  - **Input < Output**: Expansion/Tiling

### 2. Comprehensive Analysis
- **Input Comparison**: Compares all training inputs to find common elements and differences
- **Output Comparison**: Compares all training outputs to find exact matches and patterns
- **Individual Analysis**: Compares each input-output pair individually
- **Incremental Steps**: Identifies how training examples build upon each other
- **Whole Grid Patterns**: Analyzes color, shape, negative space, and common cells
- **Reference Objects**: Finds objects that remain constant across examples

### 3. Rule Generation
- Generates 3-5 sentence general rule based on comprehensive analysis
- Incorporates puzzle type, reference objects, and transition patterns

### 4. General Step Generation
- Creates general steps in format: "Step x: for each (CONDITION) A, perform transition B"
- Steps are condition-based and can be applied to multiple objects
- Supports substeps (3.1, 3.2, etc.) for complex transformations

### 5. Booklet Generation
- Generates step-by-step booklets for each training example
- Uses MCP toolcalls (crop, transform, uncrop, generate_grid, detect_objects, match_objects)
- Each step shows grid state before and after transformation
- Includes reasoning for each substep

### 6. Test Prediction
- Adapts general steps to test input
- Generates predicted output following the same transformation steps
- Uses reference objects to guide adaptation

### 7. Streamlit UI
- Interactive viewer for analysis results
- Displays puzzle overview, general steps, training booklets, and test predictions
- Visual grid comparisons with before/after states

## Usage

### Command Line

```bash
# Basic usage
python arc_comprehensive_solver.py <puzzle_file.json> [output_dir]

# Example
python arc_comprehensive_solver.py ARC-AGI-2/data/training/6a980be1.json traces
```

### Streamlit UI

```bash
streamlit run streamlit_comprehensive_viewer.py
```

## Output Structure

The solver generates a JSON file with:

```json
{
  "puzzle_id": "6a980be1",
  "puzzle_type": {
    "type_name": "Same-Size Transformation",
    "dominant_type": "input_eq_output",
    "suggested_initial_grid": {...}
  },
  "reference_objects": {...},
  "input_comparison": {...},
  "output_comparison": {...},
  "individual_comparisons": [...],
  "incremental_steps": [...],
  "whole_grid_patterns": {...},
  "transitions": {...},
  "rule": "3-5 sentence general rule...",
  "general_steps": [
    {
      "step_number": 1,
      "instruction": "Step 1: ...",
      "condition": "...",
      "transition": "..."
    }
  ],
  "training_booklets": [
    {
      "steps": [
        {
          "step_number": "1.1",
          "general_step": 1,
          "instruction": "...",
          "substep_reasoning": "...",
          "grid_before": [...],
          "grid_after": [...],
          "tool_used": "...",
          "tool_params": {...}
        }
      ],
      "input": [...],
      "output": [...]
    }
  ],
  "test_booklets": [...]
}
```

## MCP Tools

The system uses the following MCP tools for transformations:

1. **generate_grid**: Generate resulting grid after action
2. **detect_objects**: Detect all distinct objects in a grid
3. **match_objects**: Match objects between input and output grids
4. **crop**: Crop grid to a specific region
5. **transform**: Transform a cropped grid (color changes, rotations, etc.)
6. **uncrop**: Place transformed grid back into full-size grid

## Analysis Workflow

1. **Identify Puzzle Type**: Based on size relationships
2. **Find Reference Objects**: Objects that stay constant
3. **Compare All Inputs**: Find common elements and differences
4. **Compare All Outputs**: Find exact matches and patterns
5. **Individual Comparisons**: Analyze each input-output pair
6. **Incremental Steps**: Find how examples build on each other
7. **Whole Grid Patterns**: Analyze color, shape, negative space
8. **Transition Analysis**: Identify transformation patterns
9. **Rule Generation**: Create general rule
10. **Step Generation**: Create general transformation steps
11. **Booklet Generation**: Generate step-by-step booklets
12. **Test Prediction**: Apply steps to test input

## Color Mapping

Colors are referenced by number (0-9), not by name:
- Color 0: Black (background)
- Color 1: Blue
- Color 2: Red
- Color 3: Green
- Color 4: Yellow
- Color 5: Orange
- Color 6: Magenta/Pink
- Color 7: Light Blue/Cyan
- Color 8: Dark Red/Maroon
- Color 9: Purple

## Requirements

- Python 3.8+
- openai (for API calls)
- PIL/Pillow (for image generation)
- numpy
- streamlit (for UI)
- json

## Notes

- Tool names in analysis must match toolcall names exactly
- Bounding boxes use union boxes for moves
- The model performing transitions can see the state from the previous step
- Booklets should be complete with all transformation steps
- Test predictions adapt general steps based on training booklets


