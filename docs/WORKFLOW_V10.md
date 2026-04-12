# ARC Comprehensive Solver V10 - Complete Workflow

## Overview
V10 performs visual-first comprehensive analysis using GPT-5-mini, generating step-by-step booklets for training and test examples.

## Complete Workflow

### Phase 1: Comprehensive Analysis (Multi-Stage)

#### Stage 1: Puzzle Type, Visual & Grid Analysis
**Content sent (for 3 training examples, no test input):**
1. **Main prompt** - Instructions for Stage 1 analysis (500 char limit)
2. **Training Example 1 header** - "=== TRAINING EXAMPLE 1 ===\nInput dimensions: XxY\nOutput dimensions: XxY"
3. **Training Example 1 input image** - Visual representation
4. **Training Example 1 input grid** - Grid data as text
5. **Training Example 1 output image** - Visual representation
6. **Training Example 1 output grid** - Grid data as text
7. **Training Example 2 header** - Same format
8. **Training Example 2 input image**
9. **Training Example 2 input grid**
10. **Training Example 2 output image**
11. **Training Example 2 output grid**
12. **Training Example 3 header** - Same format
13. **Training Example 3 input image**
14. **Training Example 3 input grid**
15. **Training Example 3 output image**
16. **Training Example 3 output grid**
17. **Final instruction** - "Provide your analysis for Stage 1..."

**Total: 11 text items + 6 image items** (if 3 training examples, no test input)

**Output:** Analysis text (under 500 chars) covering:
- Puzzle type identification
- Visual analysis for each example
- Grid analysis for each example

#### Stage 2: Comprehensive Comparisons
**Content sent:**
1. **Main prompt** - Instructions for Stage 2 comparisons (500 char limit)
2. **Previous analysis summary** - Stage 1 summary (300 chars)
3. **Training Example 1 header**
4. **Training Example 1 input image**
5. **Training Example 1 input grid text**
6. **Training Example 1 output image**
7. **Training Example 1 output grid text**
8. **Training Example 2 header**
9. **Training Example 2 input image**
10. **Training Example 2 input grid text**
11. **Training Example 2 output image**
12. **Training Example 2 output grid text**
13. **Training Example 3 header**
14. **Training Example 3 input image**
15. **Training Example 3 input grid text**
16. **Training Example 3 output image**
17. **Training Example 3 output grid text**
18. **Test input** (if provided) - header, image, grid
19. **Final instruction**

**Output:** Analysis text (under 500 chars) with sections:
- === INPUT-INPUT COMPARISONS ===
- === OUTPUT-OUTPUT COMPARISONS ===
- === INPUT-OUTPUT COMPARISONS ===

**Parsed into:** `comparison_sections` dict with `input_input`, `output_output`, `input_output`

#### Stage 3: Reference Objects & Whole Grid Patterns
**Content sent:**
1. **Main prompt** - Instructions for Stage 3 (500 char limit)
2. **Previous analyses summaries** - Stage 1 (200 chars) + Stage 2 (200 chars)
3. **Training examples** - Same format as Stage 1 (header, images, grids)
4. **Test input** (if provided)
5. **Final instruction**

**Output:** Analysis text (under 500 chars) covering:
- Reference objects
- Whole grid patterns
- Object analysis

#### Stage 4: General Rule Generation
**Content sent:**
1. **Main prompt** - Instructions for rule generation
2. **Stage 1 summary** - 600 chars
3. **Stage 2 summary** - 600 chars + comparison sections (300 chars each)
4. **Stage 3 summary** - 600 chars
5. **NO IMAGES** - Text-only

**Output:** General rule (3-5 sentences)

### Phase 2: General Step Generation

**Content sent:**
1. **Main prompt** - Instructions for step generation
2. **Full Stage 1 analysis** - Complete text
3. **Full Stage 2 analysis** - Complete text + structured comparisons
4. **Full Stage 3 analysis** - Complete text
5. **General rule** - Complete text
6. **Training Example 1** - Input image, output image
7. **Training Example 2** - Input image, output image
8. **Training Example 3** - Input image, output image
9. **Test input** (if provided) - Image
10. **Final instruction**

**Output:** JSON with general steps array:
```json
{
  "steps": [
    {
      "step_number": 1,
      "instruction": "Step 1: for each (condition) A, perform transition B",
      "condition": "condition_description",
      "transition": "transition_name"
    }
  ]
}
```

### Phase 3: Training Booklet Generation

For each training example:

1. **Detect Objects**
   - Input: Grid image + grid text
   - Output: List of objects with bbox, colors, description, size

2. **Match Objects** (if output available)
   - Input: Input objects, output objects, grids
   - Output: Mapping of input objects to output objects

3. **For Each General Step:**
   - Filter objects by condition
   - For each matching object:
     - **Crop Step (sub-substep .1)**
       - Crop grid to object bbox
       - Store: Full grid (shows bbox), cropped grid
     - **Transform Step (sub-substep .2)**
       - LLM transforms cropped object
       - Store: Cropped before, cropped after, full grid before
     - **Uncrop Step (sub-substep .3)**
       - Place transformed object back in full grid
       - Store: Full grid before, full grid after

**Booklet Structure:**
- General Step 1
  - Object 1 (substep 1.1)
    - Crop (1.1.1)
    - Transform (1.1.2)
    - Uncrop (1.1.3)
  - Object 2 (substep 1.2)
    - Crop (1.2.1)
    - Transform (1.2.2)
    - Uncrop (1.2.3)
- General Step 2
  - ...

### Phase 4: Test Booklet Generation

Same as training, but:
- Re-analyze with test input included
- No output ground truth available
- Adapt general steps to test input

### Phase 5: Save Results

Saves to `booklets_ARCAGI/traces/{puzzle_id}_v10_analysis.json`:
- Full analysis (all stages)
- Structured analysis sections
- Comparison sections (input-input, output-output, input-output)
- General rule
- General steps
- Training booklets
- Test booklets

## Key Features

### Analysis Stages
- **Stage 1**: Puzzle type, visual & grid analysis (500 char limit)
- **Stage 2**: Comprehensive comparisons with structured sections (500 char limit)
- **Stage 3**: Reference objects & whole grid patterns (500 char limit)
- **Stage 4**: General rule (text-only, no images)

### Step Generation
- Receives ALL analysis (full text from all stages)
- Receives structured comparisons
- Receives general rule
- Does additional textual analysis on training examples

### Booklet Generation
- Object-by-object processing
- Crop → Transform → Uncrop workflow
- Each step shows appropriate visuals:
  - Crop: Full grid with bbox
  - Transform: Cropped view before/after
  - Uncrop: Full grid before/after
- State tracking: Model sees previous step state

## Text Items Breakdown (Stage 1, 3 training examples, no test input)

1. Main prompt text
2. Example 1 header text
3. Example 1 input grid text
4. Example 1 output grid text
5. Example 2 header text
6. Example 2 input grid text
7. Example 2 output grid text
8. Example 3 header text
9. Example 3 input grid text
10. Example 3 output grid text
11. Final instruction text

**Plus 6 image items** (2 per training example: input + output)

## Critical Requirements

- **Color naming**: Always use color 0, color 1, etc. NEVER color names
- **Transition names**: Must match toolcall transformation_type exactly
- **Analysis limits**: Each stage under 500 characters
- **Visual-first**: Images analyzed before grid data
- **Objects are distinct**: Background can be any color, objects are distinct regions
- **No toolcalls in analysis**: Stage 2 is text-only analysis


