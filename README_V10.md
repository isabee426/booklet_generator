# ARC Comprehensive Solver V10

## Overview

V10 implements a **visual-first comprehensive analysis** approach using GPT-5-mini, following the complete puzzle-solving guide and patterns from v4 and v6.

**Key Principle**: Visual analysis FIRST, then grid analysis. Use BOTH for each step.

## Key Features

### Visual-First Analysis
1. **Visual Analysis First**: Images are analyzed before grid data
   - Shapes, colors, spatial relationships
   - Negative space, object arrangements
   - Visual transformations
   - Object arrangements and patterns

2. **Grid Analysis Second**: Precise grid data analysis
   - Exact dimensions and size relationships
   - Color patterns and counts (using color 0, color 1, etc. - NEVER color names)
   - Grid structure analysis
   - Precise transformations

3. **Combined Synthesis**: Visual + Grid insights combined for comprehensive understanding

### Comprehensive Analysis Workflow (Following Complete Guide)

#### Step 1: Puzzle Type Identification
- **Size Relationship Analysis**: 
  - Input Size > Output Size: Pattern Extraction/Cropping (73.7% pattern extraction, 71.4% cropping)
  - Input Size = Output Size: Same-Size Transformation (46.5% color mapping, 0.4% spatial transforms)
  - Input Size < Output Size: Expansion/Tiling (74.7% expansion, 63.3% tiling)
- **Ratio Variation Check**: Does input/output ratio vary between training samples? If yes, determine what corresponds to output size first
- **Initial Grid Determination**: 
  - Usually start with input size
  - May need to crop if output size smaller
  - Or something in input indicates getting to output size

#### Step 2: Comprehensive Comparisons
- **All Inputs vs Each Other** (including test input):
  - What do differences in inputs say about differences in outputs?
  - What is same among inputs? Similar objects going through similar transitions?
  - What is different? How do they build upon each other?
  - Is there correspondence (if-then) being built upon?
  
- **All Outputs vs Each Other**:
  - Anything consistent in all training outputs (exact same) should be in test output
  - What patterns emerge?
  
- **Input to Output for Each Training Example Individually**:
  - What is similar between inputs and outputs?
  - What is common in transition steps?
  - What is changed?
  - Shape, color, size, location analysis
  
- **Training Sample to Training Sample**:
  - Incremental steps building on each other
  - What builds upon what?

#### Step 3: Reference Objects (CRITICAL FOCUS)
- Objects that stay same **input-to-input** OR **input-to-output**
- Object types, object transitions
- Reference objects can be used for shape or color
- Sometimes order in reference object matters (left to right, up to down)
- Reference objects can be a **bar of different color** that divides two parts:
  - Either common quality on both sides
  - Or edit one side based on other side (based on conditions)
- If they move, why do they move? What does it say in transition?
- If no solid reference object, something about input/output tells location, color, or shape
- When reference object same color/shape/pattern/location/size as other objects:
  - What changes about this/these object(s)?

#### Step 4: Whole Grid Analysis (Per Training Sample)
Using the **WHOLE GRID** to determine why output is output:

- **Color Patterns**:
  - Color transitions and mappings
  - 1-to-1 correlations between shapes and colors, or colors and shapes
  
- **Shape Patterns**:
  - Shape transformations
  - Awareness of sized up/down objects
  - When object is scaled version of another (same grid or input to output)
  - Reshaping: "rectangularizing" or making fuzzy edges into standard shape
  
- **Negative Space** (CRITICAL):
  - **SEE NEGATIVE SPACE!** (color 0)
  - Cells common between parts of input
  - Cells common to only one part of input
  - Get rid of stray cells - difference between objects and stray cells
  
- **Common Cells**:
  - Cells that stay same position and color
  - What stays constant?
  
- **Divided Sections**:
  - Parts divided by reference object bars
  - Common quality on both sides?
  - Edit one side based on other?
  
- **Incremental Steps**:
  - Look for incremental steps building on each other
  - Training sample to training sample: what builds upon what?

#### Step 5: Object Analysis
- Same colored cells are usually one object. Focus on **object by object** unless lines
- Objects inside objects (or distinct parts of objects can be objects too)
- Usually can find operation to do over and over, maybe more than one transition step when crop
- **Lines are different type of object** - follow different rules:
  - Sequential
  - Can have starting/ending point indicated by puzzle
  - Can change color by section and turn conditionally
  - Can be drawn in order if multiple lines
  - Lines can connect objects (see output similarities)
- Understanding of dimension: when objects overlap, which is frontmost?
- Something countable about objects: holes, cell length, height, whatever
- Multiple objects same shape or color count: most common (similarly shaped/colored object) is recurring conditional

#### Step 6: Grid Size Change Specifics

**If Input Size < Output Size (up)**:
- May repeat input on output (copy/paste)
- Look for cells filled in for patterns, also patterns in background, repeated
- Look for input pattern repeated, reflected, or shown in output
- Line objects section by section (different object when perpendicular even if same color)
- Cells can have 1-1 correlation with scaled up block (say 3x3, 9x9) in output

**If Input Size > Output Size (down)**:
- There may be pattern to complete or some part of input is output repeated
- There may be "zoom", "crop", you may delete objects
- What is important about input? Keep in output? Color? Shape? Pattern?
- What does input tell about location, size, color, patterns about output?
- What does output say about input?

**Generally if grid changes**:
- Try to overlay input on output or output on input

#### Step 7: Rule Generation
- **3-5 sentence general rule**
- Based on **WHOLE analysis** above

#### Step 8: General Step Generation
- Format: **"Step x: for each (CONDITION) A, perform transition B"**
- **CONDITION**: color, location, shape, etc. Something that indicates this A experiences common transition with all other A
- **A**: object, section of line, section of grid (each 3x3 section, 9x9, etc.) but usually object by object unless no objects
- **B**: transition name (**MUST MATCH toolcall transformation_type EXACTLY**)
- Each step targets same kind of transition, compares against ground truth for each object

### Booklet Generation

- **Visual + Grid Analysis per Step**: Each step performs visual analysis FIRST, then grid analysis
- **Crop → Transform → Uncrop Workflow**: Following v4/v6 patterns
  - Crop to object/region
  - Transform the cropped region
  - Uncrop back to full grid
  - Guided by cropping, transforming, and uncropping with cropped output ground truth
- **Object-by-Object Processing**: Focus on objects unless lines (lines follow different rules)
- **Substeps with Reasoning**: Each substep has 1-2 sentences explaining why this transformation is applied to this object based on preceding analysis
- **State Tracking**: Model performing transitions can see the state of the board from the step before
- **Substep Numbering**: Each booklet can have different number of substeps (3.1, 3.2, etc.) but general steps remain the same
- **Complete Booklets**: Booklets are complete with all transformation steps

### MCP Tools

All tools available via MCP toolcalls (transition names in analysis and toolcall names must be EXACTLY the same):

- `generate_grid`: Generate resulting grid after action (includes visual_analysis: 1-2 sentences)
- `detect_objects`: Detect distinct objects in grid (each separate filled region is DISTINCT object, even if same color)
- `match_objects`: Match objects between input/output grids
- `crop`: Crop grid to a specific region (returns cropped grid and metadata)
- `transform`: Transform cropped grid (transition name MUST match analysis exactly)
  - Transformation types: color_mapping, rotate, flip_horizontal, flip_vertical, tile, etc.
  - Must provide transformed grid
- `uncrop`: Place transformed grid back into full-size grid at original position

**Important Tool Features**:
- Bounding boxes on moves should be a union box (like in v4/v6)
- Large features: shape first then color
- See cell shapes but also see negative space!
- Conditions: what objects to include in transition based on condition
- Completing objects: recolor, reshape, or move conditionally
- "Fitting" objects together: finding edges that counter each other
- Lines vs objects: how lines change color or turn, can change in line based on conditional

### Test Prediction

- **Adapts general steps to test input**: Uses training booklets and general steps to adapt instructions to test input
- **Same workflow**: Follows same crop-transform-uncrop workflow
- **Same toolcalls**: Step generation model given exact same tools, performs crop transform in same way, in order general steps describes
- **Complete booklet**: Booklet should be complete following same steps as general steps but solely based on test input

## Usage

```bash
python arc_comprehensive_solver_v10.py <puzzle_file.json> [output_dir]
```

Example:
```bash
python arc_comprehensive_solver_v10.py "ARC-AGI-2/ARC-AGI-2/data/training/5b526a93.json" traces
```

## Output

Saves to `traces/{puzzle_id}_v10_analysis.json`:
- Comprehensive analysis (visual + grid)
- General steps
- Training booklets (step-by-step)
- Test booklets (predictions)

## Requirements

- OpenAI API key (set `OPENAI_API_KEY` environment variable)
- GPT-5-mini model access
- PIL/Pillow for image generation
- All dependencies from base solver

## Key Differences from V9

1. **Visual-First**: Images analyzed before grid data
2. **LLM-Based Analysis**: All analysis done by GPT-5-mini
3. **Comprehensive Comparisons**: All inputs/outputs compared
4. **Better Tool Integration**: Proper crop-transform-uncrop workflow
5. **State Tracking**: Model sees previous step state

## Critical Requirements

### Color Naming
- **NEVER use color names explicitly**
- Always refer to colors by NUMBER: color 0, color 1, color 2, etc.
- Color mapping for reference:
  - color 0 = Black (background)
  - color 1 = Blue
  - color 2 = Red
  - color 3 = Green
  - color 4 = Yellow
  - color 5 = Orange
  - color 6 = Magenta/Pink
  - color 7 = Light Blue/Cyan
  - color 8 = Dark Red/Maroon
  - color 9 = Purple

### Transition Names
- **Transition names in analysis MUST EXACTLY MATCH toolcall transformation_type**
- Example: If analysis says "color_mapping", toolcall must use transformation_type="color_mapping"
- Best toolcall examples are in v6 and v4

### Analysis Requirements
- Visual analysis FIRST, then grid analysis (for each step)
- Use WHOLE GRID to determine why output is output (per training sample)
- Focus on reference objects (critical)
- Compare everything comprehensively (all inputs, all outputs, individual comparisons)
- Look for incremental steps building on each other
- See negative space! (color 0)

### Booklet Requirements
- Each substep should have 1-2 sentences about why it does this transition to this object based on preceding analysis
- Bounding boxes on moves should be a union box
- Model performing transitions can see state of board from step before
- Booklets are complete with all transformation steps

