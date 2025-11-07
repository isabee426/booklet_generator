#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Step Generator - Uses training booklets to solve test examples

This script:
1. Loads ALL training example booklets (step descriptions + images)
2. Asks model to summarize the reasoning pattern from the visual booklets
3. Applies that pattern to solve test examples step-by-step
4. Each test step receives: training images, current step instruction, and test state

Usage:
    python scripts/test_step_generator.py --puzzle 05f2a901 [--test-num 1]
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from PIL import Image
import sys

# Set UTF-8 encoding for stdout/stderr on Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.arc_visualizer import grid_to_image
from scripts.visual_step_generator import VisualStepGenerator


class TestStepGenerator(VisualStepGenerator):
    """
    Generate test solutions using training booklet patterns.
    
    Inherits all parsing methods from VisualStepGenerator:
    - _parse_grid: Robust grid parsing with multiple strategies
    - _parse_description: Extract description from response
    - _validate_grid_structure: Validate grid dimensions
    - call_api: OpenAI API calls with image support
    """
    
    def load_training_booklets(self, puzzle_id: str) -> List[Dict]:
        """Load all training example booklets"""
        # Check visual_step_results first, then successful_runs
        results_dir = Path("visual_step_results") / puzzle_id
        
        if not results_dir.exists():
            # Try successful_runs as fallback
            results_dir = Path("successful_runs") / puzzle_id
            if not results_dir.exists():
                raise FileNotFoundError(
                    f"No training booklets found for {puzzle_id}.\n"
                    f"Run training first: python scripts/visual_step_generator.py --puzzle {puzzle_id} --all\n"
                    f"Or mark an existing run as successful: python scripts/mark_successful.py --puzzle {puzzle_id}"
                )
        
        # Find all training directories
        training_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("training_")])
        
        if not training_dirs:
            raise FileNotFoundError(
                f"No training results found in {results_dir}.\n"
                f"The directory exists but has no training_XX folders."
            )
        
        booklets = []
        for td in training_dirs:
            results_file = td / "results.json"
            if results_file.exists():
                with open(results_file, encoding='utf-8') as f:
                    data = json.load(f)
                    booklets.append({
                        "training_num": data['training_num'],
                        "input_grid": data['input_grid'],
                        "output_grid": data['expected_output_grid'],
                        "steps": data['steps'],
                        "images_dir": td
                    })
        
        return booklets
    
    def generate_training_commonalities(self, booklets: List[Dict]) -> str:
        """Generate a concise summary of what's common across all training examples"""
        common_text = []
        
        # Analyze input/output sizes (handle missing grids gracefully)
        input_sizes = []
        output_sizes = []
        for b in booklets:
            input_grid = b.get('input_grid')
            output_grid = b.get('output_grid')
            if input_grid:
                input_sizes.append((len(input_grid), len(input_grid[0]) if input_grid and input_grid[0] else 0))
            else:
                input_sizes.append((0, 0))
            if output_grid:
                output_sizes.append((len(output_grid), len(output_grid[0]) if output_grid and output_grid[0] else 0))
            else:
                output_sizes.append((0, 0))
        
        common_text.append("TRAINING EXAMPLES COMMONALITIES:")
        common_text.append(f"- {len(booklets)} training examples total")
        
        # Size patterns
        if len(set(input_sizes)) == 1:
            common_text.append(f"- ALL inputs same size: {input_sizes[0][0]}×{input_sizes[0][1]}")
        else:
            common_text.append(f"- Input sizes vary: {[f'{h}×{w}' for h,w in set(input_sizes)]}")
        
        if len(set(output_sizes)) == 1:
            common_text.append(f"- ALL outputs same size: {output_sizes[0][0]}×{output_sizes[0][1]}")
        else:
            common_text.append(f"- Output sizes vary: {[f'{h}×{w}' for h,w in set(output_sizes)]}")
        
        # Size transformations
        size_changes = []
        for i, (b, inp_size, out_size) in enumerate(zip(booklets, input_sizes, output_sizes), 1):
            if inp_size == (0, 0) or out_size == (0, 0):
                size_changes.append(f"Training {i}: size info unavailable")
                continue
            if inp_size[0] * inp_size[1] > out_size[0] * out_size[1]:
                size_changes.append(f"Training {i}: {inp_size[0]}×{inp_size[1]}→{out_size[0]}×{out_size[1]} (CROP)")
            elif inp_size[0] * inp_size[1] < out_size[0] * out_size[1]:
                size_changes.append(f"Training {i}: {inp_size[0]}×{inp_size[1]}→{out_size[0]}×{out_size[1]} (EXPAND)")
            else:
                size_changes.append(f"Training {i}: {inp_size[0]}×{inp_size[1]} (SAME SIZE)")
        
        common_text.append("- Size transformations:")
        for sc in size_changes:
            common_text.append(f"  * {sc}")
        
        # Step counts
        step_counts = [len(b['steps']) for b in booklets]
        if len(set(step_counts)) == 1:
            common_text.append(f"- ALL used {step_counts[0]} steps")
        else:
            common_text.append(f"- Step counts vary: {step_counts}")
        
        return "\n".join(common_text)
    
    def summarize_reasoning_pattern(self, analysis: str, booklets: List[Dict]) -> str:
        """
        Ask model to summarize the reasoning pattern from training booklets.
        Note: 'analysis' parameter is kept for backwards compatibility but not used in prompt.
        """
        print("\n" + "="*80)
        print("SUMMARIZING REASONING PATTERN FROM TRAINING EXAMPLES")
        print("="*80 + "\n")
        
        # Prepare images: ALL steps from ALL training examples
        images = []
        img_descriptions = []
        img_idx = 1
        
        for booklet in booklets:
            training_num = booklet['training_num']
            
            # Add input/output for this training example
            images.append(grid_to_image(booklet['input_grid'], 30))
            img_descriptions.append(f"Image {img_idx}: Training {training_num} - INPUT")
            img_idx += 1
            
            images.append(grid_to_image(booklet['output_grid'], 30))
            img_descriptions.append(f"Image {img_idx}: Training {training_num} - EXPECTED OUTPUT")
            img_idx += 1
            
            # Add each step's final grid
            for step in booklet['steps']:
                step_num = step['step_num']
                step_img_path = booklet['images_dir'] / f"step_{step_num:02d}_final.png"
                
                if step_img_path.exists():
                    images.append(Image.open(step_img_path))
                    img_descriptions.append(f"Image {img_idx}: Training {training_num} - Step {step_num}")
                    img_idx += 1
        
        img_desc_text = "\n".join(img_descriptions)
        
        # Build text descriptions of all steps WITH grid data
        booklet_text = []
        for booklet in booklets:
            training_num = booklet['training_num']
            booklet_text.append(f"\n{'='*70}")
            booklet_text.append(f"TRAINING EXAMPLE {training_num}:")
            booklet_text.append(f"{'='*70}")
            
            # Show input grid for verification
            booklet_text.append(f"\nINPUT GRID (for object verification):")
            input_preview = self._format_grid(booklet['input_grid'])
            booklet_text.append(input_preview[:500] + "..." if len(input_preview) > 500 else input_preview)
            
            for step in booklet['steps']:
                step_num = step['step_num']
                desc = step['final_description']
                
                # Show if it matched ground truth
                match_status = "✅ MATCHED GT" if step['comparison']['match'] else f"⚠️ {step['comparison']['accuracy']:.0%} match"
                
                booklet_text.append(f"\nStep {step_num}: {match_status}")
                booklet_text.append(f"  Description: {desc}")
        
        booklet_descriptions = "\n".join(booklet_text)
        
        prompt = f"""You are analyzing COMPLETED training example booklets for an ARC puzzle.
Your goal: Extract the UNIVERSAL REASONING PATTERN that applies to ALL training examples.

═══════════════════════════════════════════════════════════════════════════
ALL IMAGES:
═══════════════════════════════════════════════════════════════════════════

{img_desc_text}

═══════════════════════════════════════════════════════════════════════════
TRAINING EXAMPLE BOOKLETS (Step-by-Step Descriptions):
═══════════════════════════════════════════════════════════════════════════

{booklet_descriptions}

═══════════════════════════════════════════════════════════════════════════
CRITICAL: OBJECT IDENTIFICATION REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════

⚠️ BEFORE writing the pattern, IDENTIFY ALL OBJECTS in the training examples:

For EACH training example, list ALL objects by examining the IMAGES and GRID DATA:

1. COUNT the cells for each distinct object
2. Determine EXACT dimensions (e.g., 1×1, 2×2, 3×1, irregular)
3. Note the COLOR
4. Note approximate POSITION (e.g., rows 2-3, cols 1-3)

TERMINOLOGY (BE PRECISE):
- "single-cell object" = EXACTLY 1 cell (1×1)
- "2×2 block" = EXACTLY 4 cells in a square
- "2×1 rectangle" = EXACTLY 2 cells
- "3×2 rectangle" = EXACTLY 6 cells
- "multi-cell irregular object" = More than 1 cell, not a perfect rectangle

❌ WRONG: Calling a 2×2 block a "single-cell object"
❌ WRONG: Calling a 4-cell shape "one object" without specifying dimensions
✅ RIGHT: "2×2 block of color 8 (4 cells)"
✅ RIGHT: "single-cell color 7 object (1 cell)"

═══════════════════════════════════════════════════════════════════════════
YOUR TASK: EXTRACT THE UNIVERSAL REASONING PATTERN
═══════════════════════════════════════════════════════════════════════════

Based on the {len(booklets)} training examples above, create a UNIVERSAL STEP-BY-STEP PATTERN
that describes the transformation logic.

⚠️⚠️⚠️ CRITICAL: DESCRIBE OBJECTS GENERICALLY - DO NOT MENTION SPECIFIC COLORS! ⚠️⚠️⚠️

Instead of saying "color 2 object" or "color 6 loop", describe objects by their ROLE:
- "the loop/cluster that transforms" (not "color 2 cluster")
- "the loop that stays unchanged" (not "color 6 loop")
- "objects that remain fixed" (not "color 6 and color 8 objects")
- "the object that gets replaced" (not "color 2 object")

Examples of GENERIC descriptions:
❌ BAD: "Copy the color 2 object (color 2, irregular 5-cell, rows 2-4, cols 1-3)"
✅ GOOD: "Copy objects from input to output (dimensions vary, positions vary per example)"
❌ BAD: "Move the color 8 block down 6 rows"
✅ GOOD: "Move the movable object down N rows to align with target (dimensions and colors vary)"

For each step number, describe:
1. What action type (COPY/MOVE/EXPAND/FILL/MODIFY/HIGHLIGHT/TRANSFORM/UN-HIGHLIGHT/NO-OP)
2. What object/region is affected - GENERICALLY (not specific colors!)
3. What happens to it - GENERICALLY
4. IMPORTANT: Use conditional language for steps that may not apply to all examples
   (e.g., "if such object exists, otherwise do nothing")

ACTION TYPES AVAILABLE:
- COPY: Copy object from input to output at same position
- MOVE: Move object to new position
- EXPAND: Expand object by scaling factor (each cell becomes N×N block)
- HIGHLIGHT: Mark ANCHOR cells that STAY FIXED (3-step subprocess - INVERTED LOGIC)
  
  * ⚠️⚠️⚠️ INVERTED LOGIC: HIGHLIGHT = ANCHOR (CELLS THAT STAY FIXED)! ⚠️⚠️⚠️
  
  * FORBIDDEN PHRASES (NEVER USE THESE):
    - "cells that will be moved" - NO! Highlight what STAYS!
    - "cells that will change" - NO! Highlight what WON'T change!
    - "cells that will be transformed" - NO! Highlight ANCHORS!
    - "all color X cells that will..." - NO! Highlight what's FIXED!
  
  * CRITICAL RULE: HIGHLIGHT = ANCHOR (STAYS FIXED), NON-HIGHLIGHTED = WILL CHANGE
    - Cell will change (position/color/state) → DO NOT HIGHLIGHT IT
    - Cell stays the same → HIGHLIGHT IT (it's an anchor)
    - Highlighted cells mark what WON'T change
    - TRANSFORM operates on everything EXCEPT highlighted cells
  
  * GROUPING: Anchor all cells that will stay fixed
    - If background cells stay fixed → highlight them
    - If left half stays fixed → highlight it
    - Everything else (non-highlighted) will be transformed
  
  * EXAMPLES (CORRECT - INVERTED):
    - 10-cell object: 3 move right, 7 stay → highlight the 7 that STAY (anchors)
    - 8-cell object: top 2 change color, bottom 6 stay → highlight the bottom 6 (anchors)
    - Grid: left half stays, right half moves → highlight left half (anchors)
    - Background (100 cells) stays, object (5 cells) moves → highlight background (anchors)
  
  * EXAMPLES (WRONG):
    - WRONG: "Highlight cells that will change" - NO! Highlight cells that STAY FIXED (anchors)!
    - WRONG: "Highlight the moving cells" - NO! Highlight the NON-moving cells (anchors)!
  
  * 3-STEP PROCESS WITH CONCRETE EXAMPLE:
  
  Starting grid: Background (color 0) + Object (color 6 at rows 1-3, cols 2-4)
  Goal: Move color-6 object right by 1 column
  
  Step N (HIGHLIGHT): 
    Description: "HIGHLIGHT: Mark the 90 ANCHOR cells at rows 1-10, cols 1-9 (color 0) using temporary color 8. These cells will stay fixed."
    What you do in THIS STEP's grid: Change ALL color-0 cells to color 8
    Grid now has: color 8 (anchors) + color 6 (will move)
  
  Step N+1 (TRANSFORM):
    Description: "TRANSFORM: Move all non-highlighted color-6 cells right by 1 column, keeping color-8 anchors fixed."
    What you do in THIS STEP's grid: Move color-6 cells right, color-8 cells stay in place
    Grid now has: color 8 (anchors) + color 6 (moved to new position)
  
  Step N+2 (UN-HIGHLIGHT):
    Description: "UN-HIGHLIGHT: Restore all color 8 (anchor) cells to their original color 0."
    What you do in THIS STEP's grid: Change ALL color-8 cells back to color 0
    Grid now has: color 0 + color 6 (final output - no temporary colors!)
  
  * STEP 1 (HIGHLIGHT) - STRICT TEMPLATE (MAXIMUM 2 SENTENCES - INVERTED):
    
    ⚠️ INVERTED LOGIC: HIGHLIGHT = ANCHOR (cells that STAY FIXED)
    
    ⚠️ FORBIDDEN PHRASES - NEVER USE:
    - "cells that will be moved/changed/transformed" (NO! Highlight what STAYS!)
    - "all color X cells" (without positions)
    - ANY phrase with "will be" (unless saying "will stay fixed")
    
    REQUIRED TEMPLATE:
    Sentence 1: "HIGHLIGHT: In the current grid, mark the N ANCHOR cells at [positions] (currently color C) using temporary color X. These cells will stay fixed."
    Sentence 2 (optional): "All other cells will be transformed in the next step."
    
    MAXIMUM 2 SENTENCES! Be concise!
    
    VERIFICATION: List exact positions. Convert 1-based → 0-based coordinates. Count must match.
    
    EXAMPLE:
    "HIGHLIGHT: Mark the 90 ANCHOR cells at rows 1-10, cols 1-9 (color 0) using temporary color 8. These cells will stay fixed."
  
  * STEP 2 (TRANSFORM) - STRICT TEMPLATE (MAXIMUM 2 SENTENCES - INVERTED):
    
    ⚠️ INVERTED LOGIC: Transform NON-highlighted cells, keep highlighted (anchor) cells fixed!
    
    REQUIRED TEMPLATE:
    Sentence 1: "TRANSFORM: [Action] all NON-highlighted cells (those not color X) [operation details], keeping the color X anchor cells fixed."
    Sentence 2 (optional): "Grid size remains H x W."
    
    MAXIMUM 2 SENTENCES! Anchor cells (color X) stay unchanged.
    
    EXAMPLE:
    "TRANSFORM: Move all non-highlighted color-6 cells right by 1 column, keeping the color-8 anchor cells fixed."
  
  * STEP 3 (UN-HIGHLIGHT) - STRICT TEMPLATE (MAXIMUM 1 SENTENCE):
    
    REQUIRED TEMPLATE:
    "UN-HIGHLIGHT: Restore all color X (anchor) cells to their original color C."
    
    THAT'S IT! Just 1 sentence.
    
    EXAMPLE:
    "UN-HIGHLIGHT: Restore all color 8 (anchor) cells to their original color 0."
  
  * PRECISION: Highlight ONLY the exact cells that will change (specify which cells using spatial terms!)
  * EXPLAIN WHY these specific cells (based on pattern learned from training examples)
  * Specify the exact count of cells being highlighted with spatial description
- TRANSFORM: Apply transformation to highlighted cells (Step 2 of HIGHLIGHT subprocess)
  * STRICT TEMPLATE (MAXIMUM 2 SENTENCES):
  * Sentence 1: "TRANSFORM: [Action] all color X (highlighted) cells [operation]. Cells remain color X (still highlighted)."
  * Sentence 2 (optional): "Grid size remains H x W."
  * Example: "TRANSFORM: Move all color 8 (highlighted) cells right by 1 column, from (row, col) to (row, col+1). Cells remain color 8 (still highlighted)."
  
- UN-HIGHLIGHT: Restore original colors (Step 3 of HIGHLIGHT subprocess)
  * STRICT TEMPLATE (MAXIMUM 1 SENTENCE):
  * "UN-HIGHLIGHT: Restore all color X (highlighted) cells to their original color C."
  * Example: "UN-HIGHLIGHT: Restore all color 8 (highlighted) cells to their original color 6."
- FILL: Fill a region with a color
- MODIFY: Change object properties (color/shape/size)
- NO-OP: Do nothing (for conditional steps)

⚠️ INTERMEDIATE REASONING STATES:

Some transformations require INTERMEDIATE STEPS that show visual reasoning:

Example - Scaling/Tiling (3×3 → 9×9):
  Step 1: EXPAND - Expand grid to 9×9 to accommodate 3× scaling
  Step 2: EXPAND - Expand each input cell to a 3×3 block (showing intermediate scaled pattern)
  Step 3: FILL - For each 3×3 region, fill with original input pattern (tiling rule)

Step 2 is an INTERMEDIATE STATE that demonstrates understanding of the scaling rule!

⚠️ BREAKING DOWN COMPLEX STEPS:

If a training step description combines MULTIPLE atomic actions (e.g., "copy AND move"), 
you MUST break it down into sub-steps to maintain one action per step:

Example - If training says:
  "Step 3: Copy color 8 block and move it down 6 rows"

You should break it into:
  Step 3.1: COPY - Copy the color 8 block (color 8, 2×2, rows 10-11, cols 3-4) from input to output
  Step 3.2: MOVE - Move the color 8 block (color 8, 2×2, rows 10-11, cols 3-4) down 6 rows to rows 16-17

Use sub-steps (X.1, X.2, X.3) whenever needed to ensure ONE ATOMIC ACTION per step!

FORMAT YOUR RESPONSE AS:

OBJECT INVENTORY (verify first):
Training 1 objects: [List each with (color, dimensions, position)]
Training 2 objects: [List each with (color, dimensions, position)]
Training 3 objects: [List each with (color, dimensions, position)]

UNIVERSAL PATTERN SUMMARY:
[2-3 sentences describing the overall transformation]

STEP-BY-STEP PATTERN:

⚠️⚠️⚠️ CRITICAL INSTRUCTIONS FOR PATTERN GENERATION ⚠️⚠️⚠️

1. ❌ DO NOT MENTION SPECIFIC COORDINATES FROM TRAINING!
   - WRONG: "mark these 12 anchor cells (0-based): (0,1),(0,2),(3,5)..."
   - WRONG: "If the input matches Training 1 layout, mark..."
   - RIGHT: "Mark anchor cells that represent objects staying unchanged"
   - RIGHT: "Identify which object transforms by comparing to the pattern in training examples"

2. ❌ DO NOT MENTION SPECIFIC COLORS FROM TRAINING!
   - WRONG: "the color 6 loop" or "color 2 cluster"
   - RIGHT: "the larger loop object" or "the object that transforms"

3. ✅ DESCRIBE HOW TO IDENTIFY OBJECTS IN THE TEST:
   - "The object that appears to be a loop/irregular shape"
   - "The smaller of the two objects"
   - "Objects that should transform (similar to what transformed in training)"
   - "Objects that stay unchanged (similar to what stayed in training)"

4. ✅ DESCRIBE THE TRANSFORMATION LOGIC:
   - "Replace the transforming object with a 3×3 hollow square"
   - "Move the movable object to align with..."
   - "Scale each cell by factor of N"
   - "Keep certain objects fixed while transforming others"

5. ⚠️ REMEMBER: The TEST example will have:
   - DIFFERENT colors than training (e.g., training had color 2, test has color 4)
   - DIFFERENT positions (e.g., training object at rows 1-5, test at rows 2-6)
   - DIFFERENT dimensions (e.g., training had 8 cells, test might have 18 cells)
   - But the SAME TRANSFORMATION LOGIC!

6. ✅ THE PATTERN IS THE GUIDE, NOT THE EXACT RECIPE:
   - Training examples show you WHAT TYPE of transformation to do
   - Test execution should ADAPT the transformation to the test's objects
   - Focus on ROLES (what transforms vs what stays) not SPECIFICS (color 2 vs color 6)

⚠️ Number your steps EXACTLY like this (with "Step X:" at the start of each line):

Step 1: [Action type + GENERIC description of what to do]
Step 2: [Action type + GENERIC description, NO specific coordinates!]
Step 3.1: [If needed - first atomic action, GENERIC]
Step 3.2: [If needed - second atomic action, GENERIC]
Step 4: [Action type + GENERIC description]

DO NOT use "1." or "1)" or any other format - MUST be "Step 1:", "Step 2:", "Step 3.1:", etc.
You may use MORE steps than the training ground truth if needed to break down complex actions.
Each step must be ONE atomic action (COPY/MOVE/MODIFY/HIGHLIGHT/TRANSFORM/UN-HIGHLIGHT/NO-OP).

✅ GOOD PATTERN EXAMPLE:
Step 1: COPY: Copy all objects from input to output at their original positions.
Step 2: HIGHLIGHT: Mark anchor cells (objects that stay unchanged) using a temporary color not in the puzzle.
Step 3: TRANSFORM: Replace the object designated for transformation with a 3×3 hollow square pattern centered at the object's location.
Step 4: UN-HIGHLIGHT: Restore anchor cells to their original colors.

❌ BAD PATTERN EXAMPLE (TOO SPECIFIC):
Step 1: COPY: Copy color 6 loop (12 cells) and color 2 cluster (8 cells) from input.
Step 2: HIGHLIGHT: Mark cells at (0,1),(0,2),(3,5),(4,3) using color 8.
Step 3: TRANSFORM: Replace color 2 cells with 3×3 square at rows 7-9, cols 2-4.

Make this pattern GENERAL enough to apply to ANY test example with different colors/positions/sizes!
"""
        
        print(f"Sending {len(images)} images to {self.model}...")
        print(f"Prompt length: {len(prompt)} characters")
        print(f"Requesting pattern summary...\n")
        
        try:
            pattern_summary = self.call_api(prompt, images)
            
            print("\nPATTERN SUMMARY:")
            print("="*80)
            print(pattern_summary)
            print("="*80 + "\n")
            
            # Verify we got a proper response
            if not pattern_summary or len(pattern_summary) < 100:
                print("⚠️ WARNING: Pattern summary is very short or empty!")
                print(f"   Length: {len(pattern_summary)} characters")
                print(f"   This may cause issues in test generation.")
            
            # Check if it contains the required sections
            if "STEP-BY-STEP PATTERN" not in pattern_summary:
                print("⚠️ WARNING: Pattern summary missing 'STEP-BY-STEP PATTERN' section!")
            
            if "Step 1" not in pattern_summary:
                print("⚠️ WARNING: Pattern summary doesn't contain any 'Step X' entries!")
                print(f"   Full response:\n{pattern_summary}")
            
            return pattern_summary
            
        except Exception as e:
            print(f"❌ ERROR generating pattern summary: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def solve_test_example(self, puzzle_id: str, test_num: int = 1):
        """Solve a test example using the learned pattern"""
        print(f"\n{'='*80}")
        print(f"TEST SOLVER: {puzzle_id} (test_{test_num:02d})")
        print(f"{'='*80}\n")
        
        # Load ARC test data
        arc = self.load_arc_puzzle(puzzle_id)
        if 'test' not in arc or len(arc['test']) < test_num:
            raise ValueError(f"Test example {test_num} not found for puzzle {puzzle_id}")
        
        test_example = arc['test'][test_num - 1]
        test_input = test_example['input']
        test_output = test_example.get('output', None)  # May not exist for evaluation set
        
        print(f"Test input: {len(test_input)}×{len(test_input[0])}")
        if test_output:
            print(f"Test output (ground truth): {len(test_output)}×{len(test_output[0])}")
        else:
            print("Test output: Not available (evaluation set)")
        print()
        
        # Load Phase 1 analysis (check both visual_step_results and successful_runs)
        analysis_file = Path("visual_step_results") / puzzle_id / "phase1_analysis.txt"
        if not analysis_file.exists():
            # Try successful_runs as fallback
            analysis_file = Path("successful_runs") / puzzle_id / "phase1_analysis.txt"
            if not analysis_file.exists():
                raise FileNotFoundError(
                    f"Phase 1 analysis not found for {puzzle_id}.\n"
                    f"Run training first: python scripts/visual_step_generator.py --puzzle {puzzle_id} --all"
                )
        
        with open(analysis_file, encoding='utf-8') as f:
            analysis = f.read()
        
        print("✓ Loaded Phase 1 analysis (for saving to results only, not sent to model)\n")
        
        # Load training booklets
        booklets = self.load_training_booklets(puzzle_id)
        print(f"✓ Loaded {len(booklets)} training booklets\n")
        
        # Generate concise commonalities summary
        training_commonalities = self.generate_training_commonalities(booklets)
        print(f"\n{training_commonalities}\n")
        
        # Summarize reasoning pattern
        pattern_summary = self.summarize_reasoning_pattern(analysis, booklets)
        
        # Now solve the test using the pattern
        print(f"\n{'='*80}")
        print(f"APPLYING PATTERN TO TEST EXAMPLE")
        print(f"{'='*80}\n")
        
        # Try to detect ALL steps from the pattern (NOT limited to training step count!)
        import re
        
        # Find all step identifiers including sub-steps: "Step 1:", "Step 2:", "Step 3.1:", "Step 3.2:", etc.
        step_matches = re.findall(r'Step\s+([\d\.]+)\s*:', pattern_summary, re.IGNORECASE)
        
        if not step_matches:
            # Try "Step 1 -" or "Step 1."
            step_matches = re.findall(r'Step\s+([\d\.]+)\s*[-\)—]', pattern_summary, re.IGNORECASE)
        
        if step_matches:
            # Remove duplicates while preserving order
            seen = set()
            step_identifiers = []
            for step in step_matches:
                if step not in seen:
                    seen.add(step)
                    step_identifiers.append(step)
            
            # Sort properly handling sub-steps (1, 2, 2.1, 2.2, 3, etc.)
            def step_sort_key(step_str):
                """Convert step string to sortable tuple (main_step, sub_step)"""
                parts = step_str.split('.')
                main = int(parts[0])
                sub = int(parts[1]) if len(parts) > 1 else 0
                return (main, sub)
            
            step_identifiers = sorted(step_identifiers, key=step_sort_key)
            pattern_steps = len(step_identifiers)
            
            print(f"✓ Pattern defines {pattern_steps} steps (detected from pattern summary)")
            print(f"  Steps found: {step_identifiers}")
            
            # Show comparison with training
            training_step_counts = [len(b['steps']) for b in booklets]
            print(f"  Training examples had: {training_step_counts} steps")
            print(f"  Pattern may have more/fewer steps - that's OK!\n")
        else:
            # Fallback: use training step count
            pattern_steps = max(len(b['steps']) for b in booklets)
            step_identifiers = [str(i) for i in range(1, pattern_steps + 1)]
            print(f"⚠️ Could not detect step count from pattern summary!")
            print(f"  Using training example step count: {pattern_steps} steps")
            print(f"  Pattern summary preview:")
            print(f"  {pattern_summary[:300]}...\n")
        
        # Initialize - determine starting size from training pattern
        input_h, input_w = len(test_input), len(test_input[0])
        
        # Analyze training examples to determine if they start with input or output size
        # Look at first training booklet's first step to see what size it started with
        first_booklet = booklets[0]
        first_train_input = first_booklet['input_grid']
        first_train_output = first_booklet['output_grid']
        train_input_h, train_input_w = len(first_train_input), len(first_train_input[0])
        train_output_h, train_output_w = len(first_train_output), len(first_train_output[0])
        
        # If training input was larger than training output, we should start with input size
        if train_input_h * train_input_w > train_output_h * train_output_w:
            current_grid = [[0] * input_w for _ in range(input_h)]
            print(f"Starting grid: {input_h}×{input_w} (INPUT size - following training pattern)")
        else:
            # For same size or expanding, we don't know the output size in test
            # Default to input size to be safe
            current_grid = [[0] * input_w for _ in range(input_h)]
            print(f"Starting grid: {input_h}×{input_w} (INPUT size - safe default)")
        
        test_input_img = grid_to_image(test_input, 30)
        
        all_step_results = []
        previous_images = []
        previous_descriptions = []
        
        # Get valid colors from training examples AND test input
        valid_colors = self._get_valid_colors(arc)
        # Add colors from test input (test may have different colors than training!)
        for row in test_input:
            valid_colors.update(row)
        if test_output:
            for row in test_output:
                valid_colors.update(row)
        
        print(f"Valid colors for this puzzle (training + test): {sorted(valid_colors)}\n")
        
        # Execute exactly the steps defined in the pattern (including sub-steps)
        for step_idx, step_id in enumerate(step_identifiers, 1):
            print(f"\nTEST STEP {step_id} ({step_idx}/{pattern_steps}):")
            
            # Prepare images: include full training booklets + test state
            images = []
            img_descriptions = []
            img_idx = 1
            
            # Add ALL training booklets (input, output, and all steps)
            for booklet in booklets:
                training_num = booklet['training_num']
                
                # Training input
                train_input_img = grid_to_image(booklet['input_grid'], 30)
                images.append(train_input_img)
                img_descriptions.append(f"Image {img_idx}: Training {training_num} - INPUT")
                img_idx += 1
                
                # Training expected output
                train_output_img = grid_to_image(booklet['output_grid'], 30)
                images.append(train_output_img)
                img_descriptions.append(f"Image {img_idx}: Training {training_num} - EXPECTED OUTPUT")
                img_idx += 1
                
                # All training steps
                for step in booklet['steps']:
                    step_num = step['step_num']
                    step_img_path = booklet['images_dir'] / f"step_{step_num:02d}_final.png"
                    if step_img_path.exists():
                        images.append(Image.open(step_img_path))
                        img_descriptions.append(f"Image {img_idx}: Training {training_num} - Step {step_num}")
                        img_idx += 1
            
            # Add test input (only for first step)
            if step_idx == 1:
                images.append(test_input_img)
                img_descriptions.append(f"Image {img_idx}: TEST INPUT")
                img_idx += 1
            
            # Add only the most recent previous test step (not entire history)
            if previous_images:
                images.append(previous_images[-1])
                img_descriptions.append(f"Image {img_idx}: TEST - Previous Step")
                img_idx += 1
            
            img_desc_text = "\n".join(img_descriptions)
            
            prev_text = "\n".join([
                f"Step {i+1}: {desc}" 
                for i, desc in enumerate(previous_descriptions)
            ]) if previous_descriptions else "None - this is the first step"
            
            # Extract ONLY the current step from the pattern
            import re
            current_step_text = ""
            pattern_lines = pattern_summary.split('\n')
            
            # Find the step-by-step section
            in_step_section = False
            for i, line in enumerate(pattern_lines):
                if "STEP-BY-STEP PATTERN:" in line:
                    in_step_section = True
                    continue
                
                if in_step_section:
                    # Look for the current step
                    step_match = re.match(r'^Step\s+' + re.escape(str(step_id)) + r'\s*:', line, re.IGNORECASE)
                    if step_match:
                        # Capture this step and potentially the next line if it's a continuation
                        current_step_text = line
                        # Check if next line is a continuation (indented or not starting with "Step")
                        if i + 1 < len(pattern_lines):
                            next_line = pattern_lines[i + 1]
                            if next_line.strip() and not re.match(r'^Step\s+\d', next_line):
                                current_step_text += "\n" + next_line
                        break
            
            if not current_step_text:
                # Fallback: couldn't parse, use full pattern
                current_step_text = f"Step {step_id}: (see full pattern below)\n\n{pattern_summary}"
                print(f"  ⚠️ Warning: Could not extract step {step_id}, using full pattern")
            else:
                print(f"  ℹ️ Extracted step instruction: {current_step_text[:100]}...")
            
            prompt = f"""TEST EXAMPLE STEP GENERATION - STEP {step_id}

You are solving a TEST example using the learned pattern from training examples.

═══════════════════════════════════════════════════════════════════════════
TRAINING COMMONALITIES (WHAT'S SHARED ACROSS ALL TRAINING):
═══════════════════════════════════════════════════════════════════════════

{training_commonalities}

⚠️ These are the shared patterns - apply this SAME transformation logic to the test!

═══════════════════════════════════════════════════════════════════════════
TRAINING BOOKLETS (ALL EXAMPLES WITH VISUAL STEPS):
═══════════════════════════════════════════════════════════════════════════

{img_desc_text}

These images show you the COMPLETE training examples with all their step-by-step visualizations.
Use these as visual references to understand the pattern and apply it to the test example.

═══════════════════════════════════════════════════════════════════════════
CURRENT STEP FROM LEARNED PATTERN:
═══════════════════════════════════════════════════════════════════════════

{current_step_text}

⚠️⚠️⚠️ CRITICAL REMINDERS FOR APPLYING TRAINING PATTERN TO TEST ⚠️⚠️⚠️

1. **LOOK AT THE TRAINING BOOKLET IMAGES CAREFULLY:**
   - See EXACTLY what the training steps did
   - Which objects were copied? → Copy equivalent objects in test
   - Which parts moved? → Move equivalent parts in test
   - Which parts stayed fixed? → Keep equivalent parts fixed in test

2. **HIGHLIGHT IS FOR PARTS OF OBJECTS (PARTIAL TRANSFORMATIONS):**
   - If training highlighted the TOP HALF of an object → Highlight the TOP HALF of test object
   - If training highlighted LEFT SIDE of an object → Highlight LEFT SIDE of test object
   - If training highlighted SPECIFIC CELLS (e.g., corners) → Highlight equivalent cells in test
   - If training highlighted cells that STAY FIXED → Highlight cells that STAY FIXED in test
   - ⚠️ Pay attention to WHICH PORTION of the object was highlighted, not just "highlight color X"!

3. **CELL-BY-CELL EQUIVALENCE:**
   - Training: "Move cells from rows 1-3 to rows 4-6" → Test: Move SAME RELATIVE PORTION
   - Training: "Top 2 rows of object change color" → Test: Top 2 rows of YOUR object change color
   - Training: "Bottom-right corner stays fixed" → Test: Bottom-right corner of YOUR object stays fixed
   - Training: "Left half of object transforms" → Test: Left half of YOUR object transforms

4. **APPLY TRAINING LOGIC TO TEST OBJECTS:**
   - Training had color 2 → Test has color 4 → SAME transformation but with color 4
   - Training object was 8 cells → Test object is 18 cells → SAME type of transformation
   - Training highlighted 3 cells of object → Test: highlight EQUIVALENT portion of YOUR object

5. **EXAMPLE - PARTIAL HIGHLIGHT:**
   Training showed: "Color 6 loop (12 cells), highlight 6 cells at top, move 6 cells at bottom"
   → Test: Identify YOUR loop, highlight TOP PORTION (anchor), move BOTTOM PORTION

   Training showed: "Color 2 cluster (8 cells), corners stay fixed (4 cells highlighted), center changes"
   → Test: In YOUR cluster, corners stay fixed (highlight them), center transforms

FOR EVERY ACTION IN TRAINING BOOKLETS:
→ DO THE SAME ACTION ON THE EQUIVALENT CELLS IN TEST INPUT!
→ Match the PATTERN, not the specific colors/positions!

⚠️⚠️⚠️ CRITICAL: STEP-BY-STEP OBJECT-ORIENTED VISUALIZATION ⚠️⚠️⚠️

THIS IS AN INSTRUCTIONAL SOLUTION - EACH STEP MUST:
1. BUILD THE OUTPUT GRID STEP-BY-STEP (no giant leaps)
2. Work with ONE SPECIFIC OBJECT per step (keep actions incremental)
3. Give a brief reason WHY this step is needed (connect to training insight)
4. Use the HIGHLIGHT subprocess ONLY to anchor part of an object during partial transformations
5. Show INCREMENTAL progress - small changes from previous step
6. Follow the pattern, but be PEDAGOGICAL - teach the transformation rule
7. Use COPY to copy from input, HIGHLIGHT only for partial transformations

❌ FORBIDDEN: "Copy entire input to output" - TOO BIG! Break it down into objects!
❌ FORBIDDEN: Using HIGHLIGHT to copy things - Use COPY for copying!
✅ CORRECT: "COPY: Copy objects from input (color/dimensions vary per example)"
✅ CORRECT: "HIGHLIGHT: Mark anchor cells that stay fixed during transformation"

⚠️⚠️⚠️ FOLLOW THE TRAINING EXAMPLES ⚠️⚠️⚠️

LOOK AT THE TRAINING BOOKLET IMAGES:
- What size did training Step 1 grids have?
- Did training start with input size or output size?
- Did training CROP the grid at some point?
- When did training change grid dimensions?

DO THE EXACT SAME PROCESS FOR TEST:
- If training started with input size → You start with test input size
- If training cropped at step 5 → You crop at step 5
- If training expanded grid → You expand grid
- Match the training approach step-by-step!

⚠️ GRID SIZE CAN CHANGE:
- Your current grid size can change during transformation
- Use CROP/RESIZE action when you see training did the same
- Follow training's lead on when and how to resize

WHEN TO USE EACH ACTION:
- Objects not in grid yet? → COPY from INPUT first (ONE OBJECT PER STEP)
- Current grid wrong size? → CROP/RESIZE to change dimensions
- Input larger than output? → Either extract directly OR copy full input then CROP
- After COPY, only PART of object changes? → Use HIGHLIGHT subprocess (MANDATORY! 3 steps)
- After COPY, ENTIRE object moves/changes uniformly? → Use MOVE/MODIFY (1 step)
- Objects already in grid, need to move? → Use MOVE
- Need to fill regions by location/pattern? → Use FILL

⚠️ CRITICAL: COPY objects from INPUT BEFORE using HIGHLIGHT on them!
You cannot highlight objects that aren't in the current grid yet!

ACTION FUNCTIONS (choose ONE per step):

⚠️⚠️⚠️ KEY DISTINCTION ⚠️⚠️⚠️
- COPY = Transfer objects FROM INPUT to output (use this for copying!)
- HIGHLIGHT = Mark anchors during partial transformation (NOT for copying!)

1. COPY(source=INPUT, objects=[list], positions=[coords])
   ⚠️ THIS IS HOW YOU COPY THINGS FROM INPUT!
   Parameters:
   - source: MUST be "INPUT" (can only copy from input grid!)
   - objects: List of objects with (color, dimensions, position)
   - positions: Where to place in output
   ⚠️ IF COPYING ALL OBJECTS FROM INPUT: Copy them ALL in ONE step!
   ⚠️ IF COPYING SOME OBJECTS: Copy each subset in separate steps OR multiple in one step if logical
   ⚠️ IF INPUT LARGER THAN OUTPUT: Extract/condense - copy only relevant parts from larger input
   
   Examples:
   - "COPY: Copy objects from input (12 cells) to output" 
- "COPY: Copy color 2 cluster (8 cells) FROM INPUT to output" - Next COPY step handles next object
   - "COPY: Extract left portion of input (rows 0-2, cols 0-2) to output" - Condensing from 3×7 to 3×3
   
   When to use: When transferring/extracting objects from input to output
   Note: Input may be LARGER than output - extract only relevant parts!

2. MOVE(objects=[list], from=[coords], to=[coords])
   Parameters:
   - objects: Object being moved (color, dimensions)
   - from: Current position in grid
   - to: New position
   Example: "MOVE: Move object (2×2) down 6 rows"

3. EXPAND(source=INPUT, scale_factor=N, mapping=[rules])
   Parameters:
   - source: Input grid
   - scale_factor: How much to scale (e.g., 3 for 3×3 blocks)
   - mapping: Color mapping rules
   Example: "EXPAND: Replicate each input cell into 3×3 block"

4. FILL(regions=[list], pattern=[data], colors=[list])
   Parameters:
   - regions: Which regions/cells to fill (can be by location, color, pattern)
   - pattern: What to fill with (solid color, pattern, etc.)
   - colors: Colors used
   
   Examples:
   - "FILL: Fill all cells at even rows with color 2"
   - "FILL: Fill all cells in top half (rows 1-5) with color 0"
   - "FILL: Fill all background cells surrounding objects with color 5"
   - "FILL: Fill cells matching specific pattern with color 7"
   
   When to use:
   - Filling regions based on location (rows, cols, quadrants)
   - Filling all cells of a certain type
   - Pattern-based filling

5. MODIFY(objects=[list], change=[old→new])
   Parameters:
   - objects: What to modify (color, dimensions, position)
   - change: What changes (color, size, etc.)
   Example: "MODIFY: Replace object shape with new pattern"

5a. CROP/RESIZE(from_size=[H×W], to_size=[H×W], region=[coords])
   Parameters:
   - from_size: Current grid dimensions
   - to_size: Target grid dimensions
   - region: Which region to keep/extract (or how to resize)
   
   Examples:
   - "CROP: Remove rightmost 4 columns, changing grid from 3×7 to 3×3"
   - "CROP: Extract central 5×5 region from current 10×10 grid"
   - "RESIZE: Expand grid from 3×3 to 6×6, padding with background"
   
   When to use: When grid dimensions need to change

6. HIGHLIGHT_SUBPROCESS - 3-STEP PROCESS (marks what DOESN'T change):
   
   USE: PARTIAL TRANSFORMATIONS (temporary highlighting for anchors):
   - MUST use when ONLY PARTS of an object/grid are modified
   - MUST use when SOME cells stay fixed and SOME cells change
   - This is the ONLY WAY to handle partial object modifications!
   - 3-step process: HIGHLIGHT → TRANSFORM → UN-HIGHLIGHT
   - Highlights are TEMPORARY and removed by final step
   
   ⚠️⚠️⚠️ LOOK AT TRAINING IMAGES - WHICH PARTS WERE ANCHORED? ⚠️⚠️⚠️
   - Training highlighted TOP of object → Highlight TOP of YOUR test object
   - Training highlighted CORNERS → Highlight CORNERS of YOUR test object
   - Training highlighted LEFT SIDE → Highlight LEFT SIDE of YOUR test object
   - Training highlighted ANCHORS (cells that don't move) → Highlight ANCHORS in YOUR test
   
   **EXAMPLE - Analyzing Training Highlight:**
   Training image shows: 12-cell loop, 6 cells highlighted at top (stay fixed), 6 cells at bottom transform
   → Your test: Find YOUR loop, highlight TOP PORTION (anchors), transform BOTTOM PORTION
   
   Training image shows: 8-cell cluster, 4 corner cells highlighted (stay fixed), 4 center cells change color
   → Your test: Find YOUR cluster, highlight CORNERS (anchors), change CENTER cells
   
   ⚠️⚠️⚠️ CRITICAL SEQUENCING RULE ⚠️⚠️⚠️
   BEFORE using HIGHLIGHT, you MUST COPY objects from INPUT first!
   - Cannot highlight objects that aren't in current grid yet!
   - Sequence: COPY objects from INPUT → then HIGHLIGHT → then TRANSFORM → then UN-HIGHLIGHT
   
   ⚠️⚠️⚠️ DO NOT USE HIGHLIGHT FOR: ⚠️⚠️⚠️
   - Copying objects (use COPY instead!)
   - Entire object moves/transforms uniformly (use MOVE/MODIFY instead!)
   
   ⚠️⚠️⚠️ YOU MUST USE HIGHLIGHT FOR: ⚠️⚠️⚠️
   - ONLY part of an object is modified (some cells stay, some change)
   - SPECIFIC PORTION of object needs to stay fixed (anchors)
   - Any partial transformation where some cells are anchored
   
   ⚠️ BEFORE STEP 1: Look at training images to see WHICH PORTION is highlighted!
   ⚠️ Highlight the SAME RELATIVE PORTION of YOUR test object (anchors)
   
   STEP 1: HIGHLIGHT(anchors=[SPECIFIC_POSITIONS], temp_color=X)
   - Mark ONLY the SPECIFIC OBJECT cells that will STAY FIXED (not move/change)
   - These must be OBJECT cells (NOT background!) that are IDENTICAL in input and output
   - Parameters:
     * anchors: EXACT positions of SPECIFIC OBJECT cells that DON'T change
     * temp_color: Temporary color NOT in puzzle
   - ⚠️⚠️⚠️ CRITICAL: DO NOT HIGHLIGHT BACKGROUND CELLS! ⚠️⚠️⚠️
   - ⚠️ Background = the most prevalent/common color filling empty space (often but not always color 0)
   - ⚠️ ONLY HIGHLIGHT OBJECT CELLS (distinct objects, NOT background) THAT STAY FIXED!
   - ⚠️ Background cells are NEVER anchors - only actual object cells can be anchors!
   - Template: "HIGHLIGHT: Input→output diff shows N OBJECT anchor cells at [positions]. Mark using temp color T."
   - Example: "HIGHLIGHT: Diff shows OBJECT cells (12 cells at rows 1-5, cols 2-6) stay fixed. Mark as anchors using temp color 8."
   
   STEP 2: TRANSFORM(operation=[action], targets=NON_HIGHLIGHTED, keep_anchors=True)
   - Modify ONLY non-highlighted cells
   - Keep highlighted (anchor) cells fixed
   - Parameters:
     * operation: What to do (move, color change, replace, etc.)
     * targets: NON-highlighted cells
     * keep_anchors: Always True (anchors don't move)
   - Template: "TRANSFORM: [Action] NON-highlighted cells, keeping anchors fixed."
   
   STEP 3: UN_HIGHLIGHT(temp_color=X, restore_to=C)
   - Restore anchors to original colors
   - Parameters:
     * temp_color: Which temp color to remove
     * restore_to: Original color
   - Template: "UN-HIGHLIGHT: Restore temp color X to original color C."

7. NO-OP - No action (condition not met)

⚠️ DECISION TREE - WHICH ACTION TO USE:

Q1: Are objects in current grid already, or do I need to copy from INPUT?
→ NOT in grid yet: COPY from INPUT first (then proceed to Q2)
→ Already in grid: Continue to Q2...

Q2: Are ONLY PARTS of objects modified (some cells change, some stay fixed)?
→ YES: MUST use HIGHLIGHT subprocess after COPY (MANDATORY for partial modifications!)
→ NO: Continue...

Q3: Are SOME SPECIFIC cells identical in input/output while OTHER SPECIFIC cells change?
→ YES: MUST use HIGHLIGHT subprocess (3 steps: mark SPECIFIC anchors, transform non-anchors, un-highlight)
→ NO: Continue...

Q4: Does an entire object move/change uniformly (ALL cells of object affected equally)?
→ Move whole object: Use MOVE (no HIGHLIGHT needed)
→ Change ALL cells of object: Use MODIFY (no HIGHLIGHT needed)
→ Fill regions: Use FILL

⚠️ CRITICAL INSTRUCTIONS:
- Pattern defines steps - follow the sequence
- Right now you are on STEP {step_id}
- Execute ONLY what pattern says for Step {step_id}
- If conditional, adapt to what's present in THIS test example
- Test may have different colors/objects than training - adapt the pattern!

═══════════════════════════════════════════════════════════════════════════
PREVIOUS STEPS YOU'VE COMPLETED ON TEST:
═══════════════════════════════════════════════════════════════════════════

{prev_text}

═══════════════════════════════════════════════════════════════════════════
TEST INPUT GRID (what you're starting from):
═══════════════════════════════════════════════════════════════════════════

Size: {len(test_input)}×{len(test_input[0])}

{self._format_grid(test_input)}

⚠️ GOAL ANALYSIS:
- Look at the training examples to understand what the final transformation should achieve
- Training shows you WHAT to do to the test input
- Apply the same transformation logic to reach the correct output

═══════════════════════════════════════════════════════════════════════════
CURRENT TEST GRID STATE (your work in progress):
═══════════════════════════════════════════════════════════════════════════

{self._format_grid(current_grid)}

═══════════════════════════════════════════════════════════════════════════
EXECUTE PATTERN STEP {step_id}:
═══════════════════════════════════════════════════════════════════════════

Look at the LEARNED PATTERN above. Find what it says for "Step {step_id}".

BEFORE executing, FOLLOW TRAINING EXAMPLES:
1. ⚠️ Look at training booklet images - what did they do at this step?
2. ⚠️ What size was their grid at this step? Match that approach for test!
3. Look at TEST INPUT GRID - what objects exist?
4. Identify EACH distinct object by color
5. COUNT cells, determine EXACT dimensions, note positions
6. Apply the SAME transformation you see in training, adapted to test's specific objects/colors

Grid sizes:
- Test Input: {len(test_input)}×{len(test_input[0])}
- Current grid: {len(current_grid)}×{len(current_grid[0])}

⚠️ PATTERN USES GENERIC DESCRIPTIONS - YOU MUST ADAPT TO TEST:
- Pattern says "objects that transform" → Find which color transforms in THIS test!
- Pattern says "objects that stay fixed" → Find which colors stay fixed in THIS test!
- Pattern says "extract/condense from input" → Identify which region/objects to extract!
- Colors in test may differ from training - apply the SAME ROLE/TRANSFORMATION!

⚠️⚠️⚠️ MANDATORY OUTPUT FORMAT ⚠️⚠️⚠️

YOU MUST RESPOND WITH EXACTLY THIS FORMAT (NO EXCEPTIONS):

Description: [ACTION: what you're doing with specific objects in THIS test (color, dimensions, position)]

GRID:
[[row1],
 [row2],
 ...]

⚠️ THE WORD "GRID:" IS MANDATORY! DO NOT SKIP IT!
⚠️ YOU MUST INCLUDE BOTH "Description:" AND "GRID:" SECTIONS!
⚠️ If you don't include "GRID:", your response will be REJECTED!

CORRECT EXAMPLE:

Description: COPY: Copy color 4 loop (18 cells, rows 1-5, cols 1-9) FROM INPUT to output

GRID:
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 4, 4, 4, 4, 4, 4, 0, 0, 0],
 [0, 4, 0, 0, 0, 0, 0, 4, 0, 0],
 [0, 4, 0, 0, 0, 0, 0, 4, 4, 0],
 [0, 4, 4, 0, 0, 0, 0, 0, 4, 0],
 [0, 0, 0, 4, 4, 4, 4, 4, 4, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

More examples (BE CONCISE - 1-2 sentences for Description):
- "COPY: Copy color 4 loop (18 cells) FROM INPUT to output at same positions" - Same size
- "COPY: Copy entire INPUT grid to current (becomes 3×7)" - Start with input size
- "CROP: Remove rightmost 4 columns from current 3×7 grid, changing to 3×3"
- "COPY: Extract color 1 cells from left of INPUT (rows 0-2, cols 0-2) to output" - Direct extraction
- "FILL: Fill enclosed region (rows 3-5, cols 2-4) with color 4"
- "HIGHLIGHT: Diff shows object cells (12 cells, rows 1-5) stay fixed. Mark as anchors using temp color 8."
- "TRANSFORM: Replace NON-highlighted cells with the required pattern, keeping anchors fixed."
- "UN-HIGHLIGHT: Restore anchor cells to their original color."
- "MOVE: Move color 8 block (2×2, rows 10-11) down 6 rows to rows 16-17"
- "MODIFY: Replace color 2 cells (8 cells, rows 7-9) with 3×3 hollow square"
- "NO-OP: Pattern specifies object type not present in test; grid unchanged"

⚠️ REMINDER FOR HIGHLIGHT STEPS:
- ONLY highlight distinct object cells that stay fixed (NOT background cells)!
- Background = the most common/prevalent color filling empty space (identify from context)
- Background is NEVER an anchor - NEVER highlight background cells!
- If pattern says "highlight anchors", highlight OBJECT cells that don't move, NOT background!

⚠️⚠️⚠️ CRITICAL COLOR RULES ⚠️⚠️⚠️:
- This puzzle ONLY uses these colors: {sorted(valid_colors)}
- You MUST use ONLY these exact colors: {sorted(valid_colors)}
- NEVER invent or use any other color values
- For COPY/NO-OP steps, use the EXACT same color values from the current grid
- For HIGHLIGHT steps ONLY, you may use ONE additional temporary color NOT in {sorted(valid_colors)}
- Double-check every cell value before generating the grid

⚠️⚠️⚠️ CRITICAL GRID SIZE RULES ⚠️⚠️⚠️:
- Current grid size: {len(current_grid)}×{len(current_grid[0])}
- Test Input size: {len(test_input)}×{len(test_input[0])}
- ⚠️ LOOK AT TRAINING: What size did their grids have at each step?
- ⚠️ MATCH THAT APPROACH: If training changed size, you change size too
- Your output grid size can change during transformation - that's OK!
- Use CROP/RESIZE action when you need to change grid dimensions (like training did)
"""
            
            print(f"  🔧 Debug: Sending {len(images)} images (training booklets + test state)")
            print(f"  Prompt length: {len(prompt)} characters")
            
            # Retry up to 3 times
            max_retries = 3
            success = False
            
            for attempt in range(1, max_retries + 1):
                try:
                    if attempt > 1:
                        print(f"  🔄 Retry attempt {attempt}/{max_retries}")
                    
                    response = self.call_api(prompt, images)
                    
                    # Parse
                    description = self._parse_description(response)
                    grid = self._parse_grid(response)
                    
                    if not grid:
                        print(f"  ❌ Attempt {attempt}: Failed to parse grid from response")
                        print(f"  Description parsed: {description[:150] if description else 'None'}")
                        if attempt < max_retries:
                            print(f"  Response preview: {response[:400]}...")
                            # Check if GRID: keyword exists
                            if "GRID:" not in response and "Grid:" not in response:
                                print(f"  ⚠️ Response missing 'GRID:' section! Adding reminder to retry...")
                                # Add a special follow-up prompt to emphasize GRID requirement
                                prompt += f"""

⚠️⚠️⚠️ CRITICAL ERROR IN PREVIOUS ATTEMPT ⚠️⚠️⚠️

Your previous response was MISSING the required "GRID:" section!

YOU MUST include BOTH:
1. Description: [your action description]
2. GRID: [the actual grid array]

Example of CORRECT format:

Description: COPY: Copy color 4 loop (18 cells) FROM INPUT to output

GRID:
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 4, 4, 4, 4, 4, 4, 0, 0, 0],
 [0, 4, 0, 0, 0, 0, 0, 4, 0, 0]]

DO NOT FORGET THE "GRID:" LINE AND THE GRID ARRAY!
"""
                            continue
                        else:
                            print(f"  Full response:")
                            print(f"  {response}")
                            print(f"  ⚠️ Step {step_id} generation failed after {max_retries} attempts, stopping")
                            break
                    
                    # Allow temporary highlight color only during anchor steps
                    allow_temp = description.startswith("HIGHLIGHT:")
                    if not self._validate_colors(grid, valid_colors, allow_temp_color=allow_temp):
                        print(f"  ❌ Step {step_id}: Generated grid uses invalid colors" + (f" on attempt {attempt}" if attempt > 1 else ""))
                        continue
                    
                    if description.startswith("HIGHLIGHT:"):
                        print(f"  ✓ Step {step_id} generated successfully (HIGHLIGHT step)" + (f" on attempt {attempt}" if attempt > 1 else ""))
                    else:
                        print(f"  ✓ Step {step_id} generated successfully" + (f" on attempt {attempt}" if attempt > 1 else ""))
                    
                    current_grid = grid
                    previous_images.append(grid_to_image(grid, 30))
                    previous_descriptions.append(description)
                    
                    all_step_results.append({
                        "step_num": step_id,  # Use step_id which may include sub-steps like "3.1"
                        "description": description,
                        "grid": grid,
                        "success": True
                    })
                    
                    success = True
                    break  # Exit retry loop on success
                        
                except Exception as e:
                    print(f"  ❌ Attempt {attempt}: Error: {e}")
                    if attempt < max_retries:
                        import traceback
                        print(f"  Traceback (brief):")
                        traceback.print_exc()
                        continue
                    else:
                        import traceback
                        print(f"  Traceback:")
                        traceback.print_exc()
                        print(f"  ⚠️ Stopping at step {step_id} after {max_retries} attempts")
                        break
            
            # If all retries failed, stop the entire process
            if not success:
                break
        
        # All pattern steps executed
        if all_step_results:
            print(f"\n✅ Executed all {pattern_steps} steps from the pattern")
            
            # Final validation: Check for temporary colors in final grid
            final_grid = all_step_results[-1]['grid']
            final_desc = all_step_results[-1]['description']
            if final_grid:
                import numpy as np
                unique_colors = set(np.array(final_grid).flatten())
                invalid_colors = unique_colors - valid_colors
                
                if invalid_colors:
                    print(f"\n⚠️⚠️⚠️ WARNING: TEMPORARY COLORS DETECTED IN FINAL TEST OUTPUT! ⚠️⚠️⚠️")
                    print(f"  Valid colors: {sorted(valid_colors)}")
                    print(f"  Found invalid colors: {sorted(invalid_colors)}")
                    print(f"  ❌ All HIGHLIGHT temporary colors should be removed by the final step!")
                    print(f"  ❌ Final output must ONLY contain valid puzzle colors!")
                else:
                    print(f"\n✓ Final test output validation: NO temporary colors - only valid colors {sorted(valid_colors)}")
        
        # Save test results
        self._save_test_results(puzzle_id, test_num, analysis, pattern_summary, 
                                test_input, test_output, all_step_results, booklets)
        
        return all_step_results
    
    def _save_test_results(self, puzzle_id: str, test_num: int, analysis: str, 
                          pattern_summary: str, test_input: List[List[int]], 
                          test_output: List[List[int]], steps: List[Dict], 
                          training_booklets: List[Dict]):
        """Save test results"""
        output_dir = Path("visual_step_results") / puzzle_id / f"test_{test_num:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate final accuracy if ground truth exists
        final_accuracy = None
        if test_output and steps:
            final_grid = steps[-1]['grid']
            if len(final_grid) == len(test_output) and len(final_grid[0]) == len(test_output[0]):
                import numpy as np
                matches = np.array(final_grid) == np.array(test_output)
                final_accuracy = matches.sum() / matches.size
        
        data = {
            "puzzle_id": puzzle_id,
            "test_num": test_num,
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "phase1_analysis": analysis,
            "pattern_summary": pattern_summary,
            "test_input_grid": test_input,
            "test_output_grid": test_output,
            "num_training_booklets_used": len(training_booklets),
            "steps": steps,
            "final_accuracy": final_accuracy
        }
        
        with open(output_dir / "results.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save images
        input_img = grid_to_image(test_input, 30)
        input_img.save(output_dir / "test_input.png")
        
        if test_output:
            output_img = grid_to_image(test_output, 30)
            output_img.save(output_dir / "test_output.png")
        
        for step in steps:
            step_id = step['step_num']  # May be string like "3.1"
            img = grid_to_image(step['grid'], 30)
            # Handle both integer and sub-step format
            step_filename = str(step_id).replace('.', '_')
            img.save(output_dir / f"step_{step_filename}.png")
        
        print(f"\n{'='*80}")
        print(f"SAVED TO: {output_dir}/")
        print(f"  results.json")
        print(f"  test_input.png")
        if test_output:
            print(f"  test_output.png")
            if final_accuracy is not None:
                print(f"\n  FINAL ACCURACY: {final_accuracy:.1%}")
        print(f"  step_XX.png ({len(steps)} steps)")
        print(f"{'='*80}\n")
    
    def _verify_highlight_grid(self, description: str, generated_grid, current_grid):
        """
        Verify that the HIGHLIGHT grid actually highlights the cells mentioned in the description.
        Returns (success, message)
        """
        import re
        import numpy as np
        
        gen_array = np.array(generated_grid)
        curr_array = np.array(current_grid)
        
        # Find which temporary color is being used (should be mentioned in description)
        temp_color_match = re.search(r'temporary\s+color\s+(\d+)', description, re.IGNORECASE)
        if not temp_color_match:
            temp_color_match = re.search(r'color\s+(\d+)\s+\(temporary', description, re.IGNORECASE)
        temp_color = int(temp_color_match.group(1)) if temp_color_match else None
        
        # Find cells that are the temporary color in generated grid
        if temp_color is not None:
            highlighted_mask = gen_array == temp_color
            highlighted_positions_0based = set(tuple(pos) for pos in np.argwhere(highlighted_mask).tolist())
        else:
            # Fallback: find cells that changed
            changed_mask = gen_array != curr_array
            highlighted_positions_0based = set(tuple(pos) for pos in np.argwhere(changed_mask).tolist())
        
        # Extract positions from description (1-based coordinates)
        expected_positions_1based = set()
        
        # Pattern 1: "rows 1-5, cols 2-6"
        pattern1 = r'rows?\s+(\d+)-(\d+),?\s+cols?\s+(\d+)-(\d+)'
        for match in re.finditer(pattern1, description, re.IGNORECASE):
            row_start, row_end, col_start, col_end = map(int, match.groups())
            for r in range(row_start, row_end + 1):
                for c in range(col_start, col_end + 1):
                    expected_positions_1based.add((r, c))
        
        # Pattern 2: "row 7 cols 2-4" (but not already matched by pattern1)
        pattern2 = r'row\s+(\d+),?\s+cols?\s+(\d+)-(\d+)'
        for match in re.finditer(pattern2, description, re.IGNORECASE):
            # Check if this was already matched by pattern1 (rows X-Y format)
            match_str = match.group(0)
            if not re.search(r'rows?\s+\d+-\d+', match_str, re.IGNORECASE):
                row, col_start, col_end = map(int, match.groups())
                for c in range(col_start, col_end + 1):
                    expected_positions_1based.add((row, c))
        
        # Pattern 3: "row 8 cols 2 and 5" or "row 8 col 2 and col 5"
        pattern3 = r'row\s+(\d+),?\s+cols?\s+(\d+)\s+and\s+(\d+)'
        for match in re.finditer(pattern3, description, re.IGNORECASE):
            row, col1, col2 = map(int, match.groups())
            expected_positions_1based.add((row, col1))
            expected_positions_1based.add((row, col2))
        
        # Pattern 4: "row 8 col 5" (single cell)
        pattern4 = r'row\s+(\d+),?\s+col\s+(\d+)'
        for match in re.finditer(pattern4, description, re.IGNORECASE):
            # Make sure this wasn't part of "cols 2 and 5" pattern
            match_str = match.group(0)
            if 'and' not in match_str.lower():
                row, col = map(int, match.groups())
                expected_positions_1based.add((row, col))
        
        # Convert 1-based to 0-based coordinates
        expected_positions_0based = set((r-1, c-1) for r, c in expected_positions_1based)
        
        # Compare
        if expected_positions_0based:
            missing = expected_positions_0based - highlighted_positions_0based
            extra = highlighted_positions_0based - expected_positions_0based
            
            if missing or extra:
                msg = []
                if missing:
                    missing_1based = sorted([(r+1, c+1) for r, c in missing])
                    msg.append(f"Missing highlights at (1-based): {missing_1based[:10]}")
                if extra:
                    extra_1based = sorted([(r+1, c+1) for r, c in extra])
                    msg.append(f"Extra highlights at (1-based): {extra_1based[:10]}")
                
                # Also show what color was found vs expected
                if temp_color is not None:
                    wrong_colors = []
                    for r, c in missing:
                        if r < gen_array.shape[0] and c < gen_array.shape[1]:
                            wrong_colors.append(f"({r+1},{c+1})={gen_array[r,c]}")
                    if wrong_colors:
                        msg.append(f"Found colors: {wrong_colors[:5]}")
                
                return False, "; ".join(msg)
            
            return True, f"All {len(expected_positions_0based)} cells correctly highlighted with temporary color {temp_color}"
        
        # If we couldn't parse positions, check if something changed
        if len(highlighted_positions_0based) == 0:
            return False, "No cells were highlighted (grid identical to current or no temporary color found)"
        
        return True, f"{len(highlighted_positions_0based)} cells changed/highlighted (could not parse exact positions from description)"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Solve test examples using training booklets")
    parser.add_argument('--puzzle', required=True, help='Puzzle ID')
    parser.add_argument('--test-num', type=int, default=1, help='Test example number (default: 1)')
    parser.add_argument('--model', default='gpt-5-mini', help='Model to use')
    args = parser.parse_args()
    
    solver = TestStepGenerator(args.model)
    solver.solve_test_example(args.puzzle, args.test_num)
    
    print("\n✅ Test solving complete!")
    print(f"\nView results:")
    print(f"  streamlit run scripts/view_visual_steps.py")


if __name__ == "__main__":
    main()

