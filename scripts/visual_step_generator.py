#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual Step Generator - Final Clean Version

Flow:
1. Phase 1: Analyze all inputs/outputs → transformation rule + grid size
2. Step 0: Model generates empty grid based on analysis
3. Steps 1-N: Model sees (analysis + output image + all previous + current grid) → generates next
4. 3 attempts per step, if all fail → use ground truth for next step
5. Save all attempts + ground truth for Streamlit UI

Usage:
    python scripts/visual_step_generator.py --puzzle 05f2a901
"""

import os
import sys
import json
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from io import BytesIO

# Set UTF-8 encoding for stdout/stderr on Windows
if sys.platform == 'win32':
    import codecs
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

try:
    from openai import OpenAI
    from PIL import Image
    import numpy as np
    from arc_visualizer import grid_to_image
except ImportError as e:
    print(f"❌ Install dependencies or check imports: {e}")
    sys.exit(1)


class VisualStepGenerator:
    
    def __init__(self, model: str = "gpt-5-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        print(f"✓ Initialized with {model}")
    
    def call_api(self, prompt: str, images: List[Image.Image] = None) -> str:
        """Call OpenAI API"""
        try:
            content = [{"type": "text", "text": prompt}]
            
            if images:
                for img in images:
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode()
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    })
            
            print(f"  🔧 Debug: Sending {len(images) if images else 0} images, prompt length: {len(prompt)} chars")
            
            if "gpt-5" in self.model:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_completion_tokens=8000
                )
            else:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=4000
                )
            
            response_text = resp.choices[0].message.content
            
            if not response_text or response_text.strip() == "":
                print(f"  ⚠️ API returned empty response!")
                print(f"  Response object: {resp}")
                if hasattr(resp.choices[0], 'finish_reason'):
                    print(f"  Finish reason: {resp.choices[0].finish_reason}")
                return ""
            
            return response_text
            
        except Exception as e:
            print(f"  ❌ API Error: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""
    
    def load_ground_truth(self, puzzle_id: str) -> Dict:
        """Load your manual booklet from visual_traces"""
        base = Path(f"visual_traces/{puzzle_id}")
        
        # Load steps from training_01
        steps = []
        for step_dir in sorted((base / "training_01").glob("step_*")):
            with open(step_dir / "step.json") as f:
                step_data = json.load(f)
                step_data['image_path'] = str(step_dir / "grid.png")
                steps.append(step_data)
        
        return {"puzzle_id": puzzle_id, "steps": steps}
    
    def load_arc_puzzle(self, puzzle_id: str) -> Dict:
        """Load from ARC dataset"""
        paths = [
            Path("../saturn-arc/ARC-AGI-2/ARC-AGI-2/data"),
            Path("C:/Users/Isabe/New folder (3)/saturn-arc/ARC-AGI-2/ARC-AGI-2/data"),
        ]
        
        for p in paths:
            for subset in ["training", "evaluation"]:
                f = p / subset / f"{puzzle_id}.json"
                if f.exists():
                    return json.load(open(f))
        
        raise FileNotFoundError(f"Puzzle {puzzle_id} not found")
    
    def phase1_analyze(self, arc_puzzle: Dict, example_to_use: Dict, puzzle_id: str) -> str:
        """Phase 1: Model analyzes ALL training examples"""
        print("\n" + "="*80)
        print("PHASE 1: ANALYZING ALL TRAINING EXAMPLES")
        print("="*80 + "\n")
        
        train = arc_puzzle['train']
        
        # First, extract which colors are actually in THIS puzzle
        actual_colors = set()
        for ex in train:
            for row in ex['input'] + ex['output']:
                actual_colors.update(row)
        
        print(f"Colors present in this puzzle: {sorted(actual_colors)}")
        print(f"Showing first example data for verification...\n")
        
        # Detect varying step counts across training examples
        step_counts = self._get_step_counts_per_training_example(puzzle_id)
        print(f"Step counts per training example: {step_counts}")
        has_variable_steps = len(set(step_counts)) > 1
        if has_variable_steps:
            print(f"⚠️ Variable step counts detected! Some examples need more steps than others.\n")
        
        # Create images for ALL examples (input, output, AND side-by-side comparison)
        images = []
        from PIL import Image
        for ex in train:
            input_img = grid_to_image(ex['input'], 30)
            output_img = grid_to_image(ex['output'], 30)
            
            # Create side-by-side comparison for visual diff
            width1, height1 = input_img.size
            width2, height2 = output_img.size
            max_height = max(height1, height2)
            
            # Combined image with arrow between them
            arrow_width = 40
            combined = Image.new('RGB', (width1 + arrow_width + width2, max_height), color=(255, 255, 255))
            combined.paste(input_img, (0, 0))
            combined.paste(output_img, (width1 + arrow_width, 0))
            
            images.append(input_img)   # Input alone
            images.append(output_img)  # Output alone  
            images.append(combined)    # Side-by-side comparison
        
        # Build grid size info
        grid_size_info = []
        for i, ex in enumerate(train, 1):
            in_h, in_w = len(ex['input']), len(ex['input'][0])
            out_h, out_w = len(ex['output']), len(ex['output'][0])
            grid_size_info.append(f"Example {i}: Input {in_h}×{in_w} → Output {out_h}×{out_w}")
        
        grid_size_text = "\n".join(grid_size_info)
        
        # Build detailed image reference with grid data
        image_ref_lines = []
        grid_data_lines = []
        image_idx = 1
        for i, ex in enumerate(train, 1):
            image_ref_lines.append(f"Example {i}:")
            image_ref_lines.append(f"  - Image {image_idx}: INPUT (alone)")
            image_ref_lines.append(f"  - Image {image_idx+1}: OUTPUT (alone)")
            image_ref_lines.append(f"  - Image {image_idx+2}: INPUT→OUTPUT (side-by-side comparison)")
            image_idx += 3
            
            # Show grid data for detailed inspection
            grid_data_lines.append(f"\nExample {i} GRID DATA:")
            grid_data_lines.append(f"  INPUT ({len(ex['input'])}×{len(ex['input'][0])}):")
            for row_idx, row in enumerate(ex['input']):
                grid_data_lines.append(f"    Row {row_idx}: {row}")
            grid_data_lines.append(f"  OUTPUT ({len(ex['output'])}×{len(ex['output'][0])}):")
            for row_idx, row in enumerate(ex['output']):
                grid_data_lines.append(f"    Row {row_idx}: {row}")
        
        image_ref_text = "\n".join(image_ref_lines)
        grid_data_text = "\n".join(grid_data_lines)
        
        # Build cell-by-cell change analysis
        change_analysis_lines = []
        for i, ex in enumerate(train, 1):
            changes = self._analyze_cell_changes(ex['input'], ex['output'])
            if 'error' not in changes:
                change_analysis_lines.append(f"\nExample {i} CELL-BY-CELL CHANGES:")
                change_analysis_lines.append(f"  Total cells changed: {changes['total_changes']}")
                change_analysis_lines.append(f"  Cells stayed same: {changes['stayed_same']}")
                if changes['color_changes']:
                    change_analysis_lines.append(f"  Color transformations:")
                    for from_color, to_color, count in changes['color_changes']:
                        change_analysis_lines.append(f"    - Color {from_color} → Color {to_color}: {count} cells")
                if changes['sample_changes']:
                    change_analysis_lines.append(f"  Sample specific changes (first 15):")
                    for cell_change in changes['sample_changes']:
                        change_analysis_lines.append(f"    - Cell ({cell_change['row']},{cell_change['col']}): Color {cell_change['from_color']} → Color {cell_change['to_color']}")
        
        change_analysis_text = "\n".join(change_analysis_lines)
        
        prompt = f"""You are analyzing an ARC puzzle with {len(train)} training examples.

IMAGES PROVIDED (LOOK CAREFULLY AT SIDE-BY-SIDE COMPARISONS):
═══════════════════════════════════════════════════════════════════════════
{image_ref_text}

⚠️ USE THE SIDE-BY-SIDE COMPARISONS to see exactly what changes from input to output!
Look at the visual differences carefully - what cells change color? What stays the same?

═══════════════════════════════════════════════════════════════════════════
GRID SIZES (VERIFY THESE - DO NOT GUESS):
═══════════════════════════════════════════════════════════════════════════

{grid_size_text}

⚠️ Use these EXACT sizes in your analysis! Count them if needed.

═══════════════════════════════════════════════════════════════════════════
ARC COLOR PALETTE (what you SEE in images):
═══════════════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════════════
COLOR VERIFICATION (CRITICAL - READ THIS FIRST!):
═══════════════════════════════════════════════════════════════════════════

This specific puzzle uses ONLY these colors: {sorted(actual_colors)}

Here's sample data from first training example INPUT to verify:
First 3 rows of grid data:
{chr(10).join([str(row) for row in example_to_use['input'][:3]])}

ARC COLOR PALETTE (what each number looks like):
- Color 0 = Black (background)
- Color 1 = Blue  
- Color 2 = Red
- Color 3 = Green
- Color 4 = Yellow
- Color 5 = Orange
- Color 6 = Magenta/Pink
- Color 7 = Light Blue/Cyan
- Color 8 = Dark Red/Maroon
- Color 9 = Purple

FOR THIS PUZZLE, you will ONLY see colors: {sorted(actual_colors)}
Match what you SEE in the images to these specific numbers!

Example: If you see a maroon/dark red object, check the grid data above - it's color 8!
If you see a light blue object, check the data - it's color 7!

CRITICAL: Throughout your ENTIRE analysis, use ONLY colors from {sorted(actual_colors)}!
Do NOT mention color numbers that aren't in this set!

═══════════════════════════════════════════════════════════════════════════
DETAILED GRID DATA (Use this to see EXACT cell-by-cell changes):
═══════════════════════════════════════════════════════════════════════════

{grid_data_text}

⚠️ COMPARE INPUT vs OUTPUT arrays carefully!
- Which cells stay the same?
- Which cells change color?
- Are new cells added?
- What patterns do you see?

═══════════════════════════════════════════════════════════════════════════
CELL-BY-CELL CHANGE SUMMARY (PAY ATTENTION TO SMALL DIFFERENCES!):
═══════════════════════════════════════════════════════════════════════════

{change_analysis_text}

⚠️ CRITICAL - ANALYZE THESE CHANGES CAREFULLY:
- Are ALL cells of a certain color changing? Or only SOME?
- If only some cells change, what determines WHICH ones? (position? neighbors? shape?)
- What are the exact color transformations? (Color X → Color Y)
- Do ANY cells stay the same color?
- Is the transformation position-dependent or rule-based?

═══════════════════════════════════════════════════════════════════════════
STEP 1: ANALYZE EACH EXAMPLE INDIVIDUALLY
═══════════════════════════════════════════════════════════════════════════

For EACH example, look at the images AND grid data carefully and explain WHY:

Example 1 (Images 1→2):
- What SPECIFIC objects are in the INPUT? (list by color, position, dimensions: "color X object, NxM cells, at rows A-B, cols C-D")
- What SPECIFIC objects are in the OUTPUT? (list by color, position, dimensions)
- What CHANGED from input to output:
  * Did any objects MOVE? (which colors? from where to where? by how many rows/cols?)
  * Did any objects change COLOR? (which cells? old→new colors?)
  * Did any objects change SIZE/SHAPE? (which ones? how?)
  * Were any objects ADDED or REMOVED?
- What STAYED THE SAME? (dimensions? background? certain objects? specific positions?)
- WHY did it change this way? What's the purpose/rule? (e.g., "grid expanded 3× BECAUSE each input cell needs to map to a 3×3 region")

Example 2 (Images 3→4):
[Same questions - be VERY specific about movements and color changes!]
- If objects moved, are they moving by the SAME amount as Example 1?
- If colors changed, are they changing the SAME way as Example 1?
- WHY is this transformation happening? What rule explains it?

Example 3 (Images 5→6):
[Same questions - look for CONSISTENCY in movements/changes!]
- Are movements consistent? (same direction? same distance?)
- Are color changes consistent? (same old→new mapping?)
- WHY does this follow the same pattern? What's consistent?

═══════════════════════════════════════════════════════════════════════════
STEP 2: COMPARE ACROSS EXAMPLES (and explain WHY)
═══════════════════════════════════════════════════════════════════════════

INPUTS - What's common vs different:
- SAME in ALL inputs: Which colors appear? How many objects of each color?
- DIFFERS between inputs: Grid sizes? Object positions?
- WHY are some things consistent? (e.g., "all inputs have same structure BECAUSE the rule applies to any configuration")

OUTPUTS - What's common vs different:
- SAME in ALL outputs: Which colors? What happened to each color?
- DIFFERS: Positions? Sizes?
- WHY is the output pattern consistent? (e.g., "all outputs are 3× larger BECAUSE the rule is about scaling")

═══════════════════════════════════════════════════════════════════════════
STEP 3: IDENTIFY THE TRANSFORMATION PATTERNS (FOCUS ON MOVEMENTS & CHANGES)
═══════════════════════════════════════════════════════════════════════════

⚠️ CRITICAL: Look at the CELL-BY-CELL CHANGES above to identify EXACT patterns!

For EACH color that appears in the puzzles, answer:
- Color 0 (background): Does it stay? Change? WHY?
- Color 1: What happens to objects of this color?
  * Does it MOVE? (if yes: direction? distance? ALWAYS the same?)
  * Does it change COLOR? (if yes: to what color? ALWAYS the same?)
  * Does it change SIZE/SHAPE? (if yes: how? ALWAYS the same?)
  * WHY does this specific color get this transformation?
- Color 2: [Same detailed questions]
- Color 3: [Same detailed questions]
... (for each color present)

MOVEMENT PATTERNS (if any objects move):
- Which colors move? Which don't?
- Direction of movement? (up/down/left/right? by how many cells?)
- Is movement consistent across ALL examples?
- WHY do these specific cells move? (based on position? color? neighbor? something else?)

COLOR CHANGE PATTERNS (if any colors change):
- Which colors change? (exact mapping: Color X → Color Y)
- Are ALL cells of Color X changing, or only SOME? (if only some, WHICH ones?)
- Is the color change consistent across ALL examples?
- WHY do these specific cells change color? (position-dependent? rule-based?)

SPATIAL PATTERNS:
- Do changes depend on POSITION? (e.g., only top row? only specific region?)
- Do changes depend on NEIGHBORS? (e.g., cells next to color X?)
- Do changes depend on SHAPE? (e.g., only certain patterns?)
- WHY is this spatial pattern important?

CAUSAL REASONING:
- What's the PURPOSE of each transformation? (e.g., "expand to create space for tiling")
- What's the GOAL? (e.g., "create a repeating pattern", "establish spatial relationships")
- What's the LOGIC? (e.g., "each input cell becomes a 3×3 region to enable pattern replication")

═══════════════════════════════════════════════════════════════════════════
STEP 4: GRID SIZE PATTERN (and explain WHY)
═══════════════════════════════════════════════════════════════════════════

GRID SIZES (already provided above):
{grid_size_text}

Pattern: Describe the relationship between input and output sizes
- Are outputs the SAME size as inputs?
- Are outputs LARGER? By what factor or formula?
- Are outputs SMALLER? By what factor?
- WHY is this size relationship necessary? (e.g., "output is 3× larger BECAUSE each input cell needs to map to a 3×3 region for tiling")
- What's the PURPOSE of this sizing? (e.g., "create space for replication", "maintain aspect ratio")

⚠️ DO NOT GUESS - use the exact sizes shown above!

═══════════════════════════════════════════════════════════════════════════
STEP 5: SUMMARIZE THE GENERAL TRANSFORMATION PATTERN (with WHY)
═══════════════════════════════════════════════════════════════════════════

REVIEW the images one more time:
- Look at all 6 images again
- Verify your color identifications (remember: this puzzle ONLY uses {sorted(actual_colors)})
- Check your observations are true for ALL examples

GENERAL PATTERN (3-5 sentences with CAUSAL REASONING):
Describe what happens AND WHY:
- What happens to objects? WHY does this transformation occur?
- Which colors are involved? WHY these specific colors?
- What changes vs what stays the same? WHY?
- What's the GOAL/PURPOSE of this transformation? (e.g., "create tiled pattern", "establish adjacency")
- What's the UNDERLYING RULE? (e.g., "each input cell maps to a 3×3 output region BECAUSE the pattern needs to be replicated at scale")

Example with WHY reasoning:
"The transformation expands the grid by 3× in each dimension BECAUSE each input cell needs to map to a 3×3 region. 
First, the input pattern is scaled up (each cell becomes a 3×3 block) to show the scaling relationship. 
Then, each 3×3 region is filled with the complete input pattern to create a tiled replication. 
The PURPOSE is to create a 3×3 array of the original pattern."

Use "color X" format and describe PATTERNS with their PURPOSE, not just what happens.
This should be general enough to apply to ANY example, not just these 3.

CRITICAL: Use ONLY colors from {sorted(actual_colors)} throughout!

═══════════════════════════════════════════════════════════════════════════
STEP 6: VARIABLE STEP COUNTS (IF APPLICABLE)
═══════════════════════════════════════════════════════════════════════════

{self._build_variable_steps_section(step_counts, has_variable_steps)}

Provide complete analysis for all {6 if has_variable_steps else 5} steps above.
"""
        
        print(f"Sending {len(images)} images to {self.model}...")
        analysis = self.call_api(prompt, images)
        
        print("\nANALYSIS RESULT:")
        print("="*80)
        print(analysis)
        print("="*80 + "\n")
        
        return analysis
    
    def _format_grid(self, grid):
        return '\n'.join(['[' + ','.join(str(c) for c in row) + ']' for row in grid])
    
    def _analyze_cell_changes(self, input_grid, output_grid):
        """Analyze cell-by-cell changes between input and output"""
        if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
            return {"error": "Different dimensions"}
        
        changes = []
        color_change_counts = {}
        stayed_same = 0
        
        for r in range(len(input_grid)):
            for c in range(len(input_grid[0])):
                in_color = input_grid[r][c]
                out_color = output_grid[r][c]
                
                if in_color != out_color:
                    changes.append({
                        'row': r,
                        'col': c,
                        'from_color': in_color,
                        'to_color': out_color
                    })
                    key = (in_color, out_color)
                    color_change_counts[key] = color_change_counts.get(key, 0) + 1
                else:
                    stayed_same += 1
        
        # Sort color changes by count (most common first)
        color_changes = sorted(
            [(from_c, to_c, count) for (from_c, to_c), count in color_change_counts.items()],
            key=lambda x: x[2],
            reverse=True
        )
        
        return {
            'total_changes': len(changes),
            'stayed_same': stayed_same,
            'color_changes': color_changes,
            'sample_changes': changes[:15]  # First 15 for detailed view
        }
    
    def _print_grid_diff(self, current, generated):
        """Print a visual diff showing which cells changed (for HIGHLIGHT debugging)"""
        import numpy as np
        curr_array = np.array(current)
        gen_array = np.array(generated)
        
        if curr_array.shape != gen_array.shape:
            print(f"      Grids have different shapes: {curr_array.shape} vs {gen_array.shape}")
            return
        
        changed = curr_array != gen_array
        changed_positions = np.argwhere(changed)
        
        if len(changed_positions) == 0:
            print(f"      No cells changed (grids are identical)")
            return
        
        print(f"      Changed cells ({len(changed_positions)} total):")
        for idx, (r, c) in enumerate(changed_positions[:20]):  # Show first 20
            old_color = curr_array[r, c]
            new_color = gen_array[r, c]
            print(f"        Row {r+1}, Col {c+1}: {old_color} → {new_color}")
        if len(changed_positions) > 20:
            print(f"        ... and {len(changed_positions) - 20} more")
    
    def _format_grid_diff(self, current, expected):
        """Show cell-by-cell differences between current grid and expected output"""
        if len(current) != len(expected) or len(current[0]) != len(expected[0]):
            return "Grids have different dimensions - cannot compare cell-by-cell"
        
        diff_lines = []
        total_cells = len(current) * len(current[0])
        diff_count = 0
        
        for r in range(len(current)):
            for c in range(len(current[0])):
                if current[r][c] != expected[r][c]:
                    diff_count += 1
                    diff_lines.append(f"  Cell ({r},{c}): current={current[r][c]}, expected={expected[r][c]}")
        
        if diff_count == 0:
            return "✓ Current grid MATCHES expected output! (all cells correct)"
        
        match_count = total_cells - diff_count
        accuracy = (match_count / total_cells) * 100
        
        summary = f"Accuracy: {match_count}/{total_cells} cells correct ({accuracy:.1f}%)\n"
        summary += f"Cells that still need to change: {diff_count}\n\n"
        
        if diff_count <= 20:
            # Show all differences if there aren't too many
            summary += "Differences:\n" + '\n'.join(diff_lines[:20])
        else:
            # Show first 20 if there are many
            summary += "First 20 differences:\n" + '\n'.join(diff_lines[:20])
            summary += f"\n... and {diff_count - 20} more cells differ"
        
        return summary
    
    def _build_variable_steps_section(self, step_counts: List[int], has_variable_steps: bool) -> str:
        """Build the variable steps section for phase 1 prompt"""
        if not has_variable_steps:
            return "Not applicable - all examples have the same number of steps."
        
        return f"""⚠️ IMPORTANT: Different training examples require different numbers of steps!

Step counts by example: {', '.join([f'Ex{i+1}: {count} steps' for i, count in enumerate(step_counts)])}

WHY THIS HAPPENS:
Some examples have MORE objects to process than others. The SAME logical steps apply to ALL examples, 
but some examples require REPEATING certain steps for additional objects.

EXAMPLE PATTERN:
- Example 1 (3 objects): Step 1: copy object 1, Step 2: copy object 2, Step 3: copy object 3
- Example 2 (2 objects): Step 1: copy object 1, Step 2: copy object 2, Step 3: do nothing (no object 3)
- Example 3 (4 objects): Step 1: copy object 1, Step 2: copy object 2, Step 3: copy object 3, Step 4: copy object 4

HOW TO HANDLE IN YOUR STEPS:
When generating steps, use CONDITIONAL LANGUAGE for steps that might not apply to all examples:
- ✅ "COPY: Copy the FIRST color 8 object from input to output"
- ✅ "COPY: Copy ANY REMAINING color 8 objects from input to output (if they exist, otherwise do nothing)"
- ✅ "MOVE: Move the FIRST color 2 object down 6 rows"
- ✅ "MOVE: Move ANY ADDITIONAL color 2 objects down 6 rows (if they exist, otherwise do nothing)"

This allows the same step sequence to work across ALL examples, even when some have fewer objects!"""
    
    def step0_generate_empty_grid(self, analysis: str, input_grid: List[List[int]], expected_output_grid: List[List[int]]) -> List[List[int]]:
        """Step 0: Model generates empty grid based on analysis"""
        print("="*80)
        print("STEP 0: GENERATE EMPTY OUTPUT GRID")
        print("="*80 + "\n")
        
        expected_h = len(expected_output_grid)
        expected_w = len(expected_output_grid[0])
        
        input_h = len(input_grid)
        input_w = len(input_grid[0])
        
        prompt = f"""Your Phase 1 analysis determined the grid size rule:

{analysis[:800]}

NOW: Generate the empty output grid for this specific example.

═══════════════════════════════════════════════════════════════════════════
INPUT GRID SIZE:
═══════════════════════════════════════════════════════════════════════════

Input: {input_h} rows × {input_w} columns ({input_h}×{input_w})

═══════════════════════════════════════════════════════════════════════════
EXPECTED OUTPUT GRID SIZE (from your Phase 1 analysis):
═══════════════════════════════════════════════════════════════════════════

The expected output for this example is: {expected_h} rows × {expected_w} columns ({expected_h}×{expected_w})

This matches your grid size rule because:
- Input is {input_h}×{input_w}
- Output is {expected_h}×{expected_w}
- Relationship: {expected_h}/{input_h} = {expected_h//input_h if input_h > 0 and expected_h % input_h == 0 else 'custom'} (height), {expected_w}/{input_w} = {expected_w//input_w if input_w > 0 and expected_w % input_w == 0 else 'custom'} (width)

═══════════════════════════════════════════════════════════════════════════
TASK: Generate initial empty grid
═══════════════════════════════════════════════════════════════════════════

⚠️⚠️⚠️ CRITICAL RULE FOR STARTING GRID SIZE ⚠️⚠️⚠️

IF INPUT LARGER THAN OUTPUT ({input_h}×{input_w} > {expected_h}×{expected_w}):
→ YOU MUST START WITH INPUT SIZE: {input_h}×{input_w}
→ Later steps will CROP/RESIZE to output size
→ Generate {input_h} rows, each with {input_w} zeros

IF INPUT SAME SIZE OR SMALLER THAN OUTPUT:
→ START WITH OUTPUT SIZE: {expected_h}×{expected_w}
→ Generate {expected_h} rows, each with {expected_w} zeros

YOUR SITUATION:
- Input: {input_h}×{input_w}
- Output: {expected_h}×{expected_w}
- Input area: {input_h * input_w} cells
- Output area: {expected_h * expected_w} cells

⚠️ REQUIRED STARTING SIZE:
{f"→ START WITH INPUT SIZE {input_h}×{input_w} (input larger - will crop later)" if input_h * input_w > expected_h * expected_w else f"→ START WITH OUTPUT SIZE {expected_h}×{expected_w}"}

OUTPUT FORMAT:

GRID:
[[0,0,0,...,0],
 [0,0,0,...,0],
 ...
 [0,0,0,...,0]]

Generate the appropriate size based on the rule above.
"""
        
        response = self.call_api(prompt, [])
        
        print("MODEL RESPONSE:")
        print("─"*70)
        print(response)
        print("─"*70 + "\n")
        
        # Parse grid
        grid = self._parse_grid(response)
        
        if grid:
            actual_h, actual_w = len(grid), len(grid[0])
            print(f"✓ Model generated: {actual_h}×{actual_w} empty grid")
            
            if (actual_h, actual_w) == (expected_h, expected_w):
                print(f"  ✅ SIZE CORRECT! Matches OUTPUT size {expected_h}×{expected_w}")
                print(f"  Approach: Start with blank output-sized grid\n")
                return grid
            elif (actual_h, actual_w) == (input_h, input_w):
                print(f"  ✅ SIZE CORRECT! Matches INPUT size {input_h}×{input_w}")
                print(f"  Approach: Start with blank input-sized grid (will crop/resize later)\n")
                return grid
            else:
                print(f"  ⚠️ SIZE UNEXPECTED! Not input ({input_h}×{input_w}) or output ({expected_h}×{expected_w}), got {actual_h}×{actual_w}")
                print(f"  Using model's size anyway - model may have a different approach...\n")
                return grid
        else:
            print(f"  ❌ Could not parse grid, using OUTPUT size {expected_h}×{expected_w}\n")
            return [[0] * expected_w for _ in range(expected_h)]
    
    def generate_step(self, 
                     step_num: int,
                     arc: dict,
                     analysis: str,  # Keep parameter for compatibility but won't send to prompt
                     current_grid: List[List[int]],
                     output_image: Image.Image,
                     input_grid: List[List[int]],
                     output_grid: List[List[int]],
                     previous_step_images: List[Image.Image],
                     previous_descriptions: List[str],
                     valid_colors: set) -> List[Dict]:
        """
        Generate step with up to 3 attempts
        
        Returns list of attempts (each attempt is a dict)
        """
        print(f"\nSTEP {step_num}:")
        
        attempts = []
        
        for attempt_num in range(1, 4):
            print(f"  Attempt {attempt_num}/3...")
            
            # Prepare images: input + output goal + most recent previous step
            images = []
            from PIL import Image
            
            # Add current puzzle: input + output goal + N-1 step
            images.append(grid_to_image(input_grid, 30))  # Current input
            images.append(output_image)  # Current expected output (CRITICAL - shows goal!)
            
            # Add most recent previous step (N-1)
            if previous_step_images:
                images.append(previous_step_images[-1])  # Just the last step
            
            prev_text = "\n".join([
                f"Step {i+1}: {desc}" 
                for i, desc in enumerate(previous_descriptions)
            ]) if previous_descriptions else "None - this is the first step"
            
            # Build image description text
            num_training = len(arc['train'])
            img_desc_lines = []
            img_desc_lines.append("Image 1: CURRENT PUZZLE INPUT")
            img_desc_lines.append("Image 2: CURRENT PUZZLE EXPECTED OUTPUT (goal)")
            
            if previous_step_images:
                img_desc_lines.append(f"Image 3: Previous step (Step {len(previous_step_images)}) result")
            
            img_descriptions = "\n".join(img_desc_lines)
            
            prompt = f"""STEP {step_num} - Generate next visual step

⚠️⚠️⚠️ CRITICAL: STEP-BY-STEP OBJECT-ORIENTED VISUALIZATION ⚠️⚠️⚠️

THIS IS AN INSTRUCTIONAL BOOKLET - EACH STEP MUST:
1. BUILD THE OUTPUT GRID STEP-BY-STEP (never jump to final answer)
2. Work with ONE SPECIFIC OBJECT per step (keep actions incremental)
3. Show INCREMENTAL progress - small changes from previous step
4. Use COPY to copy from input, HIGHLIGHT only for partial transformations

⚠️⚠️⚠️ CRITICAL GRID SIZE RULE ⚠️⚠️⚠️

IF INPUT LARGER THAN OUTPUT (Input {len(input_grid)}×{len(input_grid[0])} > Output {len(output_grid)}×{len(output_grid[0])}):
→ ✅ MUST START WITH INPUT SIZE!
→ Step 1: COPY entire INPUT grid to current (current becomes input size)
→ Later steps: CROP/RESIZE down to output size

IF INPUT SAME SIZE AS OUTPUT:
→ Start with blank OUTPUT-sized grid
→ COPY/transform objects from input

IF INPUT SMALLER THAN OUTPUT:
→ Start with blank OUTPUT-sized grid
→ Use EXPAND or RESIZE to grow from input

⚠️ RULE SUMMARY:
- Input > Output? MUST start with INPUT size, then crop down
- Input ≤ Output? Start with OUTPUT size, build up

PREVIOUS STEPS:
{prev_text}

IMAGES: {img_descriptions}

INPUT: {len(input_grid)}×{len(input_grid[0])}
{self._format_grid(input_grid)}

EXPECTED OUTPUT (GOAL): {len(output_grid)}×{len(output_grid[0])}
{self._format_grid(output_grid)}

CURRENT STATE: {len(current_grid)}×{len(current_grid[0])}
Valid colors: {sorted(valid_colors)}
{self._format_grid(current_grid)}

DIFFERENCES (current vs expected):
{self._format_grid_diff(current_grid, output_grid)}

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
   ⚠️ COPY EXACTLY ONE OBJECT PER STEP (incremental, object-oriented)
   ⚠️ NEVER copy multiple objects in a single step – break it into separate COPY steps
   ⚠️ IF INPUT LARGER THAN OUTPUT: Extract/condense - copy only relevant parts from input (one object per step)
   
   Examples:
   - "COPY: Copy color 6 loop (12 cells, rows 1-5, cols 2-6) FROM INPUT to output" - One object
   - "COPY: Copy color 2 cluster (8 cells) FROM INPUT to output" - Next object in separate step
   - "COPY: Extract color 1 cells from left side of INPUT (rows 0-2, cols 0-2) to output grid" - Condensing from larger input
   
   When to use: When transferring objects from input to output (one object per COPY step)
   Note: Input may be larger than output - extract only what's needed, one object at a time!

2. MOVE(objects=[list], from=[coords], to=[coords])
   Parameters:
   - objects: Object being moved (color, dimensions)
   - from: Current position in grid
   - to: New position
   Example: "MOVE: Move color 8 block (2×2, rows 10-11, cols 3-4) down 6 rows to rows 16-17"

3. EXPAND(source=INPUT, scale_factor=N, mapping=[rules])
   Parameters:
   - source: Input grid
   - scale_factor: How much to scale (e.g., 3 for 3×3 blocks)
   - mapping: Color mapping rules
   Example: "EXPAND: Replicate each cell from 3×3 input into 3×3 block. Color 0→0s, color 7→7s."

4. FILL(regions=[list], pattern=[data], colors=[list])
   Parameters:
   - regions: Which regions/cells to fill (can be by location, color, pattern)
   - pattern: What to fill with (solid color, pattern, etc.)
   - colors: Colors used
   
   Examples:
   - "FILL: Fill all cells at even rows with color 2"
   - "FILL: Fill all cells in top half (rows 1-5) with color 0"
   - "FILL: Fill all cells matching pattern X with color 7"
   - "FILL: Fill all background cells (color 0) surrounding objects with color 5"
   - "FILL: For each 3×3 block with color 7, replace with 3×3 input pattern"
   
   When to use:
   - Filling regions based on location (rows, cols, quadrants)
   - Filling all cells of a certain type
   - Pattern-based filling

5. MODIFY(objects=[list], change=[old→new])
   Parameters:
   - objects: What to modify (color, dimensions, position)
   - change: What changes (color, size, etc.)
   Example: "MODIFY: Change color 2 cells from irregular shape to 3×3 hollow square"

5a. CROP/RESIZE(from_size=[H×W], to_size=[H×W], region=[coords])
   Parameters:
   - from_size: Current grid dimensions
   - to_size: Target grid dimensions
   - region: Which region to keep/extract (or how to resize)
   
   Examples:
   - "CROP: Remove rightmost 4 columns, changing grid from 3×7 to 3×3 (keep left portion)"
   - "CROP: Extract central 5×5 region from current 10×10 grid"
   - "RESIZE: Expand grid from 3×3 to 6×6, padding with background"
   
   When to use: When grid dimensions need to change (crop, trim, expand canvas)

6. HIGHLIGHT_SUBPROCESS - 3-STEP PROCESS (MANDATORY for partial transformations):
   
   ⚠️ USE ONLY WHEN: Some cells stay fixed, some cells change (partial transformation)
   ⚠️ DO NOT USE FOR: Copying objects (use COPY) or entire object moves/changes (use MOVE/MODIFY)
   
   CRITICAL SEQUENCE: COPY objects FROM INPUT first → then HIGHLIGHT → TRANSFORM → UN-HIGHLIGHT
   
   STEP 1: HIGHLIGHT(anchors=[SPECIFIC_POSITIONS], temp_color=X)
   - Compare INPUT vs EXPECTED OUTPUT to find SPECIFIC OBJECT cells that are IDENTICAL (anchors)
   - ⚠️ DO NOT count background cells - only OBJECT cells can be anchors!
   - ⚠️ Background = most common/prevalent color filling empty space (identify from context)
   - Mark ONLY those specific anchor OBJECT cells with temporary color (NOT in valid colors)
   - Template: "HIGHLIGHT: Input→output diff shows N OBJECT cells at [exact positions] unchanged. Mark as ANCHORS using temp color X."
   - Example: "HIGHLIGHT: Diff shows object cells (12 cells at rows 1-5, cols 2-6) unchanged. Mark as anchors using temp color 8."
   
   STEP 2: TRANSFORM(operation=[action], targets=NON_HIGHLIGHTED, keep_anchors=True)
   - Modify ONLY non-highlighted cells, keep highlighted anchors fixed
   - Template: "TRANSFORM: [Action] all NON-highlighted cells, keeping color X anchors fixed."
   
   STEP 3: UN_HIGHLIGHT(temp_color=X, restore_to=C)
   - Restore anchors to original colors
   - Template: "UN-HIGHLIGHT: Restore all color X (anchor) cells to original color C."
   
   Example scenario:
   Step 1: "COPY: Copy color 6 loop (12 cells) AND color 2 cluster (8 cells) FROM INPUT to output"
   Step 2: "HIGHLIGHT: Diff shows color 6 OBJECT cells (12 cells at rows 1-5, cols 2-6) unchanged. Mark as anchors using temp color 8."
   Step 3: "TRANSFORM: Replace NON-highlighted color-2 cells with 3×3 hollow square, keeping color-8 anchors fixed."
   Step 4: "UN-HIGHLIGHT: Restore color 8 anchors to original color 6."

7. NO-OP - No action (condition not met)

⚠️ DECISION TREE - WHICH ACTION TO USE:

Q1: Are objects in current grid already, or do I need to copy from INPUT?
→ NOT in grid yet: COPY from INPUT first (then proceed to Q2)
→ Already in grid: Continue to Q2...

Q2: Are ONLY PARTS of objects modified (some cells change, some stay fixed)?
→ YES: MUST use HIGHLIGHT subprocess (3 steps: HIGHLIGHT → TRANSFORM → UN-HIGHLIGHT)
→ NO: Continue...

Q3: Does an entire object move/change uniformly (ALL cells affected equally)?
→ Move whole object: Use MOVE (no HIGHLIGHT needed)
→ Change ALL cells: Use MODIFY (no HIGHLIGHT needed)
→ Fill regions: Use FILL

⚠️⚠️⚠️ MANDATORY OUTPUT FORMAT ⚠️⚠️⚠️

YOU MUST RESPOND WITH EXACTLY THIS FORMAT (NO EXCEPTIONS):

Visual Analysis: [1-2 sentences max - what this step does, dimensions/colors]

Description: [ACTION: explanation with dimensions, colors, positions]

GRID:
[[row1],
 [row2],
 ...]

⚠️ THE WORD "GRID:" IS MANDATORY! DO NOT SKIP IT!
⚠️ YOU MUST INCLUDE "Visual Analysis:", "Description:", AND "GRID:" SECTIONS!

⚠️ CRITICAL REQUIREMENTS:
✅ ALWAYS track: DIMENSIONS ("3×3"), COLORS ("color 7"), POSITIONS ("rows 0-2, cols 3-5")
✅ Description and Grid MUST match EXACTLY
✅ Grid size: {len(current_grid)}×{len(current_grid[0])}
✅ Use ONLY colors from {sorted(valid_colors)} (except HIGHLIGHT can use 1 temporary color)

Examples (CONCISE format):
- "COPY: Copy color 6 loop (12 cells) FROM INPUT to output"
- "COPY: Copy entire INPUT grid to current (becomes 3×7)" - Start with input size
- "CROP: Remove rightmost 4 columns from current grid, changing from 3×7 to 3×3"
- "HIGHLIGHT: Diff shows object cells (12 cells, rows 1-5, cols 2-6) unchanged. Mark as anchors using temp color 8."
- "TRANSFORM: Replace NON-highlighted cells with required pattern, keeping color-8 anchors fixed."
- "UN-HIGHLIGHT: Restore color 8 anchors to original color 6."
"""
            
            try:
                response = self.call_api(prompt, images)
                
                # Parse
                description = self._parse_description(response)
                grid = self._parse_grid(response)
                
                attempt_data = {
                    "attempt": attempt_num,
                    "response": response,
                    "description": description,
                    "grid": grid,
                    "success": False,
                    "errors": []
                }
                
                # Debug output - always show what was generated
                print(f"    📝 Description: {description[:100]}...")
                if grid:
                    print(f"    📊 Grid size: {len(grid)}×{len(grid[0]) if grid else 0}")
                    # Show first 3 rows to verify
                    print(f"    Grid sample (first 3 rows):")
                    for i, row in enumerate(grid[:3]):
                        print(f"      Row {i}: {row}")
                
                # Validate
                if not grid:
                    attempt_data["errors"].append("Could not parse grid")
                    print(f"    ❌ Grid parsing failed")
                    print(f"    Response preview (first 500 chars):")
                    print(f"    {response[:500]}")
                    print(f"    ---")
                else:
                    # Allow temporary highlight color only during anchor steps
                    allow_temp = description.startswith("HIGHLIGHT:")
                    if not self._validate_colors(grid, valid_colors, allow_temp_color=allow_temp):
                        attempt_data["errors"].append("Grid contains invalid colors")
                        print(f"    ❌ Invalid colors detected in grid")
                        print(f"    Response preview (first 500 chars):")
                        print(f"    {response[:500]}")
                        print(f"    ---")
                        attempt_data["success"] = False
                    else:
                        attempt_data["success"] = True
                        if description.startswith("HIGHLIGHT:"):
                            print(f"    ✓ Grid generated (HIGHLIGHT step)")
                        else:
                            print(f"    ✓ Grid generated")
                
                attempts.append(attempt_data)
                
                if attempt_data["success"]:
                    break
                    
            except Exception as e:
                attempts.append({
                    "attempt": attempt_num,
                    "response": "",
                    "error": str(e),
                    "success": False
                })
                print(f"    ❌ Error: {e}")
        
        return attempts
    
    def _parse_description(self, text):
        for line in text.split('\n'):
            if 'description:' in line.lower():
                return line.split(':', 1)[1].strip()
        return text.split('\n')[0]
    
    def _parse_grid(self, text):
        """Parse grid from model response - robust version"""
        import re
        
        # Try to find GRID: section first
        grid_section = None
        if "GRID:" in text:
            grid_section = text.split("GRID:", 1)[1]
        elif "Grid:" in text:
            grid_section = text.split("Grid:", 1)[1]
        else:
            grid_section = text
        
        # Try multiple parsing strategies
        
        # Strategy 1: Look for [[...], [...], ...] pattern
        full_grid_pattern = r'\[\s*\[[\d,\s]+\](?:\s*,\s*\[[\d,\s]+\])*\s*\]'
        full_match = re.search(full_grid_pattern, grid_section)
        if full_match:
            try:
                grid_str = full_match.group(0)
                # Clean up and parse
                grid_str = grid_str.replace(' ', '')
                grid = eval(grid_str)
                if self._validate_grid_structure(grid):
                    return grid
            except:
                pass
        
        # Strategy 2: Look for individual rows [...]
        pattern = r'\[[\d,\s]+\]'
        matches = re.findall(pattern, grid_section)
        
        grid = []
        for match in matches:
            numbers_str = match.strip('[]').replace(',', ' ')
            numbers = [int(n) for n in numbers_str.split() if n.isdigit()]
            if numbers:
                grid.append(numbers)
        
        if self._validate_grid_structure(grid):
            return grid
        
        return None
    
    def _validate_grid_structure(self, grid):
        """Check if grid has valid structure"""
        if not grid or len(grid) == 0:
            return False
        
        row_lengths = [len(row) for row in grid]
        if len(set(row_lengths)) != 1:  # All rows must have same length
            return False
        
        return True
    
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
    
    def _validate_colors(self, grid, valid_colors, allow_temp_color=False):
        """Validate colors in grid
        
        Args:
            grid: The grid to validate
            valid_colors: Set of valid colors for this puzzle
            allow_temp_color: If True, allows temporary colors (for HIGHLIGHT steps)
        """
        import numpy as np
        unique_colors = set(np.array(grid).flatten())
        
        if allow_temp_color:
            # For HIGHLIGHT steps, allow ONE temporary color
            # (Multiple groups with same transformation use the same temp color)
            # (Different transformations use separate HIGHLIGHT→TRANSFORM sequences)
            extra_colors = unique_colors - valid_colors
            if len(extra_colors) <= 1:  # At most one temporary color per HIGHLIGHT step
                return True
            else:
                return False  # Too many invalid colors (different transformations should be separate steps!)
        else:
            # Normal validation: all colors must be in valid set
            for row in grid:
                for c in row:
                    if c not in valid_colors:
                        return False
            return True
    
    def compare_to_gt(self, gen_grid, gt_grid):
        """Compare to ground truth"""
        if not gen_grid or not gt_grid:
            return {"match": False, "accuracy": 0.0}
        
        ga = np.array(gen_grid)
        gta = np.array(gt_grid)
        
        if ga.shape != gta.shape:
            return {"match": False, "accuracy": 0.0, "note": "Size mismatch"}
        
        acc = float(np.mean(ga == gta))
        
        diffs = []
        for r, c in np.argwhere(ga != gta)[:20]:
            diffs.append({
                "position": [int(r), int(c)],
                "generated": int(ga[r, c]),
                "expected": int(gta[r, c])
            })
        
        return {
            "match": acc == 1.0,
            "accuracy": acc,
            "differences": diffs
        }
    
    def get_available_training_examples(self, puzzle_id: str) -> List[int]:
        """Get list of available training example numbers for a puzzle"""
        base = Path(f"visual_traces/{puzzle_id}")
        if not base.exists():
            raise FileNotFoundError(f"Puzzle {puzzle_id} not found in visual_traces")
        
        training_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("training_")]
        training_nums = sorted([int(d.name.split("_")[1]) for d in training_dirs])
        return training_nums
    
    def _get_step_counts_per_training_example(self, puzzle_id: str) -> List[int]:
        """Get number of steps for each training example"""
        base = Path(f"visual_traces/{puzzle_id}")
        if not base.exists():
            return []
        
        step_counts = []
        training_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("training_")])
        
        for training_dir in training_dirs:
            num_steps = len(list(training_dir.glob("step_*")))
            step_counts.append(num_steps)
        
        return step_counts
    
    def generate_or_load_analysis(self, puzzle_id: str, force_regenerate: bool = False, skip_if_missing: bool = False) -> str:
        """Generate Phase 1 analysis once per puzzle (or load if exists)"""
        # If skip flag set, always return placeholder (don't use cached, don't generate)
        if skip_if_missing:
            print("⏸️  Phase 1 analysis SKIPPED (--skip-phase1 flag)")
            print("  Not using cached analysis - proceeding without Phase 1\n")
            return "PHASE 1 ANALYSIS SKIPPED - Using minimal context for step generation."
        
        # Check if analysis already exists
        analysis_file = Path("visual_step_results") / puzzle_id / "phase1_analysis.txt"
        
        if analysis_file.exists() and not force_regenerate:
            print("Loading existing Phase 1 analysis...")
            with open(analysis_file, encoding='utf-8') as f:
                analysis = f.read()
            print("  ✓ Analysis loaded from cache\n")
            
            # Print the loaded analysis
            print("ANALYSIS RESULT:")
            print("="*80)
            print(analysis)
            print("="*80 + "\n")
            
            return analysis
        
        # Generate new analysis
        print("Generating new Phase 1 analysis...")
        arc = self.load_arc_puzzle(puzzle_id)
        print(f"  ✓ {len(arc['train'])} training examples\n")
        
        # Use first ground truth example for color verification
        gt = self.load_ground_truth(puzzle_id, 1)
        example = {'input': gt['input_grid'], 'output': gt['output_grid']}
        
        # Generate analysis
        analysis = self.phase1_analyze(arc, example, puzzle_id)
        
        # Save for reuse
        analysis_file.parent.mkdir(parents=True, exist_ok=True)
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(analysis)
        print(f"  ✓ Analysis saved to {analysis_file}\n")
        
        return analysis
    
    def run(self, puzzle_id: str, training_num: int = 1, shared_analysis: str = None):
        """Main generation flow
        
        Args:
            puzzle_id: ARC puzzle ID
            training_num: Which training example to process (1, 2, 3, etc.)
            shared_analysis: Pre-generated Phase 1 analysis (optional, will generate if None)
        """
        print(f"\n{'='*80}")
        print(f"VISUAL STEP GENERATOR: {puzzle_id} (training_{training_num:02d})")
        print(f"{'='*80}\n")
        
        # Load data
        print(f"Loading ground truth from visual_traces (training_{training_num:02d})...")
        gt = self.load_ground_truth(puzzle_id, training_num)
        print(f"  ✓ {len(gt['steps'])} ground truth steps\n")
        
        print("Loading ARC puzzle data...")
        arc = self.load_arc_puzzle(puzzle_id)
        print(f"  ✓ {len(arc['train'])} training examples\n")
        
        # Use input/output from ground truth (visual_traces)
        input_grid = gt['input_grid']
        output_grid = gt['output_grid']
        
        print(f"Using ground truth input/output:")
        print(f"  Input: {len(input_grid)}×{len(input_grid[0])}")
        print(f"  Output: {len(output_grid)}×{len(output_grid[0])}\n")
        
        # Phase 1: Use shared analysis or generate new
        if shared_analysis:
            if "SKIPPED" in shared_analysis:
                print("⏸️  Using minimal context (Phase 1 skipped)\n")
            else:
                print("Using shared Phase 1 analysis (generated once for all training examples)\n")
                # Print it for this run
                print("ANALYSIS BEING USED:")
                print("="*80)
                print(shared_analysis)
                print("="*80 + "\n")
            analysis = shared_analysis
        else:
            print("Generating Phase 1 analysis for this puzzle...")
            analysis = self.generate_or_load_analysis(puzzle_id)
        
        # Get valid colors
        valid_colors = self._get_valid_colors(arc)
        print(f"Valid colors: {sorted(valid_colors)}\n")
        
        # Generate each step (Step 1 = empty grid, matches your ground truth!)
        output_image = grid_to_image(output_grid, 30)
        input_image = grid_to_image(input_grid, 30)
        
        all_step_results = []
        current_grid = input_grid  # Start from input
        previous_images = []
        previous_descriptions = []
        
        # Allow generating more steps than ground truth if needed (e.g., for HIGHLIGHT subprocesses)
        max_steps = len(gt['steps']) + 10  # Allow up to 10 extra steps
        step_num = 0
        
        # Track consecutive NO-OP steps to detect stuck loops
        consecutive_noops = 0
        max_consecutive_noops = 3
        
        while step_num < max_steps:
            step_num += 1
            
            # Get ground truth step if available (may not exist for extra steps)
            gt_step = gt['steps'][step_num - 1] if step_num <= len(gt['steps']) else None
            
            # Special case: Step 1 is always "create empty grid"
            if step_num == 1:
                # step0_generate_empty_grid prints its own header
                empty_grid = self.step0_generate_empty_grid(analysis, input_grid, output_grid)
                
                # Compare to ground truth Step 1 if available
                if gt_step:
                    comparison = self.compare_to_gt(empty_grid, gt_step['grid'])
                    print(f"  📝 MODEL: Create empty output grid")
                    print(f"  📝 GT: {gt_step.get('description', 'N/A')}")
                    print(f"  📊 Accuracy: {comparison['accuracy']:.1%}")
                else:
                    comparison = {"accuracy": 1.0, "match": True, "diffs": []}
                    print(f"  📝 MODEL: Create empty output grid")
                
                all_step_results.append({
                    "step_num": 1,
                    "attempts": [{
                        "attempt": 1,
                        "description": "Create empty output grid",
                        "grid": empty_grid,
                        "success": True
                    }],
                    "used_ground_truth": False,
                    "final_grid": empty_grid,
                    "final_description": "Create empty output grid",
                    "ground_truth": {
                        "grid": gt_step['grid'] if gt_step else None,
                        "description": gt_step.get('description', '') if gt_step else ''
                    },
                    "comparison": comparison
                })
                
                current_grid = empty_grid
                previous_images.append(grid_to_image(empty_grid, 30))
                previous_descriptions.append("Create empty output grid")
                
                continue
            
            # For steps 2+, generate normally
            print(f"STEP {step_num}:")
            
            # Generate with 3 attempts
            attempts = self.generate_step(
                step_num=step_num,
                arc=arc,
                analysis=analysis,
                current_grid=current_grid,
                output_image=output_image,
                input_grid=input_grid,
                output_grid=output_grid,
                previous_step_images=previous_images,
                previous_descriptions=previous_descriptions,
                valid_colors=valid_colors
            )
            
            # Check if any attempt succeeded
            successful_attempt = next((a for a in attempts if a['success']), None)
            
            if successful_attempt:
                next_grid = successful_attempt['grid']
                next_desc = successful_attempt['description']
                used_ground_truth = False
                print(f"  ✓ Success on attempt {successful_attempt['attempt']}")
                print(f"\n  📝 MODEL DESCRIPTION:")
                print(f"     {next_desc}")
            else:
                # All failed
                if gt_step:
                    # Use ground truth if available
                    next_grid = gt_step['grid']
                    next_desc = gt_step.get('description', f"[Ground truth step {step_num}]")
                    used_ground_truth = True
                    print(f"  ⚠️ All attempts failed - using ground truth")
                    print(f"\n  📝 GT DESCRIPTION:")
                    print(f"     {next_desc}")
                else:
                    # No ground truth available and all failed - stop here
                    print(f"  ⚠️ All attempts failed and no ground truth available")
                    print(f"  Stopping at step {step_num - 1}")
                    break
            
            # Compare to ground truth step if available, otherwise compare to final expected output
            if gt_step:
                comparison = self.compare_to_gt(
                    successful_attempt['grid'] if successful_attempt else None,
                    gt_step['grid']
                )
                print(f"\n  📊 COMPARISON TO GROUND TRUTH:")
                print(f"     Accuracy: {comparison['accuracy']:.1%}")
                print(f"     GT Description: {gt_step.get('description', 'N/A')}")
            else:
                # No GT step - this is an extra step beyond ground truth
                # Compare to final expected output
                gen_grid = successful_attempt['grid'] if successful_attempt else None
                comparison = self.compare_to_gt(gen_grid, output_grid)
                
                # Check grid size first
                if gen_grid:
                    size_match = (len(gen_grid) == len(output_grid) and 
                                 len(gen_grid[0]) == len(output_grid[0]))
                    if not size_match:
                        print(f"\n  📊 COMPARISON TO EXPECTED OUTPUT:")
                        print(f"     ⚠️ Grid size mismatch!")
                        print(f"     Generated: {len(gen_grid)}×{len(gen_grid[0])}")
                        print(f"     Expected: {len(output_grid)}×{len(output_grid[0])}")
                        print(f"     Accuracy: 0.0% (size mismatch)")
                        print(f"     (Extra step {step_num - len(gt['steps'])} beyond ground truth)")
                    else:
                        print(f"\n  📊 COMPARISON TO EXPECTED OUTPUT:")
                        print(f"     Accuracy: {comparison['accuracy']:.1%}")
                        print(f"     (Extra step {step_num - len(gt['steps'])} beyond ground truth)")
                else:
                    print(f"\n  📊 COMPARISON TO EXPECTED OUTPUT:")
                    print(f"     Accuracy: 0.0% (no grid generated)")
                    print(f"     (Extra step {step_num - len(gt['steps'])} beyond ground truth)")
            
            # DEBUG: Show grid sizes
            if successful_attempt and successful_attempt.get('grid'):
                gen_grid = successful_attempt['grid']
                print(f"     Generated: {len(gen_grid)}×{len(gen_grid[0])} grid")
                if gt_step:
                    print(f"     Ground truth: {len(gt_step['grid'])}×{len(gt_step['grid'][0])} grid")
                else:
                    print(f"     Expected output: {len(output_grid)}×{len(output_grid[0])} grid")
                
                if comparison['accuracy'] == 0.0:
                    print(f"     ⚠️ 0% accuracy - grids completely different!")
                    print(f"     Generated sample: {gen_grid[0][:5] if gen_grid else 'none'}")
                    if gt_step:
                        print(f"     GT sample: {gt_step['grid'][0][:5]}")
            
            if comparison['accuracy'] < 1.0 and comparison.get('differences'):
                print(f"     First 3 diffs: {comparison['differences'][:3]}")
            
            # Save step result
            step_result = {
                "step_num": step_num,
                "attempts": attempts,
                "used_ground_truth": used_ground_truth,
                "final_grid": next_grid,
                "final_description": next_desc,
                "ground_truth": {
                    "grid": gt_step['grid'] if gt_step else None,
                    "description": gt_step.get('description', '') if gt_step else ''
                },
                "comparison": comparison
            }
            
            all_step_results.append(step_result)
            
            # Update for next step
            current_grid = next_grid
            previous_images.append(grid_to_image(next_grid, 30))
            previous_descriptions.append(next_desc)
            
            # Check if we've reached the expected output (100% accuracy)
            final_comparison = self.compare_to_gt(next_grid, output_grid)
            if final_comparison['accuracy'] == 1.0:
                print(f"\n✅ REACHED EXPECTED OUTPUT! Grid matches 100% at step {step_num}")
                break
            
            # Detect stuck loops: consecutive NO-OP steps with wrong grid size
            is_noop = next_desc.upper().startswith("NO-OP")
            grid_size_matches = (len(next_grid) == len(output_grid) and 
                                len(next_grid[0]) == len(output_grid[0]))
            
            if is_noop:
                consecutive_noops += 1
                if not grid_size_matches:
                    print(f"\n⚠️ WARNING: NO-OP step {step_num} has wrong grid size!")
                    print(f"   Current: {len(next_grid)}×{len(next_grid[0])}, Expected: {len(output_grid)}×{len(output_grid[0])}")
                    print(f"   Model appears stuck - stopping to avoid infinite loop")
                    break
                elif consecutive_noops >= max_consecutive_noops:
                    print(f"\n⚠️ WARNING: {consecutive_noops} consecutive NO-OP steps detected!")
                    print(f"   Model appears stuck in a loop - stopping")
                    break
            else:
                consecutive_noops = 0  # Reset counter on non-NO-OP step
            
            # Additional check: if grid size is wrong and we're past ground truth steps, stop
            if step_num > len(gt['steps']) and not grid_size_matches:
                print(f"\n⚠️ WARNING: Extra step {step_num} has wrong grid size!")
                print(f"   Current: {len(next_grid)}×{len(next_grid[0])}, Expected: {len(output_grid)}×{len(output_grid[0])}")
                print(f"   Stopping to avoid further incorrect steps")
                break
        
        # Final validation: Check for temporary colors in final grid
        if all_step_results:
            final_grid = all_step_results[-1]['final_grid']
            final_desc = all_step_results[-1]['final_description']
            if final_grid:
                import numpy as np
                unique_colors = set(np.array(final_grid).flatten())
                invalid_colors = unique_colors - valid_colors
                
                if invalid_colors:
                    print(f"\n⚠️⚠️⚠️ WARNING: TEMPORARY COLORS DETECTED IN FINAL OUTPUT! ⚠️⚠️⚠️")
                    print(f"  Valid colors: {sorted(valid_colors)}")
                    print(f"  Found invalid colors: {sorted(invalid_colors)}")
                    print(f"  ❌ All HIGHLIGHT temporary colors should be removed by the final step!")
                    print(f"  ❌ Final output must ONLY contain valid puzzle colors!")
                else:
                    print(f"\n✓ Final output validation: NO temporary colors - only valid colors {sorted(valid_colors)}")
        
        # Save everything
        self._save_results(puzzle_id, training_num, analysis, all_step_results, input_grid, output_grid)
        
        return all_step_results
    
    def _save_results(self, puzzle_id: str, training_num: int, analysis: str, results: List[Dict], input_grid: List[List[int]], output_grid: List[List[int]]):
        """Save all data for Streamlit UI"""
        output_dir = Path("visual_step_results") / puzzle_id / f"training_{training_num:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save shared analysis at puzzle level (if not already saved)
        analysis_file = Path("visual_step_results") / puzzle_id / "phase1_analysis.txt"
        if not analysis_file.exists():
            analysis_file.parent.mkdir(parents=True, exist_ok=True)
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write(analysis)
        
        # Save JSON
        data = {
            "puzzle_id": puzzle_id,
            "training_num": training_num,
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "phase1_analysis": analysis,
            "input_grid": input_grid,
            "expected_output_grid": output_grid,
            "steps": results
        }
        
        with open(output_dir / "results.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save input/output overview images
        input_img = grid_to_image(input_grid, 30)
        input_img.save(output_dir / "input.png")
        
        output_img = grid_to_image(output_grid, 30)
        output_img.save(output_dir / "expected_output.png")
        
        # Save images for each step and attempt
        for step_result in results:
            step_num = step_result['step_num']
            
            # Save each attempt's grid as image
            for attempt in step_result['attempts']:
                if attempt.get('grid'):
                    img = grid_to_image(attempt['grid'], 30)
                    img.save(output_dir / f"step_{step_num:02d}_attempt_{attempt['attempt']}.png")
            
            # Save ground truth (only if available - extra steps may not have GT)
            gt_grid = step_result['ground_truth'].get('grid')
            if gt_grid is not None:
                gt_img = grid_to_image(gt_grid, 30)
                gt_img.save(output_dir / f"step_{step_num:02d}_ground_truth.png")
            
            # Save final (what was used for next step)
            final_img = grid_to_image(step_result['final_grid'], 30)
            final_img.save(output_dir / f"step_{step_num:02d}_final.png")
        
        print(f"\n{'='*80}")
        print(f"SAVED TO: {output_dir}/")
        print(f"  results.json")
        print(f"  input.png (puzzle input)")
        print(f"  expected_output.png (final goal)")
        print(f"  step_XX_attempt_Y.png (all attempts)")
        print(f"  step_XX_ground_truth.png")
        print(f"  step_XX_final.png (what was used)")
        print(f"{'='*80}\n")
    
    def _parse_description(self, response: str) -> str:
        """Parse description from model response"""
        # Look for "Description:" tag
        if "Description:" in response:
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith("Description:"):
                    # Get the description line
                    desc = line.split("Description:", 1)[1].strip()
                    if desc:
                        return desc
                    # If empty, try next line
                    if i + 1 < len(lines):
                        return lines[i + 1].strip()
        
        # Fallback: try to extract first meaningful sentence
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        for line in lines:
            if any(line.startswith(action + ":") for action in ["EXPAND", "HIGHLIGHT", "TRANSFORM", "FILL", "COPY", "MOVE", "MODIFY", "NO-OP"]):
                return line
        
        # Last resort: return first non-empty line
        return lines[0] if lines else "No description provided"
    
    def _get_valid_colors(self, arc):
        colors = set()
        for ex in arc['train']:
            for row in ex['input'] + ex['output']:
                colors.update(row)
        return colors
    
    def load_ground_truth(self, puzzle_id: str, training_num: int = 1) -> Dict:
        """Load from visual_traces
        
        Args:
            puzzle_id: ARC puzzle ID
            training_num: Which training example (1, 2, 3, etc.)
        """
        base = Path(f"visual_traces/{puzzle_id}")
        training_dir = base / f"training_{training_num:02d}"
        
        if not training_dir.exists():
            raise FileNotFoundError(f"Training example {training_num} not found for puzzle {puzzle_id}")
        
        steps = []
        for step_dir in sorted(training_dir.glob("step_*")):
            with open(step_dir / "step.json") as f:
                steps.append(json.load(f))
        
        # Extract input/output from first step
        input_grid = steps[0]['original_input'] if steps else None
        output_grid = steps[0]['original_output'] if steps else None
        
        return {
            "puzzle_id": puzzle_id,
            "training_num": training_num,
            "steps": steps,
            "input_grid": input_grid,
            "output_grid": output_grid
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--puzzle', required=True, help='Puzzle ID')
    parser.add_argument('--model', default='gpt-5-mini', help='Model to use')
    parser.add_argument('--training', type=int, help='Specific training example number (1, 2, 3, etc.)')
    parser.add_argument('--all', action='store_true', help='Process all training examples for this puzzle')
    parser.add_argument('--skip-phase1', action='store_true', help='Skip Phase 1 analysis generation (use cached only)')
    args = parser.parse_args()
    
    gen = VisualStepGenerator(args.model)
    
    if args.all:
        # Process all training examples
        print(f"\n{'='*80}")
        print(f"BATCH MODE: Processing all training examples for {args.puzzle}")
        print(f"{'='*80}\n")
        
        training_nums = gen.get_available_training_examples(args.puzzle)
        print(f"Found {len(training_nums)} training examples: {training_nums}\n")
        
        # Generate Phase 1 analysis ONCE for the entire puzzle (or skip if flag set)
        print("="*80)
        if args.skip_phase1:
            print("PHASE 1 ANALYSIS (--skip-phase1 flag set - will use cached or skip)")
        else:
            print("GENERATING SHARED PHASE 1 ANALYSIS (used for all training examples)")
        print("="*80 + "\n")
        shared_analysis = gen.generate_or_load_analysis(args.puzzle, skip_if_missing=args.skip_phase1)
        
        print(f"\n{'='*80}")
        print(f"NOW PROCESSING EACH TRAINING EXAMPLE WITH SHARED ANALYSIS")
        print(f"{'='*80}\n")
        
        for training_num in training_nums:
            try:
                gen.run(args.puzzle, training_num, shared_analysis=shared_analysis)
            except Exception as e:
                print(f"\n❌ Error processing training_{training_num:02d}: {e}\n")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n{'='*80}")
        print(f"✅ BATCH COMPLETE: Processed {len(training_nums)} training examples")
        print(f"✅ All examples used the SAME Phase 1 analysis")
        print(f"{'='*80}")
    
    elif args.training:
        # Process specific training example
        gen.run(args.puzzle, args.training)
        print("\n✅ Generation complete!")
    
    else:
        # Default: process training_01
        gen.run(args.puzzle, 1)
        print("\n✅ Generation complete!")
    
    print(f"\nView results:")
    print(f"  streamlit run scripts/view_visual_steps.py")


if __name__ == "__main__":
    main()

