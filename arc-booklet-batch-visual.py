#!/usr/bin/env python3
"""
ARC Batch Visual Booklet Generator
Shows AI all training inputs at once to determine visual similarity and generate universal steps

TWO-CONTEXT APPROACH:
===================
This generator uses TWO separate contexts to avoid output contamination:

1. REASONING CONTEXT (Phase 1-2, 4):
   - Sees ALL inputs + outputs
   - Performs structured reasoning
   - Generates/refines universal steps
   - Has full information for hypothesis formation

2. EXECUTION CONTEXT (Phase 3, 5):
   - Each call_ai() during execution is FRESH (no prior context)
   - Never saw outputs during reasoning
   - Executes steps blindly
   - TRUE unbiased validation

This mimics human cognition: 
- Think with full information (reasoning)
- Execute mechanically without peeking (blind execution)
"""

import json
import os
import sys
import base64
from typing import Dict, List, Any, Optional
from openai import OpenAI
from PIL import Image
import numpy as np
from datetime import datetime
from pathlib import Path
import re

# Import the visualizer functions
from arc_visualizer import grid_to_image


class ARCBatchVisualGenerator:
    def __init__(self):
        """Initialize the batch visual generator"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be set in OPENAI_API_KEY environment variable")
        self.client = OpenAI(api_key=self.api_key)
        self.conversation_history = []  # Shared context for execution
        
    def load_task(self, file_path: str) -> Dict[str, Any]:
        """Load an ARC-AGI task from JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def format_grid(self, grid: List[List[int]]) -> str:
        """Format a grid for display"""
        return '\n'.join(['[' + ', '.join(str(cell) for cell in row) + ']' for row in grid])
    
    def encode_image(self, image_path: str) -> str:
        """Encode an image file to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def call_ai(self, text_prompt: str, image_paths: List[str] = None, use_history: bool = False) -> str:
        """Call OpenAI API with text and optional images
        
        Args:
            text_prompt: The prompt text
            image_paths: Optional list of image paths to include
            use_history: If True, use conversation history (for execution with context)
        """
        content = [{"type": "text", "text": text_prompt}]
        
        if image_paths:
            for img_path in image_paths:
                base64_img = self.encode_image(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                })
        
        # Build messages with or without history
        if use_history and self.conversation_history:
            messages = self.conversation_history.copy()
            messages.append({"role": "user", "content": content})
        else:
            messages = [{"role": "user", "content": content}]
        
        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages
        )
        
        return response.choices[0].message.content
    
    def parse_grid_from_response(self, response: str) -> Optional[List[List[int]]]:
        """Parse grid from AI response with multiple fallback strategies"""
        # Strategy 1: Look for bracketed rows [1,2,3]
        pattern = r'\[[\d,\s]+\]'
        matches = re.findall(pattern, response)
        
        grid = []
        for match in matches:
            numbers_str = match.strip('[]').replace(',', ' ')
            numbers = [int(n) for n in numbers_str.split() if n.isdigit()]
            if numbers:
                grid.append(numbers)
        
        if grid and len(grid) > 0:
            row_lengths = [len(row) for row in grid]
            if len(set(row_lengths)) == 1:
                return grid
        
        # Strategy 2: Look for "GRID:" or "OUTPUT:" section
        grid_section_pattern = r'(?:GRID|OUTPUT|RESULT):\s*\n((?:\[[\d,\s]+\]\s*\n?)+)'
        grid_match = re.search(grid_section_pattern, response, re.IGNORECASE | re.MULTILINE)
        
        if grid_match:
            grid_text = grid_match.group(1)
            matches = re.findall(r'\[[\d,\s]+\]', grid_text)
            grid = []
            for match in matches:
                numbers_str = match.strip('[]').replace(',', ' ')
                numbers = [int(n) for n in numbers_str.split() if n.isdigit()]
                if numbers:
                    grid.append(numbers)
            if grid:
                return grid
        
        # Strategy 3: Look for lines of digits
        lines = response.split('\n')
        grid = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            numbers = re.findall(r'\d', line)
            if len(numbers) >= 3:
                grid.append([int(n) for n in numbers])
        
        if grid and len(grid) >= 3:
            row_lengths = [len(row) for row in grid]
            if len(set(row_lengths)) == 1:
                return grid
        
        return None
    
    def generate_batch_booklet(self, task_file: str, output_dir: str = "batch_visual_booklets"):
        """Generate booklet by seeing all training examples at once"""
        
        # Load task
        task = self.load_task(task_file)
        task_name = Path(task_file).stem
        
        training_examples = task['train']
        test_examples = task.get('test', [])
        
        print(f"\n{'='*80}")
        print(f"BATCH VISUAL BOOKLET GENERATION")
        print(f"{'='*80}")
        print(f"Task: {task_name}")
        print(f"Training Examples: {len(training_examples)}")
        print(f"Test Examples: {len(test_examples)}")
        print(f"{'='*80}\n")
        
        # Create output directory with timestamp for version tracking
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"{task_name}_batch_{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Output will be saved to: {output_path}\n")
        
        # Create temp dir for images
        temp_dir = Path("img_tmp")
        temp_dir.mkdir(exist_ok=True)
        
        # PHASE 1: Show ALL examples upfront (human-like cognition)
        print("Phase 1: Structured Reasoning (Human-like Cognition)...")
        print("-" * 80)
        
        # Create images for ALL training examples (inputs + outputs)
        all_image_paths = []
        
        for i, example in enumerate(training_examples):
            input_path = str(temp_dir / f"{task_name}_ex{i+1}_input.png")
            output_img_path = str(temp_dir / f"{task_name}_ex{i+1}_output.png")
            
            img_in = grid_to_image(example['input'], 30)
            img_in.save(input_path)
            
            img_out = grid_to_image(example['output'], 30)
            img_out.save(output_img_path)
            
            all_image_paths.extend([input_path, output_img_path])
        
        # Multi-stage reasoning prompt (mirrors human cognition)
        reasoning_prompt = f"""You are solving an ARC-AGI puzzle with {len(training_examples)} training examples.

ALL TRAINING EXAMPLES (input → output pairs, see images above):
"""
        
        for i in range(len(training_examples)):
            reasoning_prompt += f"Example {i+1}: Input → Output\n"
        
        reasoning_prompt += f"""

Follow this STRUCTURED REASONING process (like a human would):

═══════════════════════════════════════════════════════════════════════════
STAGE 1: ANALYZE EACH EXAMPLE INDIVIDUALLY
═══════════════════════════════════════════════════════════════════════════
For EACH of the {len(training_examples)} examples, answer:

Example 1:
- What SPECIFIC objects/shapes are in the input? (colors, sizes, positions)
- What SPECIFIC objects/shapes are in the output? (colors, sizes, positions)
- What CHANGED from input to output? 
  - Were objects added? removed? moved? recolored?
  - Were ENTIRE objects changed or only PARTS of objects? (e.g., "only the top half was recolored")
  - If partial: which part? (top/bottom? left/right? edges? interior?)
- What STAYED THE SAME? (dimensions? certain objects? parts of objects? background?)
- In an input, if there is some object that is different than all the other objects (a reference object) what does it say about the puzzle?

Example 2:
- [Same questions]

(Repeat for all examples)

═══════════════════════════════════════════════════════════════════════════
STAGE 2: COMPARE INPUTS (Focus on Differences)
═══════════════════════════════════════════════════════════════════════════
**CRITICAL: Explicitly identify what DIFFERS between inputs - this reveals what your rule must handle!**

Compare INPUTS side-by-side:
- Example 1 input vs Example 2 input: WHAT IS DIFFERENT?
  (e.g., "Ex1 has 1 hole, Ex2 has 2 holes", "Ex1 is 6x6, Ex2 is 10x10")
- Example 1 input vs Example 3 input: WHAT IS DIFFERENT?
- Example 2 input vs Example 3 input: WHAT IS DIFFERENT?
- (Continue for all pairs)

What INPUT features vary?
- Grid dimensions? [list all sizes: 6x6, 10x10, 20x20]
- Number of objects/regions? [list all counts: 1, 2, 3, 5]
- Object sizes? [small single-pixel vs large rectangular]
- Object positions? [scattered vs clustered]
- Colors used? [do all use same colors or different?]

**ALSO Compare OUTPUTS side-by-side:**
- Example 1 output vs Example 2 output: WHAT IS DIFFERENT?
  (e.g., "Ex1 filled 1 region, Ex2 filled 2 regions", "Ex1 output 6x6, Ex2 output 10x10")
- What OUTPUT features vary?
  (This helps confirm what the transformation does to different inputs)
- What OUTPUT features are CONSTANT?
  (e.g., "all outputs use same colors", "all have yellow-filled regions", "dimensions always match input")

═══════════════════════════════════════════════════════════════════════════
STAGE 3: FIND THE PATTERN (Constants Despite Differences)
═══════════════════════════════════════════════════════════════════════════
Despite the INPUT differences you found, what is CONSTANT in the transformation?

CONSTANTS (True for EVERY example):
- Output dimensions: [same as input? different? specific size?]
- Transformation type: [fill? move? copy? recolor? add? remove? extend?]
- Which objects/regions are affected: [all? some? based on what property?]
- ENTIRE objects or PARTS of objects?: [whole objects change? or only top/bottom/edges?]
- What criterion determines the transformation: [position? color? enclosure? connectivity?]
- Spatial rules: [positions? alignments? distances? directions?]
- If partial transformations: which part? [top rows? bottom edge? left side? interior vs border?]

**KEY INSIGHT:** If inputs differ in count/size but transformation is CONSTANT, 
your rule must use "for each" or "all" to handle the variation!

VARIANTS (Features that differ but don't affect the transformation):
- What varies: [object count? sizes? positions? colors?]
- Range of variation: [2-5 objects? sizes 1-3? grid sizes 6x6 to 20x20?]

═══════════════════════════════════════════════════════════════════════════
STAGE 4: FORM & TEST HYPOTHESIS
═══════════════════════════════════════════════════════════════════════════

Based on the CONSTANTS you found (ignoring the variations), state THE RULE in ONE sentence:
"The rule is: [FILL IN - be specific but general]"

Now TEST this rule example-by-example:

Example 1: If I apply "[your rule]" to Example 1's input...
  → Would I get Example 1's output? [YES/NO and why]
  
Example 2: If I apply "[your rule]" to Example 2's input...
  → Would I get Example 2's output? [YES/NO and why]

(Test all examples)

If ANY example fails → REVISE your rule and test again.

═══════════════════════════════════════════════════════════════════════════
STAGE 5: EDGE CASES & REFINEMENT
═══════════════════════════════════════════════════════════════════════════

Check these common mistakes:
1. Did you account for ALL objects in each example? (don't miss small details)
2. Is your rule specific about HOW things change? (not just "fill" but "fill WHAT with WHAT")
3. Does your rule handle variation? (different counts/sizes/positions)
4. Is there a REFERENCE object or pattern that guides the transformation?

═══════════════════════════════════════════════════════════════════════════
OUTPUT YOUR REASONING
═══════════════════════════════════════════════════════════════════════════
Provide:

1. INDIVIDUAL ANALYSIS: What changed in each example (be specific!)

2. INPUT DIFFERENCES: What varies between the inputs? (dimensions? object counts? sizes?)

3. OUTPUT DIFFERENCES: What varies between the outputs? (confirms how transformation handles variation)

4. UNIVERSAL CONSTANTS: Despite input/output differences, what's ALWAYS true in every transformation?

5. THE RULE: One clear sentence describing the transformation (must handle all variations)

6. VALIDATION: For each example, confirm your rule produces the correct output

7. CONFIDENCE: How certain are you this rule is correct? Any ambiguities?"""
        
        reasoning_analysis = self.call_ai(reasoning_prompt, all_image_paths)
        
        try:
            print(f"Reasoning Analysis: {reasoning_analysis[:400]}...\n")
        except:
            print(f"Reasoning Analysis: [Generated - {len(reasoning_analysis)} chars]\n")
        
        # PHASE 1.5: COLOR VERIFICATION - Map visual appearance to actual numeric values
        print("\nPhase 1.5: Verifying Color Values...")
        print("-" * 80)
        
        # Extract all unique color values from training examples
        all_color_values = set()
        for ex in training_examples:
            for row in ex['input']:
                all_color_values.update(row)
            for row in ex['output']:
                all_color_values.update(row)
        
        color_values_sorted = sorted(all_color_values)
        
        # Show grid data samples to verify colors
        color_verify_prompt = f"""CRITICAL COLOR VERIFICATION:

You just analyzed the puzzle visually. Now we need to verify the EXACT numeric color values.

**ACTUAL COLOR VALUES in the grid data:**
Values present: {color_values_sorted}

**Example grid data from training example 1:**
Input (first 3 rows):
{chr(10).join([str(row) for row in training_examples[0]['input'][:3]])}

Output (first 3 rows):
{chr(10).join([str(row) for row in training_examples[0]['output'][:3]])}

**TASK: Create a color mapping**

For each numeric value in {color_values_sorted}, identify what you SEE visually in the images:

Example format:
- Color 0: Black (background)
- Color 1: Blue
- Color 2: Red
- Color 5: Gray/Orange-looking squares
- Color 7: Orange

**IMPORTANT:** 
1. Use the ACTUAL NUMERIC VALUES from the grid data
2. Don't assume color names from appearance - verify with grid data
3. If you called something "orange" in your analysis, check if it's actually value 5 or 7
4. The NUMBERS are ground truth, not the visual appearance

Output a clear mapping for each color value present."""
        
        color_mapping = self.call_ai(color_verify_prompt, all_image_paths[:2])  # Show first input/output pair
        
        try:
            print(f"Color Mapping:\n{color_mapping}\n")
        except:
            print(f"Color Mapping: [Generated - {len(color_mapping)} chars]\n")
        
        # PHASE 2: Generate universal steps based on validated hypothesis
        print("\nPhase 2: Generating Universal Steps from Validated Hypothesis...")
        print("-" * 80)
        
        all_images_with_outputs = all_image_paths  # Already have all images
        
        # Extract the rule from reasoning analysis
        rule_section = "THE RULE"
        if rule_section in reasoning_analysis:
            rule_start = reasoning_analysis.index(rule_section)
            rule_text = reasoning_analysis[rule_start:rule_start+500]
        else:
            rule_text = "[Rule not explicitly extracted - review full reasoning]"
        
        steps_prompt = f"""Based on your structured reasoning, generate UNIVERSAL STEPS that work for ALL {len(training_examples)} training examples.

**YOUR COMPLETE STRUCTURED REASONING:**
{reasoning_analysis}

**VERIFIED COLOR MAPPING (use these EXACT values in your steps):**
{color_mapping}

🚨 **CRITICAL - USE NUMERIC COLOR VALUES ONLY:**
- In your reasoning, you may have used color NAMES (e.g., "orange", "gray")
- Those were VISUAL descriptions and may not match actual values
- In your STEPS, use ONLY the verified numeric values from the mapping above
- Example: If you said "orange rectangles" but the mapping shows "Color 5: gray/orange-looking"
  → Your step MUST say "color 5" or "gray (5)", NOT "orange (7)"

**CRITICAL: Your steps MUST be a direct translation of your validated rule. Do NOT add complexity or deviate!**

**BEFORE GENERATING STEPS - EXTRACT THE CORE ACTION:**

From your hypothesis and analysis of INPUT DIFFERENCES, identify:
1. **WHAT to identify/find**: [What objects/regions/patterns to locate visually]
2. **HOW MANY**: [Does count vary? If so, use "for each" or "all"]
3. **WHAT to do to them**: [What transformation/action to apply]
4. **WHAT to preserve**: [What should NOT change]

Example:
- Hypothesis: "Fill black regions enclosed by green with yellow"
- INPUT DIFFERENCES: Grid sizes vary (6x6, 10x10, 20x20), hole count varies (1-5), hole sizes vary
- WHAT to find: Black (0) regions completely enclosed by green (3) (not touching border)
- HOW MANY: VARIES (1-5 regions) → must use "for each" or "all"
- WHAT to do: Fill with yellow (4)
- WHAT to preserve: Green (3) pixels, border-touching black (0) pixels

TRANSLATES TO STEPS:
1. Find all black (0) regions that are completely enclosed by green (3) and don't touch the image border
2. For each found region, fill it completely with yellow (4)
3. Leave all green (3) pixels and border-touching black (0) pixels unchanged

See how differences in inputs → "for each" pattern? This handles variation!
No BFS, no algorithms, just visual action applied to ALL matching objects!

**ALL TRAINING EXAMPLES (see images - input/output pairs):**
"""
        
        for i in range(len(training_examples)):
            steps_prompt += f"\nExample {i+1}: Input → Output\n"
        
        steps_prompt += f"""

TASK: Using THE RULE you validated above, break it into 2-8 VISUAL, EXECUTABLE steps.

**🚨 IF YOUR RULE IS COMPLEX → USE MORE STEPS, NOT ALGORITHMS!**

Examples of how to handle complex rules:

**SIMPLE RULE (2-3 steps):**
Rule: "Move colored blocks to touch the red bar"
Steps:
1. Identify the long red (2) bar
2. For each non-red colored block, slide it toward the red bar until touching
3. Preserve all shapes and colors

**COMPLEX RULE (5-8 steps) - NO ALGORITHMS:**
Rule: "Fill black regions enclosed by green with yellow"

❌ WRONG (algorithmic):
1. Allocate visited array
2. Run BFS on black pixels
3. Check touches_border flag
4. Fill if enclosed

✅ RIGHT (visual breakdown):
1. Find all separate black (0) regions in the image
2. For each black region, visually trace its boundary
3. Check: Does any part of this region touch the image border? (look at edges)
4. If the region does NOT touch the border, it is enclosed
5. For each enclosed region, fill every pixel of that region with yellow (4)
6. Keep all green (3) pixels unchanged
7. Keep border-touching black (0) pixels unchanged

**CRITICAL REQUIREMENTS:**
1. Base steps DIRECTLY on your hypothesis (don't add complexity!)
2. If your rule involves connectivity/enclosure/detection → BREAK IT INTO MORE STEPS
3. **NEVER use algorithms** - if tempted to write BFS/DFS, break into visual steps instead:
   - ❌ "Run BFS to find enclosed regions"
   - ✅ "1. Find all black regions
         2. For each region, check if any part touches the border
         3. Regions not touching border are enclosed
         4. Fill each enclosed region with yellow"
4. Steps must work for ALL {len(training_examples)} examples, handling the VARIANTS you identified
5. Use REPEATABLE PATTERNS when object count varies (use "for each" to handle variation)
6. **Think: What would a HUMAN do step-by-step?** Not what a program would do.
7. **If it takes 8 steps to explain without algorithms, use 8 steps!**

**HOW TO CONVERT YOUR RULE INTO STEPS:**

**IMPORTANT: Your hypothesis is already the answer! Break it into 2-8 visual actions.**

If your rule is simple, use fewer steps. If complex (connectivity, enclosure), use more steps!

If your rule is: "Fill each enclosed green-framed region with yellow"
Then steps should be:

OPTION A (Repeatable pattern - BEST):
'1. Find all black (0) regions that are completely enclosed by green (3) pixels
 2. For each found region, fill the entire region with yellow (4)
 3. Preserve all green (3) pixels and border-touching black (0) pixels'

OPTION B (Only if objects have specific properties):
'1. Identify reference row at bottom showing: red (2), blue (1), yellow (4)
 2. For each colored block in the grid (scanning left-to-right, top-to-bottom), draw bridge to next color in sequence using the block's color'

NOT (too sequential):
'1. Find first region, fill with yellow
 2. Find second region, fill with yellow  
 3. Find third region, fill with yellow
 (etc. - 7 steps for 7 regions)' ❌ (What if there are 10 regions? Or 3?)

**STEP PHRASING GUIDE:**
- Use VISUAL SPATIAL language: "the bottom row", "each colored block", "enclosed regions"
- NOT coordinates: avoid "cell at (x,y)" or "row 3, column 5"
- **🚨 ALWAYS use VERIFIED NUMERIC VALUES from the color mapping above**
  - Example: "color 5 tiles" or "gray (5)" NOT "orange tiles" or "orange (7)"
  - Example: "color 2" or "red (2)" NOT just "red"
  - If unsure, refer to the color mapping - use the EXACT numeric value shown
- Include measurements: "2 cells wide", "3x3 square", "entire column"
- Handle variation: "the first object", "the second object" (works for any count)

**DESCRIBING PARTS OF OBJECTS (for partial transformations):**
- "the bottom three cells of the shape"
- "the top row of each object"
- "the left edge of the blue rectangle"
- "the middle column of the grid"
- "the outer border of each region"
- "starting from the bottom of the object, going up 5 cells"
- "across the entire width but only the bottom 2 rows"
- "the leftmost cell of each colored block"

Use relative positions within objects:
✓ "the bottom-left corner of the red shape"
✓ "the top half of each enclosed region"
✓ "the rightmost column of the entire grid"
✓ "cells adjacent to the green boundary"

**COLOR REFERENCE (always use name + number):**
- 0 = black (0)
- 1 = blue (1)
- 2 = red (2)
- 3 = green (3)
- 4 = yellow (4)
- 5 = gray (5)
- 6 = magenta (6)
- 7 = orange (7)
- 8 = cyan/light blue (8)
- 9 = maroon/dark red (9)

**WHEN TO USE REPEATABLE PATTERNS:**
- If action applies to VARIABLE number of objects → use "for each" or "repeat"
- If action applies to FIXED objects → be specific about each
- Steps should be EXECUTABLE AS WRITTEN

GOOD (Repeatable pattern):
✓ "For each black (0) region enclosed by green (3), fill with yellow (4)"
✓ "Repeat: Draw bridge from current block to next color in sequence. Continue for all blocks."

GOOD (Fixed objects with specific properties):
✓ "Draw bridge from red (2) block to blue (1) block using red (2)"
✓ "Draw bridge from blue (1) block to yellow (4) block using blue (1)"

BAD (Too sequential for variable count):
❌ "Fill first region... fill second region... fill third region... [7 steps]"
❌ "Draw first bridge, draw second bridge, draw third bridge..."

**GOOD EXAMPLES:**

Example A (Repeatable pattern with variable count):
'1. Find all black (0) regions that are completely enclosed by green (3) pixels (not touching image border)
 2. For each found region, fill every pixel of that region with yellow (4)
 3. Keep all green (3) pixels and border-touching black (0) pixels unchanged'

Example B (Reference object + repeatable):
'1. Identify the bottom row as color sequence reference: red (2), blue (1), yellow (4), green (3)
 2. For each colored block in the grid (left-to-right, top-to-bottom), draw a vertical bridge connecting it to the next color in the sequence, using the current block's color and matching its width'

Example C (Fixed objects with properties):
'1. For each maroon (9) outlined object, count the number of enclosed black (0) holes inside it
 2. Recolor that object based on hole count: 1 hole → blue (1), 2 holes → green (3), 3 holes → red (2)'

Example D (Partial transformation - only parts of objects change):
'1. For each blue (1) rectangular object in the grid, identify its bottom edge
 2. Extend yellow (4) cells downward from the bottom edge until reaching a boundary or the grid edge
 3. Keep the original blue (1) rectangle unchanged'

Example E (Partial transformation with measurements):
'1. For each red (2) shape, recolor only the top 2 rows to green (3)
 2. Leave the remaining bottom portion of each shape as red (2)
 3. Preserve all other colors unchanged'

**BAD EXAMPLES (Don't do this):**
'1. Fill first region, fill second region, fill third region...' ❌ (too sequential - what if there are 10 regions?)
'1. For each region, process it' ❌ (too vague - process HOW?)
'1. Draw all the bridges' ❌ (too vague - which bridges? how?)
'1. Set cell [2,3] to value 4' ❌ (coordinates, not visual)
'1. Run BFS to find connected components' ❌ (algorithm, not visual action)
'1. Allocate visited array and iterate...' ❌ (programming, not instruction)
'1. Scan top to bottom, left to right, for each pixel check if...' ❌ (implementation details)

**ABSOLUTELY FORBIDDEN:**
❌ NO algorithms: BFS, DFS, flood fill algorithms, graph traversal
❌ NO programming: loops, arrays, visited maps, mode calculations, "for i in range", "while touching"
❌ NO implementation details: "if touches_border is false then...", "allocate array", "mark visited"
❌ NO variables or flags: "touches_border", "visited[h][w]", "mode of pixels"
❌ Stay HUMAN-READABLE: A person should understand without coding knowledge

**IF YOU'RE TEMPTED TO WRITE AN ALGORITHM:**
→ STOP! Break it into MORE visual steps instead (use 6-8 steps if needed)

**YOUR TURN - TRANSLATE YOUR HYPOTHESIS:**

Your hypothesis was: [insert your hypothesis from above]

Now translate it into 2-8 steps (use MORE steps for complex rules):
1. WHAT to find: [describe visually with verified color values]
2. HOW to check/identify: [if complex, break this into multiple steps]
3. WHAT to do: [use "for each" if count varies]
4. WHAT to preserve: [what stays unchanged]

Remember: 8 visual steps > 3 algorithmic steps!

**DO NOT:**
- Add BFS/DFS/algorithms
- Add "visited arrays" or "mode calculations"
- Write implementation code
- Make it more complex than your hypothesis

**OUTPUT FORMAT:**
Numbered list, 2-4 VISUAL steps that DIRECTLY translate your hypothesis.
Use "for each" for repeatable patterns.
Include color values: black (0), green (3), yellow (4), etc.

**MANDATORY FINAL CHECK BEFORE SUBMITTING:**
After you write your steps, ask yourself:
1. "Do these steps DIRECTLY translate my hypothesis/rule?"
2. "Did I add any complexity that wasn't in my original rule?"
3. "Would these steps produce the EXACT outputs I saw?"
4. "Am I using 'for each' to handle variation in object count?"

If ANY answer is "no", REVISE the steps to match your hypothesis EXACTLY."""
        
        steps_response = self.call_ai(steps_prompt, all_images_with_outputs)
        
        # Parse steps
        universal_steps = []
        for line in steps_response.split('\n'):
            if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                universal_steps.append(line.strip())
        
        print(f"Generated {len(universal_steps)} universal steps:")
        for i, step in enumerate(universal_steps, 1):
            try:
                print(f"  {i}. {step[:80]}{'...' if len(step) > 80 else ''}")
            except:
                print(f"  {i}. [Step {i}]")
        
        # VERIFICATION: Check if steps align with the rule
        # DISABLED FOR TESTING - Was making steps worse by forcing oversimplification
        # print("\nVerifying steps match the validated rule...")
        # verification = self.call_ai(verify_prompt, [])
        # if "MISMATCH" in verification.upper():
        #     ... regenerate steps ...
        
        print("\n[Verification disabled for testing - using initial generated steps]")
        
        # PHASE 3: Generate step-by-step booklets for each training example
        print("\nPhase 3: Generating Step-by-Step Booklets...")
        print("-" * 80)
        
        # HISTORY MODE DISABLED - Each execution is a fresh API call
        # AI only gets what's in the current prompt (no memory of rule/goal between steps)
        self.conversation_history = []  # Not used when use_history=False
        
        validation_results = []
        booklet_data = []
        
        for idx, example in enumerate(training_examples):
            print(f"\nGenerating booklet for Example {idx + 1}...")
            
            # Create booklet directory for this example
            example_dir = output_path / f"example_{idx + 1}_booklet"
            example_dir.mkdir(exist_ok=True)
            
            # Save input image
            input_img_path = str(example_dir / "input.png")
            img_input = grid_to_image(example['input'], 30)
            img_input.save(input_img_path)
            
            # Save expected output image
            expected_img_path = str(example_dir / "target_output.png")
            img_expected = grid_to_image(example['output'], 30)
            img_expected.save(expected_img_path)
            
            # Execute steps one-by-one to create booklet
            # NOTE: Using FRESH CONTEXT for blind execution (no memory of reasoning phase)
            current_grid = [row[:] for row in example['input']]
            booklet_steps = []
            failed_step_idx = None  # Track which step fails
            
            for step_idx, step in enumerate(universal_steps):
                print(f"    Step {step_idx + 1}/{len(universal_steps)}...")
                
                # Create image of current state
                current_img_path = str(temp_dir / f"{task_name}_ex{idx+1}_step{step_idx}_current.png")
                img_current = grid_to_image(current_grid, 30)
                img_current.save(current_img_path)
                
                # Execute this single step with full reasoning context
                execute_prompt = f"""🖼️ **LOOK AT THE CURRENT GRID IMAGE ABOVE FIRST**

**YOUR COMPLETE REASONING ANALYSIS:**
{reasoning_analysis}

**UNIVERSAL STEPS (based on your reasoning):**
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(universal_steps))}

**NOW EXECUTE ONLY THIS STEP:**
STEP {step_idx + 1} of {len(universal_steps)}:
{step}

**BEFORE EXECUTING:**
1. **LOOK at the IMAGE above** - this is the current state of the grid
2. Visually identify the objects/regions/patterns in the image
3. Apply this step to what you SEE in the image
4. Keep the full analysis and goal in mind

Current grid state (see IMAGE above for visual, grid data below for reference):
{self.format_grid(current_grid)}

Apply this step to the VISUAL IMAGE above.

**THINK VISUALLY:**
- Work from the IMAGE you see above
- Identify objects/regions/patterns VISUALLY in the image
- "black (0) regions" = look for ALL black areas in image
- "enclosed by green (3)" = look for green boundaries in image
- "for each" = find ALL matching objects and apply action to ALL of them
- "fill with yellow (4)" = change those pixels to yellow (4)

**IF STEP SAYS "FOR EACH":**
- Find ALL matching objects in the image
- Apply the action to EVERY single one
- Don't stop after the first few - get them ALL
- Example: "for each enclosed region" = find region 1, fill it... find region 2, fill it... find region 3, fill it... until no more regions

CRITICAL EXECUTION RULES:
- If step says "for each" or "repeat", do it for ALL matching objects
- Apply ONLY this step - nothing more
- Do NOT remove or move cells unless step explicitly requires it
- PRESERVE DETAILS: Don't accidentally modify objects unless step requires it - keep exact shapes
- CONSISTENCY: Apply transformation IDENTICALLY to all similar objects
- COMPLETENESS: Don't miss ANY objects - count carefully and check entire image
- EXACT SHAPES: If copying/transforming shapes, preserve exact dimensions
- You MUST output a complete grid

Output the resulting grid using the color VALUES (0-9).

RESULT GRID:
[row1]
[row2]
..."""
                
                # BLIND EXECUTION: Fresh API call (no context from reasoning)
                # Execute WITHOUT history - only gets current prompt
                response = self.call_ai(execute_prompt, [current_img_path], use_history=False)
                new_grid = self.parse_grid_from_response(response)
                
                if new_grid:
                    current_grid = new_grid
                    
                    # Save step image
                    step_img_path = str(example_dir / f"step_{step_idx:03d}.png")
                    img_step = grid_to_image(current_grid, 30)
                    img_step.save(step_img_path)
                    
                    # Check if this step reached target
                    reached_target = (current_grid == example['output'])
                    
                    # Track if this is where we diverged from target
                    if not reached_target and failed_step_idx is None:
                        failed_step_idx = step_idx
                    
                    # If not at target, save expected image
                    if not reached_target:
                        expected_step_path = str(example_dir / f"step_{step_idx:03d}_expected.png")
                        img_step_expected = grid_to_image(example['output'], 30)
                        img_step_expected.save(expected_step_path)
                    
                    booklet_steps.append({
                        "step_number": step_idx,
                        "description": step,
                        "grid_shape": [len(current_grid), len(current_grid[0]) if current_grid else 0],
                        "target_shape": [len(example['output']), len(example['output'][0]) if example['output'] else 0],
                        "reached_target": reached_target,
                        "tries": 1
                    })
                    
                    print(f"      {'✅' if reached_target else '📝'} Step {step_idx + 1} executed")
                else:
                    print(f"      ⚠️ Could not parse grid from step {step_idx + 1}")
                    if failed_step_idx is None:
                        failed_step_idx = step_idx
            
            # Check final result
            final_matches = current_grid == example['output'] if current_grid else False
            
            # Save booklet metadata
            booklet_meta = {
                "task_name": task_name,
                "example_number": idx + 1,
                "steps": booklet_steps,
                "total_steps": len(booklet_steps),
                "success": final_matches,
                "generated_at": datetime.now().isoformat()
            }
            
            with open(example_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(booklet_meta, f, indent=2)
            
            validation_results.append({
                "example_number": idx + 1,
                "success": final_matches,
                "result_grid": current_grid,
                "expected_grid": example['output'],
                "booklet_path": str(example_dir),
                "failed_step_idx": failed_step_idx,  # Which step first diverged
                "booklet_steps": booklet_steps  # All step results
            })
            
            booklet_data.append(booklet_meta)
            
            status = "✅ SUCCESS" if final_matches else f"❌ FAILED (diverged at step {failed_step_idx + 1})" if failed_step_idx is not None else "❌ FAILED"
            print(f"  {status} - Booklet saved to {example_dir.name}")
        
        successful_validations = sum(1 for v in validation_results if v['success'])
        print(f"\nValidation: {successful_validations}/{len(training_examples)} examples passed")
        
        # PHASE 4: Refine if needed
        max_refinements = 1  # Allow 1 refinement attempt
        refinement_count = 0
        refinement_history = []  # Track each refinement iteration
        
        # Record initial attempt
        refinement_history.append({
            "iteration": 0,
            "steps": universal_steps,
            "success_count": successful_validations,
            "failed_examples": [v['example_number'] for v in validation_results if not v['success']],
            "action": "initial_generation"
        })
        
        while successful_validations < len(training_examples) and refinement_count < max_refinements:
            refinement_count += 1
            print(f"\nPhase 4.{refinement_count}: Refining Steps (Attempt {refinement_count}/{max_refinements})...")
            print("-" * 80)
            
            # Show failed examples with step-level detail
            # NOTE: Refinement uses REASONING CONTEXT (has seen all outputs before)
            refine_images = []
            refine_prompt = f"""Your current steps work for {successful_validations}/{len(training_examples)} examples.

**CURRENT STEPS:**
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(universal_steps))}

**FAILED EXAMPLES (see visualizations with step-level analysis):**
"""
            
            for val in validation_results:
                if not val['success']:
                    ex_num = val['example_number']
                    ex = training_examples[ex_num - 1]
                    failed_step = val.get('failed_step_idx')
                    
                    # Create images
                    fail_input_path = str(temp_dir / f"{task_name}_fail{ex_num}_input.png")
                    fail_output_path = str(temp_dir / f"{task_name}_fail{ex_num}_output.png")
                    fail_expected_path = str(temp_dir / f"{task_name}_fail{ex_num}_expected.png")
                    
                    img_fail_in = grid_to_image(ex['input'], 30)
                    img_fail_in.save(fail_input_path)
                    
                    if val['result_grid']:
                        img_fail_out = grid_to_image(val['result_grid'], 30)
                        img_fail_out.save(fail_output_path)
                    
                    img_expected = grid_to_image(val['expected_grid'], 30)
                    img_expected.save(fail_expected_path)
                    
                    refine_images.extend([fail_input_path, fail_output_path if val['result_grid'] else fail_input_path, fail_expected_path])
                    
                    failure_detail = f" - DIVERGED AT STEP {failed_step + 1}" if failed_step is not None else ""
                    refine_prompt += f"\nExample {ex_num}{failure_detail}: Input → Your Output → Expected Output (see images)\n"
                    
                    # Show which step failed
                    if failed_step is not None:
                        refine_prompt += f"  Problem: Step {failed_step + 1} (\"{universal_steps[failed_step]}\") produced wrong result\n"
            
            # NEW REFINEMENT APPROACH: Break down original rule into more granular steps
            current_step_count = len(universal_steps)
            target_step_count = current_step_count + 1  # Add one more step each iteration
            
            refine_prompt += f"""

**YOUR COMPLETE STRUCTURED REASONING:**
{reasoning_analysis}

**VERIFIED COLOR MAPPING (use these EXACT values):**
{color_mapping}

🚨 **CRITICAL: Use the correct numeric color values from the mapping above!**
If your current steps have wrong color values, that's likely why they're failing.

**TASK: Your current {current_step_count} steps are failing. Break down your VALIDATED RULE into {target_step_count} MORE GRANULAR steps.**

Review your full reasoning above to understand:
- What the transformation actually does
- What differences exist between inputs
- What stays constant across all examples
- Your validated hypothesis and how you tested it
- **Most importantly: Use the CORRECT color values from the verified mapping**

**REFINEMENT STRATEGY:**
- Go back to your original rule/hypothesis
- Break it down into {target_step_count} smaller, more specific visual actions
- Each step should be EASIER to execute than before
- Make implicit operations explicit

**CRITICAL - DO NOT WRITE ALGORITHMS:**
❌ NO: "Allocate boolean visited array"
❌ NO: "Run BFS/DFS traversal"
❌ NO: "Calculate mode of border pixels"
❌ NO: "If touches_border is false then..."
❌ NO: Programming/implementation details

✅ YES: Visual, high-level actions broken into smaller pieces
✅ YES: "Find the reference region" THEN "For each other region..." (separate steps)
✅ YES: Making complex operations simpler by splitting them

**EXAMPLE:**

Original rule: "Move colored blocks to touch the red bar"
Current 2 steps (FAILED):
1. Find the red bar
2. Move all blocks to touch it

Refined to 3 steps (more granular):
1. Identify the long red (2) bar as the anchor
2. For each non-red colored object, determine the direction to move it toward the red bar
3. Slide each object in that direction until it touches the red bar

Refined to 4 steps (even more granular):
1. Identify the long red (2) bar and note if it's vertical or horizontal
2. For each non-red object, determine movement direction: horizontal if bar is vertical, vertical if bar is horizontal
3. Move each object in that direction until adjacent to the red bar
4. Preserve all object shapes and the red bar position

**YOUR TURN:**
Break down your original rule into {target_step_count} VISUAL, EXECUTABLE steps.
Include color values: black (0), red (2), green (3), yellow (4), etc.

**OUTPUT:**
Numbered list of exactly {target_step_count} visual steps."""
            
            refine_response = self.call_ai(refine_prompt, refine_images)
            
            # Parse refined steps
            universal_steps = []
            for line in refine_response.split('\n'):
                if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                    universal_steps.append(line.strip())
            
            print(f"Refined to {len(universal_steps)} steps")
            
            # Check if we got the expected number of steps
            if len(universal_steps) != target_step_count:
                print(f"⚠️  WARNING: Expected {target_step_count} steps but got {len(universal_steps)}")
                # Continue anyway - the AI might have a good reason
            
            # Re-validate by executing refined steps with full context
            print("\nRe-executing refined steps step-by-step with reasoning context...")
            validation_results = []
            
            for idx, example in enumerate(training_examples):
                print(f"  Re-validating Example {idx + 1}...")
                
                # Execute refined steps step-by-step with reasoning context
                current_grid = [row[:] for row in example['input']]
                
                for step_idx, step in enumerate(universal_steps):
                    # Create image of current state
                    current_img_path = str(temp_dir / f"{task_name}_refine_ex{idx+1}_step{step_idx}.png")
                    img_current = grid_to_image(current_grid, 30)
                    img_current.save(current_img_path)
                    
                    # Execute step WITH full reasoning context in prompt
                    execute_prompt = f"""🖼️ **LOOK AT THE CURRENT GRID IMAGE ABOVE FIRST**

**YOUR COMPLETE REASONING ANALYSIS:**
{reasoning_analysis}

**REFINED STEPS (based on your reasoning):**
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(universal_steps))}

**NOW EXECUTE ONLY THIS STEP:**
STEP {step_idx + 1} of {len(universal_steps)}:
{step}

**BEFORE EXECUTING:**
1. **LOOK at the IMAGE above** - this is the current state
2. Visually identify objects/regions/patterns in the image
3. Apply this step to what you SEE in the image
4. Keep the full analysis and goal in mind

Current grid state (IMAGE above for visual, grid data below for reference):
{self.format_grid(current_grid)}

Apply ONLY this step to the VISUAL IMAGE above.

**THINK VISUALLY:**
- Work from the IMAGE you see above
- Remember the overall goal from the rule
- "for each" = find ALL matching objects
- Apply this step completely before moving to next step

Output the transformed grid.

FINAL GRID:
[row1]
[row2]
..."""
                    
                    # Execute with reasoning context in prompt
                    response = self.call_ai(execute_prompt, [current_img_path], use_history=False)
                    new_grid = self.parse_grid_from_response(response)
                    
                    if new_grid:
                        current_grid = new_grid
                
                # Check if final result matches expected
                matches = current_grid == example['output'] if current_grid else False
                
                validation_results.append({
                    "example_number": idx + 1,
                    "success": matches,
                    "result_grid": current_grid,
                    "expected_grid": example['output']
                })
            
            successful_validations = sum(1 for v in validation_results if v['success'])
            print(f"Re-validation: {successful_validations}/{len(training_examples)} examples passed")
            
            # Track this refinement iteration
            refinement_history.append({
                "iteration": refinement_count,
                "steps": universal_steps,
                "success_count": successful_validations,
                "failed_examples": [v['example_number'] for v in validation_results if not v['success']],
                "action": "refined"
            })
        
        # PHASE 5: Apply to test cases
        # NOTE: AI has NEVER seen test outputs - this is TRUE blind execution!
        test_results = []
        
        if test_examples:
            print(f"\nPhase 5: Applying to Test Cases (True Blind Execution)...")
            print("-" * 80)
            print("(AI has never seen these outputs - unbiased validation)\n")
            
            for test_idx, test_ex in enumerate(test_examples):
                print(f"\nTest Case {test_idx + 1}/{len(test_examples)}...")
                
                test_input_path = str(temp_dir / f"{task_name}_test{test_idx+1}_input.png")
                img_test = grid_to_image(test_ex['input'], 30)
                img_test.save(test_input_path)
                
                for attempt in range(3):
                    test_prompt = f"""**YOUR COMPLETE REASONING ANALYSIS:**
{reasoning_analysis}

**UNIVERSAL STEPS (based on your reasoning):**
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(universal_steps))}

**CONTEXT:** You thoroughly analyzed this puzzle type. Now apply your steps to this test input,
keeping in mind the rule and variations you identified.

Test Input (see IMAGE above):
{self.format_grid(test_ex['input'])}

Apply ALL steps in order to transform this input.

CRITICAL EXECUTION RULES:
- Follow each step completely - if it says "for each", do ALL of them
- Apply steps in the exact order given
- Do NOT remove or move cells unless steps explicitly require it
- PRESERVE DETAILS: Keep exact shapes of objects
- CONSISTENCY: Apply transformations IDENTICALLY to all similar objects
- COMPLETENESS: Don't miss single-cell blocks or small details - count carefully
- EXACT SHAPES: If copying/transforming shapes, preserve their exact dimensions
- You MUST output a complete grid

Output the FINAL transformed grid.

FINAL GRID:
[row1]
[row2]
..."""
                    
                    # Execute WITHOUT history - only gets current prompt
                    response = self.call_ai(test_prompt, [test_input_path], use_history=False)
                    result_grid = self.parse_grid_from_response(response)
                    matches = result_grid == test_ex['output'] if result_grid else False
                    
                    if matches:
                        print(f"  ✅ SUCCESS on attempt {attempt + 1}")
                        test_results.append({
                            "test_number": test_idx + 1,
                            "attempts": attempt + 1,
                            "success": True
                        })
                        break
                else:
                    print(f"  ❌ FAILED all attempts")
                    test_results.append({
                        "test_number": test_idx + 1,
                        "attempts": 3,
                        "success": False
                    })
        
        # Save results
        test_success = sum(1 for t in test_results if t['success']) if test_results else 0
        
        # Determine success status
        training_complete = (successful_validations == len(training_examples))
        test_complete = (test_success == len(test_examples)) if test_examples else False
        overall_success = training_complete and (test_success > 0 if test_examples else True)
        
        results = {
            "task_name": task_name,
            "approach": "batch_visual_structured_reasoning",
            "reasoning_analysis": reasoning_analysis,
            "color_mapping": color_mapping,  # Verified color values
            "color_values_present": color_values_sorted,  # All numeric values used
            "universal_steps": universal_steps,
            "training_examples": len(training_examples),
            "training_success": successful_validations,
            "training_complete": training_complete,  # All training examples passed
            "refinement_iterations": refinement_count,
            "refinement_history": refinement_history,  # Full refinement tracking
            "booklets": booklet_data,
            "test_results": test_results,
            "test_success": test_success,
            "test_complete": test_complete,  # All test cases passed
            "total_test_cases": len(test_examples),
            "overall_success": overall_success,  # Training complete + at least 1 test passed
            "generated_at": datetime.now().isoformat()
        }
        
        # Save JSON
        with open(output_path / "batch_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        # Save README
        readme = f"""# Batch Visual Booklet - {task_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Approach
This booklet was generated using the "Batch Visual Structured Reasoning" approach:
- Showed AI ALL training examples (inputs + outputs) simultaneously
- AI followed structured reasoning: Observe → Compare → Hypothesize → Validate
- Generated universal steps considering entire problem space
- Created step-by-step visual booklets for each training example
- Refined steps if needed (up to 3 iterations)

## Results

Training: {successful_validations}/{len(training_examples)} examples solved
Test: {test_success}/{len(test_examples)} cases solved
Refinement Iterations: {refinement_count}

## Universal Steps

{chr(10).join(f"{i}. {step}" for i, step in enumerate(universal_steps, 1))}

## Visual Booklets Generated

{chr(10).join(f"- Example {i+1}: example_{i+1}_booklet/ ({len(booklet_data[i]['steps'])} steps)" for i in range(len(booklet_data)))}

Each booklet contains:
- input.png: Original input
- target_output.png: Expected output
- step_NNN.png: Model output after each step
- step_NNN_expected.png: Expected output (if step didn't reach target)
- metadata.json: Complete step data

## Why This Approach May Improve Generalization

1. **Upfront Problem Space View**: AI sees all variations before generating steps
2. **Structured Reasoning**: Follows human-like cognitive process
3. **Pattern Extraction**: Identifies what's constant vs what varies across ALL examples
4. **Avoids Sequential Bias**: Doesn't overfit to early examples
5. **Visual Reasoning**: Explicit analysis of visual patterns
6. **Universal Generation**: Steps are general from the start

## View in Streamlit

```bash
streamlit run streamlit_booklet_viewer.py
```

Select any example_N_booklet folder to view step-by-step visualizations.

## Test Results

"""
        
        if test_results:
            for t in test_results:
                status = "SUCCESS" if t['success'] else "FAILED"
                readme += f"Test {t['test_number']}: {status} (attempts: {t['attempts']})\n"
        else:
            readme += "No test cases in this puzzle.\n"
        
        with open(output_path / "README.txt", 'w', encoding='utf-8') as f:
            f.write(readme)
        
        print(f"\n{'='*80}")
        print("BATCH VISUAL BOOKLET COMPLETE")
        print(f"{'='*80}")
        print(f"Training Success: {successful_validations}/{len(training_examples)}")
        print(f"Test Success: {test_success}/{len(test_examples)}")
        print(f"Refinement Iterations: {refinement_count}")
        print(f"Output: {output_path}")
        print(f"{'='*80}\n")
        
        return results


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python arc-booklet-batch-visual.py <task_json_file> [output_dir]")
        print("Example: python arc-booklet-batch-visual.py ../saturn-arc/ARC-AGI-2/ARC-AGI-1/data/training/00d62c1b.json")
        sys.exit(1)
    
    task_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "batch_visual_booklets"
    
    if not os.path.exists(task_file):
        print(f"Error: Task file '{task_file}' not found")
        sys.exit(1)
    
    try:
        generator = ARCBatchVisualGenerator()
        generator.generate_batch_booklet(task_file, output_dir)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

