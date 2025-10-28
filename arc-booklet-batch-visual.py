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
    
    def call_ai(self, text_prompt: str, image_paths: List[str] = None) -> str:
        """Call OpenAI API with text and optional images"""
        content = [{"type": "text", "text": text_prompt}]
        
        if image_paths:
            for img_path in image_paths:
                base64_img = self.encode_image(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                })
        
        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": content}]
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
- What CHANGED from input to output? (added? removed? moved? recolored?)
- What STAYED THE SAME? (dimensions? certain objects? background?)

Example 2:
- [Same questions]

(Repeat for all examples)

═══════════════════════════════════════════════════════════════════════════
STAGE 2: FIND THE PATTERN
═══════════════════════════════════════════════════════════════════════════
Now look ACROSS all examples:

CONSTANTS (True for EVERY example):
- Output dimensions: [same as input? different? specific size?]
- Objects preserved: [what objects appear in both input and output?]
- Transformation type: [fill? move? copy? recolor? add? remove?]
- Spatial rules: [positions? alignments? distances?]

VARIANTS (Different between examples):
- What varies: [object count? sizes? positions? colors?]
- Range of variation: [2-5 objects? sizes 1-3? etc.]

CRITICAL: Find what's UNIVERSAL despite the variations.

═══════════════════════════════════════════════════════════════════════════
STAGE 3: FORM & TEST HYPOTHESIS
═══════════════════════════════════════════════════════════════════════════

Based on the patterns, state THE RULE in ONE sentence:
"The rule is: [FILL IN - be specific but general]"

Now TEST this rule example-by-example:

Example 1: If I apply "[your rule]" to Example 1's input...
  → Would I get Example 1's output? [YES/NO and why]
  
Example 2: If I apply "[your rule]" to Example 2's input...
  → Would I get Example 2's output? [YES/NO and why]

(Test all examples)

If ANY example fails → REVISE your rule and test again.

═══════════════════════════════════════════════════════════════════════════
STAGE 4: EDGE CASES & REFINEMENT
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

2. UNIVERSAL CONSTANTS: What's true for EVERY transformation

3. THE RULE: One clear sentence describing the transformation

4. VALIDATION: For each example, confirm your rule produces the correct output

5. CONFIDENCE: How certain are you this rule is correct? Any ambiguities?"""
        
        reasoning_analysis = self.call_ai(reasoning_prompt, all_image_paths)
        
        try:
            print(f"Reasoning Analysis: {reasoning_analysis[:400]}...\n")
        except:
            print(f"Reasoning Analysis: [Generated - {len(reasoning_analysis)} chars]\n")
        
        # PHASE 2: Generate universal steps based on validated hypothesis
        print("Phase 2: Generating Universal Steps from Validated Hypothesis...")
        print("-" * 80)
        
        all_images_with_outputs = all_image_paths  # Already have all images
        
        steps_prompt = f"""Based on your structured reasoning, generate UNIVERSAL STEPS that work for ALL {len(training_examples)} training examples.

**Your Structured Reasoning:**
{reasoning_analysis}

**ALL TRAINING EXAMPLES (see images - input/output pairs):**
"""
        
        for i in range(len(training_examples)):
            steps_prompt += f"\nExample {i+1}: Input → Output\n"
        
        steps_prompt += f"""

TASK: Using THE RULE you validated above, break it into 3-7 CONCRETE, EXECUTABLE steps.

**CRITICAL REQUIREMENTS:**
1. Base steps on THE RULE from your reasoning (don't ignore your analysis!)
2. Steps must work for ALL {len(training_examples)} examples, handling the VARIANTS you identified
3. Use the CONSTANTS you found to make steps universal
4. ONE CONCRETE ACTION PER STEP (not "for each" loops)

**HOW TO CONVERT YOUR RULE INTO STEPS:**

If your rule is: "Fill each enclosed green-framed region with yellow"
Then steps should be:
'1. Identify the first green-framed region
 2. Fill that enclosed region with yellow
 3. Identify the second green-framed region
 4. Fill that enclosed region with yellow
 (etc. - one region per step)'

NOT:
'1. For each green-framed region, fill with yellow' ❌ (too high-level, not executable)

**STEP PHRASING GUIDE:**
- Use VISUAL SPATIAL language: "the bottom row", "each colored block", "enclosed regions"
- NOT coordinates: avoid "cell at (x,y)" or "row 3, column 5"
- **ALWAYS include color VALUES:** "black (0)", "green (3)", "yellow (4)", "red (2)"
- Include measurements: "2 cells wide", "3x3 square", "entire column"
- Handle variation: "the first object", "the second object" (works for any count)

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

**ONE CONCRETE ACTION PER STEP:**
- Each step = ONE SPECIFIC ACTION (e.g., "draw the first bridge", not "draw all bridges")
- If same action on multiple objects, ONE step per object
- Steps should be concrete: "Draw bridge from red to blue using red" NOT "Draw bridges"
- NO pseudocode: NO "for each...", NO "if/else", NO loops
- Each step produces visible change

**GOOD EXAMPLES:**

Example A (Reference object + bridges):
'1. Identify the bottom row as reference: red, blue, yellow, green
 2. Draw bridge from red block to blue block using red, matching block width
 3. Draw bridge from blue block to yellow block using blue
 4. Draw bridge from yellow block to green block using yellow  
 5. Draw bridge from green block back to red block using green'

Example B (Object recoloring based on property):
'1. Count holes in first maroon-framed object (2 holes), recolor entire object to color 3
 2. Count holes in second maroon object (1 hole), recolor to color 1
 3. Count holes in third maroon object (3 holes), recolor to color 2'

Example C (Fill enclosed regions):
'1. Locate first region enclosed by green frame
 2. Fill that region with yellow
 3. Locate second region enclosed by green frame
 4. Fill that region with yellow'

**BAD EXAMPLES (Don't do this):**
'1. For each color pair in sequence, draw bridge' ❌ (pseudocode)
'1. Draw all the bridges' ❌ (too vague)
'1. Set cell [2,3] to value 4' ❌ (coordinates, not visual)
'1. Run BFS to find connected components' ❌ (algorithm, not visual action)
'1. Allocate visited array and iterate...' ❌ (programming, not instruction)

**ABSOLUTELY FORBIDDEN:**
❌ NO algorithms: BFS, DFS, flood fill algorithms, graph traversal
❌ NO programming: loops, arrays, visited maps, mode calculations
❌ NO implementation details: "if touches_border is false then..."
❌ Stay HUMAN-READABLE: A person should understand without coding knowledge

**OUTPUT FORMAT:**
Numbered list, 3-7 steps. Each step is one concrete, VISUAL action (NOT code, NOT algorithms)."""
        
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
        
        # PHASE 3: Generate step-by-step booklets for each training example
        print("\nPhase 3: Generating Step-by-Step Booklets...")
        print("-" * 80)
        
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
                
                # Execute this single step (BLIND - separate context, never saw outputs)
                execute_prompt = f"""Look at the CURRENT GRID IMAGE above and execute this visual transformation step.

STEP {step_idx + 1} of {len(universal_steps)}:
{step}

Current grid (see IMAGE above, also as text):
{self.format_grid(current_grid)}

Apply this ENTIRE step to the current grid VISUALLY. If this step has multiple parts (e.g., "for each X do Y"), complete ALL iterations in this single step.

**THINK VISUALLY:**
- Work from the IMAGE you see above
- Identify objects/regions/patterns VISUALLY in the image
- "black (0) regions" = look for black areas in image
- "enclosed by green (3)" = look for green boundaries in image
- "fill with yellow (4)" = change those pixels to yellow

CRITICAL EXECUTION RULES:
- Execute the COMPLETE step - if it says "for each", do ALL of them now
- If it has conditionals (if/else), apply them ALL in this step
- Apply ONLY this step - nothing more
- Do NOT remove or move cells unless this step explicitly requires it
- PRESERVE DETAILS: Don't accidentally modify objects unless this step explicitly requires it - keep their exact shapes
- CONSISTENCY: If step applies to multiple objects, apply it IDENTICALLY to ALL of them
- COMPLETENESS: Don't miss single-cell blocks or small details - count carefully
- EXACT SHAPES: If copying/transforming shapes, preserve their exact dimensions
- You MUST output a complete grid

Output the resulting grid using the color VALUES (0-9).

RESULT GRID:
[row1]
[row2]
..."""
                
                # BLIND EXECUTION: Fresh API call (no context from reasoning)
                response = self.call_ai(execute_prompt, [current_img_path])
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
        max_refinements = 3
        refinement_count = 0
        
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
            
            refine_prompt += f"""

**TASK: MODIFY your steps to work for ALL examples, including the failed ones above.**

**CRITICAL - DO NOT WRITE ALGORITHMS:**
❌ NO: "Allocate boolean visited array"
❌ NO: "Run BFS/DFS traversal"
❌ NO: "Calculate mode of border pixels"
❌ NO: "If touches_border is false then..."
❌ NO: Programming/implementation details

✅ YES: Visual, high-level actions
✅ YES: "Find first black region enclosed by green"
✅ YES: "Fill that region with yellow (4)"
✅ YES: Human-readable instructions

**HOW TO REFINE:**
- Look at visual differences in the images
- Identify what's missing or incorrect VISUALLY
- Make steps more specific about WHAT to do, not HOW to implement
- Use color names WITH numbers: "black (0)", "yellow (4)", "green (3)"
- Break into more granular visual steps if needed
- Keep language simple and executable

**STAY VISUAL:**
- "Find each enclosed black region" NOT "Run connected components algorithm"
- "Fill with yellow (4)" NOT "Set pixels to value 4"
- "The first hole", "the second hole" NOT "For i in range(n)"

Output REFINED steps (numbered list, 3-7 VISUAL steps, NOT code)."""
            
            refine_response = self.call_ai(refine_prompt, refine_images)
            
            # Parse refined steps
            universal_steps = []
            for line in refine_response.split('\n'):
                if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                    universal_steps.append(line.strip())
            
            print(f"Refined to {len(universal_steps)} steps")
            
            # Re-validate
            validation_results = []
            for idx, example in enumerate(training_examples):
                val_input_path = str(temp_dir / f"{task_name}_val{idx+1}_input.png")
                img_val = grid_to_image(example['input'], 30)
                img_val.save(val_input_path)
                
                validate_prompt = f"""Apply these refined steps to this input.

STEPS:
{chr(10).join(universal_steps)}

Input:
{self.format_grid(example['input'])}

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

Output the FINAL grid.

FINAL GRID:
[row1]
[row2]
..."""
                
                response = self.call_ai(validate_prompt, [val_input_path])
                result_grid = self.parse_grid_from_response(response)
                matches = result_grid == example['output'] if result_grid else False
                
                validation_results.append({
                    "example_number": idx + 1,
                    "success": matches,
                    "result_grid": result_grid,
                    "expected_grid": example['output']
                })
            
            successful_validations = sum(1 for v in validation_results if v['success'])
            print(f"Re-validation: {successful_validations}/{len(training_examples)} examples passed")
        
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
                    test_prompt = f"""Apply these universal steps to the test input.

UNIVERSAL STEPS:
{chr(10).join(universal_steps)}

Test Input:
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
                    
                    response = self.call_ai(test_prompt, [test_input_path])
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
        
        results = {
            "task_name": task_name,
            "approach": "batch_visual_structured_reasoning",
            "reasoning_analysis": reasoning_analysis,
            "universal_steps": universal_steps,
            "training_examples": len(training_examples),
            "training_success": successful_validations,
            "refinement_iterations": refinement_count,
            "booklets": booklet_data,
            "test_results": test_results,
            "test_success": test_success,
            "total_test_cases": len(test_examples),
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

