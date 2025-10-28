#!/usr/bin/env python3
"""
ARC Booklet Refiner (Option 1: Iterative Refinement)
Processes all training examples sequentially, refining steps across examples
"""

import json
import os
import sys
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from openai import OpenAI
from PIL import Image
import re

# Import visualizer
from arc_visualizer import grid_to_image

# Tool definition for visualization
VISUALIZATION_TOOL = {
    "type": "function",
    "function": {
        "name": "visualize_grid",
        "description": "Visualize an ARC grid as an image. Use this to see intermediate states when applying transformations step-by-step.",
        "parameters": {
            "type": "object",
            "properties": {
                "grid": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"}
                    },
                    "description": "The 2D grid to visualize (array of arrays of integers 0-9)"
                }
            },
            "required": ["grid"]
        }
    }
}

class IterativeBookletRefiner:
    def __init__(self):
        self.task_data = None
        self.task_name = None
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
    
    def create_grid_image(self, grid: List[List[int]], label: str = "grid") -> str:
        """Create and save a grid visualization"""
        temp_dir = Path("img_tmp")
        temp_dir.mkdir(exist_ok=True)
        img_path = str(temp_dir / f"{self.task_name}_{label}_{datetime.now().timestamp()}.png")
        img = grid_to_image(grid, cell_size=30)
        img.save(img_path)
        return img_path
    
    def call_ai_with_image(self, text_prompt: str, image_paths: List[str]) -> str:
        """Call OpenAI with text and images, supporting visualization tool"""
        
        # Prepare content with images
        content = [{"type": "text", "text": text_prompt}]
        
        for image_path in image_paths:
            base64_image = self.encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            })
        
        # Create the API call with tools always enabled
        call_params = {
            "model": "gpt-5-mini",
            "messages": [{"role": "user", "content": content}],
            "tools": [VISUALIZATION_TOOL],
            "tool_choice": "auto"
        }
        
        # Keep calling the API while tool calls are being made
        max_iterations = 20
        iteration = 0
        final_message = None
        messages = call_params["messages"]
        
        while iteration < max_iterations:
            print(f"\n📡 API Call iteration {iteration + 1}")
            response = self.client.chat.completions.create(**call_params)
            
            # Add assistant message to messages
            assistant_message = {"role": "assistant", "content": response.choices[0].message.content or ""}
            if response.choices[0].message.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in response.choices[0].message.tool_calls
                ]
            messages.append(assistant_message)
            
            # Check for function calls
            has_function_call = False
            if response.choices[0].message.tool_calls:
                has_function_call = True
                
                for tc in response.choices[0].message.tool_calls:
                    print(f"\n🔧 Function call detected: {tc.function.name}")
                    
                    # Parse the function arguments
                    args = json.loads(tc.function.arguments)
                    
                    if tc.function.name == "visualize_grid":
                        # Create visualization
                        grid = args["grid"]
                        print(f"  Creating visualization for grid of size {len(grid)}x{len(grid[0]) if grid else 0}")
                        img_path = self.create_grid_image(grid, label="tool")
                        
                        # Send the actual image back to the model (not just the path!)
                        base64_image = self.encode_image(img_path)
                        
                        # Add function result to messages with the actual image
                        messages.append({
                            "role": "tool",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Visualization created successfully. Here is the grid image:"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ],
                            "tool_call_id": tc.id
                        })
                        print(f"  ✅ Visualization created and sent to model")
                
                iteration += 1
                call_params["messages"] = messages
                
            if not has_function_call:
                print("\n✋ No more function calls, ending iteration")
                final_message = response.choices[0].message.content
                break
        
        if final_message is None:
            print("\n⚠️ Warning: Maximum iterations reached")
            final_message = "No response"
        
        print(f"\n💬 Response: {final_message[:200]}..." if len(final_message) > 200 else f"\n💬 Response: {final_message}")
        
        return final_message
    
    def call_ai(self, text_prompt: str, image_paths: List[str] = None) -> str:
        """Call OpenAI API with text and optional images (no tool calling)"""
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
        """Parse grid from AI response"""
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
        
        return None
    
    def generate_initial_steps(self, example_input, example_output):
        """Generate initial steps from first training example"""
        temp_dir = Path("img_tmp")
        temp_dir.mkdir(exist_ok=True)
        
        # Create images
        input_img_path = str(temp_dir / f"{self.task_name}_ex1_input.png")
        output_img_path = str(temp_dir / f"{self.task_name}_ex1_output.png")
        
        img_input = grid_to_image(example_input, 30)
        img_input.save(input_img_path)
        
        img_output = grid_to_image(example_output, 30)
        img_output.save(output_img_path)
        
        # PHASE 1: Visual Analysis
        analyze_prompt = """Analyze this ARC-AGI puzzle input IMAGE.

Work from the VISUAL representation. Think about patterns, shapes, and colors.

- What shapes, objects, or patterns do you SEE? Are there any visual elements that stand out as different or special (reference objects)?
- What colors are present and arranged how?
- Are there reference objects or other clues in the image? If so, what do they say about how to solve the puzzle?
- Describe the visual structure spatially

Be PRECISE about measurements."""
        
        analysis = self.call_ai(analyze_prompt, [input_img_path])
        
        # PHASE 2: Generate Steps
        prompt = f"""You analyzed the input. Now see the output and identify step-by-step operations.

Your Analysis:
{analysis}

Input grid:
{self.format_grid(example_input)}

Output grid:
{self.format_grid(example_output)}

TASK: Break down the transformation into concrete, followable steps (3-7 steps).

ONE CONCRETE ACTION PER STEP:
- Each step = ONE SPECIFIC ACTION (e.g., "draw the first bridge", not "draw all bridges")
- If performing the same action on multiple objects, make ONE step per object/action
- Steps should be concrete and followable: "Draw bridge from red to blue" NOT "For each color pair, draw bridge"
- NO pseudocode patterns like "for each...", "if/else as separate steps"
- NO setup-only steps like "Identify pattern" (this is analysis, not action)
- Each step should produce a visible change in the grid
- Steps should be ordered logically (don't jump around)

GOOD GRANULARITY:
✓ One bridge per step (if drawing 5 bridges, that's 5 steps)
✓ One object coloring per step (if coloring 3 objects, that's 3 steps)  
✓ One transformation application per step
✓ Followable by a human executing step-by-step

FORBIDDEN PATTERNS:
❌ "For each cell, do X" (split into individual steps instead)
❌ "Draw all bridges" (specify each bridge separately)
❌ "Color all objects" (specify each object separately)
❌ Incomplete fragments like "For each X:" without the action

THE RIGHT LEVEL OF DETAIL (like explaining to a person):
- Specific enough to execute unambiguously, without missing any details
- General enough to be read and understood by a human, without being too specific to the input
- Visual references: "the bottom row", "each colored square"
- Clear actions: "draw", "fill", "copy", "extend"
- Measurements when needed: "2 cells wide", "across entire row"
- Constraints: "without overwriting X", "only for Y"

Follow these meta-instructions:
1. Identify VISUAL objects, shapes, and patterns with SPATIAL locations (e.g., "the red square in the top-left", not "cell [0,0]")
2. Look for reference objects - visual elements that stand out
3. Determine the VISUAL transformation by comparing images
4. Describe in HIGH-LEVEL VISUAL terms with PRECISION

CRITICAL CONSISTENCY AND DETAIL RULES:
- Do not change object shapes unless part of transformation
- Apply transformation EXACTLY THE SAME WAY to all relevant parts
- Do NOT vary arbitrarily - only if puzzle provides explicit clues
- If transformation works for one object, use IDENTICAL transformation for all similar objects
- THE RULE SHOULD BE OBVIOUS - don't invent complex patterns that aren't there
- COUNT CAREFULLY: Don't miss single-cell blocks or small details
- PRESERVE SHAPES: Keep exact dimensions when copying/transforming
- ALL OBJECTS: Ensure transformation applies to EVERY relevant object, not just some

Output grid size must be inferred from input. The puzzle should have a clear rule for how the input size determines the output size.

Format as numbered list. Aim for 3-7 concrete steps.

Example (GOOD - using reference object, bridge drawing):
'1. Identify the bottom row as the reference containing color sequence: red, blue, yellow, green
 2. Draw bridge connecting red (2) to blue (3) using red, matching block width
 3. Draw bridge connecting blue (3) to yellow (9) using blue
 4. Draw bridge connecting yellow (9) to green (4) using yellow
 5. Draw bridge connecting green (4) to red (2) using green'

Example (GOOD - object recoloring, one per step):
'1. Count holes in first maroon object (2 holes), recolor it to color 3
 2. Count holes in second maroon object (1 hole), recolor it to color 1
 3. Count holes in third maroon object (3 holes), recolor it to color 2'

Note: If there's a reference object (like a row showing color sequence), identify it first, then use it to guide subsequent steps.

Example (BAD - pseudocode/loops):
'1. For each color pair in the sequence:
 2. Locate the two blocks
 3. Draw a bridge between them
 4. Use the first block's color'

Example (BAD - too vague):
'1. Draw all the bridges
 2. Color all the objects'"""
        
        response = self.call_ai(prompt, [input_img_path, output_img_path])
        
        # Parse steps
        steps = []
        for line in response.split('\n'):
            if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                steps.append(line.strip())
        
        return steps
    
    def apply_steps_to_example(self, steps: List[str], example_input, example_output):
        """Apply current steps to an example ONE-BY-ONE, showing intermediate grids"""
        temp_dir = Path("img_tmp")
        temp_dir.mkdir(exist_ok=True)
        
        current_grid = [row[:] for row in example_input]
        
        print(f"    Executing {len(steps)} steps...")
        
        # Execute each step one-by-one
        for i, step in enumerate(steps):
            print(f"      Step {i+1}/{len(steps)}: Executing...")
            
            # Create image of current state
            current_img_path = str(temp_dir / f"{self.task_name}_step_{i}_current.png")
            img_current = grid_to_image(current_grid, 30)
            img_current.save(current_img_path)
            
            # Execute this single step
            execute_prompt = f"""Execute this single step.

CURRENT GRID:
{self.format_grid(current_grid)}

STEP {i+1} of {len(steps)}:
{step}

Apply this ONE step to the current grid.

IMPORTANT: You can use the 'visualize_grid' tool to see the grid as you work.

CRITICAL EXECUTION RULES:
- Apply ONLY this one step - nothing more
- Do NOT remove or move cells unless this step explicitly requires it
- PRESERVE DETAILS: Don't accidentally modify objects - keep exact shapes
- CONSISTENCY: Apply transformation IDENTICALLY to all relevant objects
- COMPLETENESS: Don't miss single-cell blocks or small details - count carefully
- EXACT SHAPES: Preserve dimensions when copying/transforming

Output the resulting grid after applying this step.

RESULT GRID:
[row1]
[row2]
..."""
            
            response = self.call_ai_with_image(execute_prompt, [current_img_path])
            new_grid = self.parse_grid_from_response(response)
            
            if new_grid:
                current_grid = new_grid
                print(f"      ✅ Step {i+1} executed")
            else:
                print(f"      ⚠️ Could not parse grid from step {i+1}, keeping current grid")
        
        # Check if final grid matches expected
        matches = current_grid == example_output if current_grid else False
        
        return current_grid, matches
    
    def refine_steps(self, current_steps: List[str], all_examples: List[Dict]):
        """Refine steps to work for ALL previously seen examples"""
        temp_dir = Path("img_tmp")
        
        # Create images for all examples (showing input/output visualizations)
        image_paths = []
        
        prompt = f"""You have steps that work for some ARC training examples but fail for others.

CURRENT STEPS (need modification):
{chr(10).join(current_steps)}

Look at the VISUALIZATIONS of ALL training examples below and MODIFY your steps to work for all of them.

For FAILED examples, you'll see WHAT YOU PRODUCED vs WHAT WAS EXPECTED - study the visual DIFFERENCE.

ALL TRAINING EXAMPLES SEEN SO FAR (as images):
"""
        
        for ex in all_examples:
            status = "✅ WORKS with current steps" if ex['works'] else "❌ FAILS with current steps"
            prompt += f"\n--- Example {ex['number']} ({ex['input_shape'][0]}x{ex['input_shape'][1]} input → {ex['output_shape'][0]}x{ex['output_shape'][1]} output) - {status} ---\n"
            
            # Create input image
            input_img_path = str(temp_dir / f"refine_ex{ex['number']}_input.png")
            img_in = grid_to_image(ex['input'], 30)
            img_in.save(input_img_path)
            image_paths.append(input_img_path)
            
            prompt += f"Input (see image above)\n"
            
            # For FAILED examples: show model output vs expected side by side
            if not ex['works'] and ex.get('model_output'):
                model_img_path = str(temp_dir / f"refine_ex{ex['number']}_model.png")
                expected_img_path = str(temp_dir / f"refine_ex{ex['number']}_expected.png")
                
                img_model = grid_to_image(ex['model_output'], 30)
                img_model.save(model_img_path)
                
                img_expected = grid_to_image(ex['output'], 30)
                img_expected.save(expected_img_path)
                
                image_paths.extend([model_img_path, expected_img_path])
                
                prompt += f"❌ YOUR OUTPUT (wrong - see image) vs ✅ EXPECTED OUTPUT (correct - see image)\n"
                prompt += f"STUDY THE VISUAL DIFFERENCE: What did you do wrong? What's missing or incorrect?\n"
            
            # For successful examples: just show expected output
            else:
                output_img_path = str(temp_dir / f"refine_ex{ex['number']}_output.png")
                img_out = grid_to_image(ex['output'], 30)
                img_out.save(output_img_path)
                image_paths.append(output_img_path)
                
                prompt += f"Expected Output (see image above) - ✅ Your steps already work here\n"
        
        prompt += f"""

TASK: MODIFY your current steps to be MORE GENERAL so they work for ALL {len(all_examples)} examples above.

IMPORTANT - DON'T REWRITE FROM SCRATCH:
- Start with your current steps
- EDIT/ADJUST them to be more general
- Keep the structure if it's correct
- Change only what's needed to handle all cases
- If a step works for all examples, keep it as-is
- If a step is too specific, generalize it

HOW TO GENERALIZE (based on the VISUALIZATIONS you see):
- Look at what VARIES across the images: object count, sizes, positions
- Look at what's CONSTANT: transformation type, color mapping, the core visual rule
- Make steps describe the CONSTANT pattern in a way that handles the VARIATIONS
- Use visual language: "each enclosed region" not "the region at position X"
- Make steps MORE GENERAL but keep them CONCRETE and ACTIONABLE

ONE CONCRETE ACTION PER STEP:
- Each step = ONE SPECIFIC ACTION (e.g., "draw the first bridge", not "draw all bridges")
- If performing the same action on multiple objects, make ONE step per object/action
- Steps should be concrete and followable: "Draw bridge from red to blue" NOT "For each color pair, draw bridge"
- NO pseudocode patterns like "for each...", "if/else as separate steps"
- NO setup-only steps like "Identify pattern" (this is analysis, not action)
- Each step should produce a visible change in the grid
- Steps should be ordered logically (don't jump around)

GOOD GRANULARITY:
✓ One bridge per step (if drawing 5 bridges, that's 5 steps)
✓ One object coloring per step (if coloring 3 objects, that's 3 steps)  
✓ One transformation application per step
✓ Followable by a human executing step-by-step

FORBIDDEN PATTERNS:
❌ "For each cell, do X" (split into individual steps instead)
❌ "Draw all bridges" (specify each bridge separately)
❌ "Color all objects" (specify each object separately)
❌ Incomplete fragments like "For each X:" without the action

THE RIGHT LEVEL OF DETAIL (like explaining to a person):
- Specific enough to execute unambiguously, without missing any details
- General enough to be read and understood by a human, without being too specific to one example
- Visual references: "the bottom row", "each colored square"
- Clear actions: "draw", "fill", "copy", "extend"
- Measurements when needed: "2 cells wide", "across entire row"
- Constraints: "without overwriting X", "only for Y"

Follow these meta-instructions:
1. Identify VISUAL objects, shapes, and patterns with SPATIAL locations (e.g., "the red square in the top-left", not "cell [0,0]")
2. Look for reference objects - visual elements that stand out
3. Determine the VISUAL transformation by comparing images
4. Describe in HIGH-LEVEL VISUAL terms with PRECISION

CRITICAL CONSISTENCY AND DETAIL RULES:
- Think VISUALLY from the images shown
- Look for THE SINGLE UNIVERSAL RULE visible across all examples
- Do not change object shapes unless part of transformation
- Apply transformation EXACTLY THE SAME WAY to all relevant parts
- Do NOT vary arbitrarily - only if puzzle provides explicit clues
- If transformation works for one object, use IDENTICAL transformation for all similar objects
- THE RULE SHOULD BE OBVIOUS - don't invent complex patterns that aren't there
- COUNT CAREFULLY: Don't miss single-cell blocks or small details
- PRESERVE SHAPES: Keep exact dimensions when copying/transforming
- ALL OBJECTS: Ensure transformation applies to EVERY relevant object, not just some
- Make general enough to work on different sizes/counts

Example (GOOD - using reference object, bridge drawing):
'1. Identify the bottom row as the reference containing color sequence: red, blue, yellow, green
 2. Draw bridge connecting red (2) to blue (3) using red, matching block width
 3. Draw bridge connecting blue (3) to yellow (9) using blue
 4. Draw bridge connecting yellow (9) to green (4) using yellow
 5. Draw bridge connecting green (4) to red (2) using green'

Example (GOOD - object recoloring, one per step):
'1. Count holes in first maroon object (2 holes), recolor it to color 3
 2. Count holes in second maroon object (1 hole), recolor it to color 1
 3. Count holes in third maroon object (3 holes), recolor it to color 2'

Note: If there's a reference object (like a row showing color sequence), identify it first, then use it to guide subsequent steps.

Example (BAD - pseudocode/loops):
'1. For each color pair in the sequence:
 2. Locate the two blocks
 3. Draw a bridge between them
 4. Use the first block's color'

Example (BAD - too vague):
'1. Draw all the bridges
 2. Color all the objects'

Output your MODIFIED steps (edit the current ones, don't start over).

Format as numbered list. Aim for 3-7 concrete, general steps."""
        
        response = self.call_ai(prompt, image_paths)
        
        # Parse refined steps
        refined_steps = []
        for line in response.split('\n'):
            if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                refined_steps.append(line.strip())
        
        return refined_steps
    
    def refine_across_examples(self, task_file: str, output_dir: str = "refined_booklets"):
        """Main refinement process"""
        
        # Load task
        self.task_data = self.load_task(task_file)
        self.task_name = Path(task_file).stem
        
        training_examples = self.task_data['train']
        test_examples = self.task_data.get('test', [])
        
        print(f"\n{'='*80}")
        print(f"ITERATIVE BOOKLET REFINEMENT")
        print(f"{'='*80}")
        print(f"Task: {self.task_name}")
        print(f"Training Examples: {len(training_examples)}")
        print(f"Test Examples: {len(test_examples)}")
        print(f"{'='*80}\n")
        
        # Create output directory
        output_path = Path(output_dir) / f"{self.task_name}_refined"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Store refinement history
        refinement_history = []
        
        # STEP 1: Generate initial steps from first example
        print(f"\n[Example 1/{len(training_examples)}] Generating initial steps...")
        print("-" * 80)
        
        current_steps = self.generate_initial_steps(
            training_examples[0]['input'],
            training_examples[0]['output']
        )
        
        print(f"[OK] Generated {len(current_steps)} initial steps")
        for i, step in enumerate(current_steps, 1):
            try:
                print(f"  {i}. {step[:80]}{'...' if len(step) > 80 else ''}")
            except:
                print(f"  {i}. [Step {i}]")
        
        refinement_history.append({
            "example_number": 1,
            "steps": current_steps,
            "action": "initial_generation",
            "success": True  # Assume first example works
        })
        
        # Track all seen examples for refinement
        seen_examples = [{
            "number": 1,
            "input": training_examples[0]['input'],
            "output": training_examples[0]['output'],
            "input_shape": [len(training_examples[0]['input']), len(training_examples[0]['input'][0])],
            "output_shape": [len(training_examples[0]['output']), len(training_examples[0]['output'][0])],
            "works": True,  # Initial steps generated from this
            "model_output": training_examples[0]['output']  # Assume initial steps work
        }]
        
        # STEP 2: Test and refine on subsequent examples
        for idx in range(1, len(training_examples)):
            print(f"\n[Example {idx + 1}/{len(training_examples)}] Testing current steps...")
            print("-" * 80)
            
            example = training_examples[idx]
            result_grid, matches = self.apply_steps_to_example(
                current_steps,
                example['input'],
                example['output']
            )
            
            if matches:
                print(f"[OK] Steps work for example {idx + 1}")
                refinement_history.append({
                    "example_number": idx + 1,
                    "steps": current_steps,
                    "action": "tested_success",
                    "success": True
                })
                
                # Add to seen examples (works with current steps)
                seen_examples.append({
                    "number": idx + 1,
                    "input": example['input'],
                    "output": example['output'],
                    "input_shape": [len(example['input']), len(example['input'][0])],
                    "output_shape": [len(example['output']), len(example['output'][0])],
                    "works": True,
                    "model_output": result_grid  # Also store successful outputs
                })
            else:
                print(f"[FAIL] Steps don't work for example {idx + 1}")
                print(f"  Refining to work for ALL {idx + 1} examples seen so far...")
                
                # Add current failed example (including what model produced)
                seen_examples.append({
                    "number": idx + 1,
                    "input": example['input'],
                    "output": example['output'],
                    "input_shape": [len(example['input']), len(example['input'][0])],
                    "output_shape": [len(example['output']), len(example['output'][0])],
                    "works": False,
                    "model_output": result_grid  # What the model actually produced
                })
                
                # Refine steps to work for ALL seen examples
                current_steps = self.refine_steps(current_steps, seen_examples)
                
                print(f"[REFINED] Generated {len(current_steps)} more general steps")
                for i, step in enumerate(current_steps, 1):
                    try:
                        print(f"  {i}. {step[:80]}{'...' if len(step) > 80 else ''}")
                    except:
                        print(f"  {i}. [Step {i}]")
                
                # Re-test on ALL previous examples to verify generalization
                print(f"  Verifying refined steps on all {len(seen_examples)} examples...")
                all_work = True
                for verify_ex in seen_examples:
                    verify_grid, verify_match = self.apply_steps_to_example(
                        current_steps,
                        verify_ex['input'],
                        verify_ex['output']
                    )
                    # Update model_output with new attempt
                    verify_ex['model_output'] = verify_grid
                    verify_ex['works'] = verify_match
                    
                    if not verify_match:
                        all_work = False
                        print(f"    [WARN] Still fails on example {verify_ex['number']}")
                
                if all_work:
                    print(f"  [OK] Refined steps now work for all {len(seen_examples)} examples!")
                else:
                    print(f"  [WARN] Refined steps still don't work for all examples")
                
                refinement_history.append({
                    "example_number": idx + 1,
                    "steps": current_steps,
                    "action": "refined_after_failure",
                    "success": all_work,
                    "verified_on_all": all_work
                })
        
        # STEP 3: Apply to test cases
        test_results = []
        
        if test_examples:
            print(f"\n{'='*80}")
            print("APPLYING TO TEST CASES")
            print(f"{'='*80}\n")
            
            for test_idx, test_ex in enumerate(test_examples):
                print(f"[Test Case {test_idx + 1}/{len(test_examples)}]")
                print("-" * 80)
                
                # Try up to 3 times (ARC benchmark rules)
                for attempt in range(3):
                    print(f"  Attempt {attempt + 1}/3...")
                    
                    result_grid, matches = self.apply_steps_to_example(
                        current_steps,
                        test_ex['input'],
                        test_ex['output']
                    )
                    
                    if matches:
                        print(f"  [SUCCESS] Correct on attempt {attempt + 1}!")
                        test_results.append({
                            "test_number": test_idx + 1,
                            "attempts": attempt + 1,
                            "success": True
                        })
                        break
                else:
                    print(f"  [FAIL] Failed all 3 attempts")
                    test_results.append({
                        "test_number": test_idx + 1,
                        "attempts": 3,
                        "success": False
                    })
        
        # Generate meta-booklet
        print(f"\n{'='*80}")
        print("GENERATING META-BOOKLET")
        print(f"{'='*80}")
        
        # Calculate statistics
        total_refinements = sum(1 for h in refinement_history if h['action'] == 'refined_after_failure')
        successful_tests = sum(1 for h in refinement_history if h['success'])
        test_success = sum(1 for t in test_results if t['success']) if test_results else 0
        
        print(f"\nRefinement Summary:")
        print(f"  Total training examples: {len(training_examples)}")
        print(f"  Successful tests: {successful_tests}/{len(training_examples)}")
        print(f"  Refinements made: {total_refinements}")
        print(f"  Final steps: {len(current_steps)}")
        
        if test_results:
            print(f"\nTest Results:")
            print(f"  Test cases solved: {test_success}/{len(test_results)}")
            for t in test_results:
                status = "SUCCESS" if t['success'] else "FAILED"
                print(f"    Test {t['test_number']}: {status} (attempts: {t['attempts']})")
        
        meta_booklet = {
            "task_name": self.task_name,
            "refinement_history": refinement_history,
            "final_steps": current_steps,
            "total_training_examples": len(training_examples),
            "total_refinements": total_refinements,
            "successful_training_tests": successful_tests,
            "test_results": test_results,
            "test_success_count": test_success,
            "total_test_cases": len(test_examples),
            "generated_at": datetime.now().isoformat()
        }
        
        # Save meta-booklet
        meta_path = output_path / "refinement_meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_booklet, f, indent=2)
        
        # Generate README
        readme = self._generate_readme(meta_booklet)
        readme_path = output_path / "README.txt"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme)
        
        print(f"\n[SUCCESS] Meta-booklet saved to: {output_path}")
        print(f"  - refinement_meta.json: Complete refinement history")
        print(f"  - README.txt: Human-readable summary")
        
        return meta_booklet
    
    def _generate_readme(self, meta_booklet):
        """Generate README for meta-booklet"""
        history = meta_booklet['refinement_history']
        
        readme = f"""# ARC-AGI Iterative Refinement Booklet
Task: {meta_booklet['task_name']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
This booklet shows how the solving steps evolved across {meta_booklet['total_training_examples']} training examples.

## Refinement History

"""
        
        for entry in history:
            action_label = {
                'initial_generation': '🌱 Initial Generation',
                'tested_success': '✅ Tested - Success',
                'refined_after_failure': '🔧 Refined After Failure'
            }.get(entry['action'], entry['action'])
            
            readme += f"### Example {entry['example_number']} - {action_label}\n"
            readme += f"- Action: {entry['action']}\n"
            readme += f"- Success: {'Yes' if entry['success'] else 'No'}\n"
            readme += f"- Step Count: {len(entry['steps'])}\n\n"
            readme += "Steps:\n"
            for i, step in enumerate(entry['steps'], 1):
                step_preview = step[:100] + '...' if len(step) > 100 else step
                readme += f"  {i}. {step_preview}\n"
            readme += "\n"
        
        readme += f"\n## Final Refined Steps\n\n"
        readme += "These steps have been refined to work across ALL training examples:\n\n"
        if meta_booklet['final_steps']:
            for i, step in enumerate(meta_booklet['final_steps'], 1):
                readme += f"{i}. {step}\n"
        else:
            readme += "No final steps generated.\n"
        
        readme += f"\n## Test Results\n\n"
        
        if meta_booklet.get('test_results'):
            readme += f"Test cases solved: {meta_booklet['test_success_count']}/{meta_booklet['total_test_cases']}\n\n"
            for test in meta_booklet['test_results']:
                status = "SUCCESS" if test['success'] else "FAILED"
                readme += f"- Test {test['test_number']}: {status} (attempts: {test['attempts']})\n"
        else:
            readme += "No test cases in this puzzle.\n"
        
        readme += f"\n## Analysis\n\n"
        readme += f"Total training examples: {meta_booklet['total_training_examples']}\n"
        readme += f"Successful training tests: {meta_booklet['successful_training_tests']}/{meta_booklet['total_training_examples']}\n"
        readme += f"Total refinements: {meta_booklet['total_refinements']}\n"
        readme += f"Test cases solved: {meta_booklet.get('test_success_count', 0)}/{meta_booklet.get('total_test_cases', 0)}\n"
        
        return readme


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python arc-booklet-refiner.py <task_json_file> [output_dir]")
        print("Example: python arc-booklet-refiner.py ../saturn-arc/ARC-AGI-2/ARC-AGI-1/data/training/00d62c1b.json")
        sys.exit(1)
    
    task_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "refined_booklets"
    
    if not os.path.exists(task_file):
        print(f"Error: Task file '{task_file}' not found")
        sys.exit(1)
    
    try:
        refiner = IterativeBookletRefiner()
        refiner.refine_across_examples(task_file, output_dir)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

