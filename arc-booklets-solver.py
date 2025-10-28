#!/usr/bin/env python3
"""
ARC-AGI Booklet Solver
Generates step-by-step instructions to solve ARC puzzles following meta-instructions
"""

import json
import os
import sys
import base64
import re
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
from PIL import Image
import numpy as np
from datetime import datetime

# Import the visualizer functions
from arc_visualizer import grid_to_image, ARC_COLORS

# Define the visualization tool for function calling
VISUALIZATION_TOOL = {
    "type": "function",
    "function": {
        "name": "visualize_grid",
        "description": "Generate a visual image representation of a grid. Use this to better understand patterns and transformations in the puzzle.",
        "parameters": {
            "type": "object",
            "properties": {
                "grid": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"}
                    },
                    "description": "2D array of integers (0-9) representing the grid to visualize"
                }
            },
            "required": ["grid"]
        }
    }
}


class Step:
    """Represents a single step in the booklet"""
    def __init__(self, step_number: int, instruction: str, grid_data: Optional[List[List[int]]] = None):
        self.step_number = step_number
        self.instruction = instruction
        self.grid_data = grid_data
        self.image_path = None
        self.timestamp = datetime.now()


class Booklet:
    """Represents a complete booklet with all steps"""
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.steps: List[Step] = []
        self.final_prediction: Optional[List[List[int]]] = None
        self.actual_output: Optional[List[List[int]]] = None
        self.accuracy = None
        
    def add_step(self, instruction: str, grid_data: Optional[List[List[int]]] = None, solver=None):
        step = Step(len(self.steps) + 1, instruction, grid_data)
        
        # Generate visualization if grid data is provided and solver is available
        if grid_data is not None and solver is not None:
            try:
                step.image_path = solver.create_grid_image(grid_data, label=f"step_{step.step_number:03d}")
            except Exception as e:
                print(f"Warning: Could not create visualization for step {step.step_number}: {e}")
        
        self.steps.append(step)
        return step
    
    def to_dict(self) -> Dict:
        """Convert booklet to dictionary for JSON serialization"""
        return {
            "task_name": self.task_name,
            "steps": [
                {
                    "step_number": step.step_number,
                    "instruction": step.instruction,
                    "has_grid": step.grid_data is not None,
                    "grid_shape": list(np.array(step.grid_data).shape) if step.grid_data else None,
                    "image_path": step.image_path,
                    "has_image": step.image_path is not None
                }
                for step in self.steps
            ],
            "final_prediction_shape": list(np.array(self.final_prediction).shape) if self.final_prediction else None,
            "actual_output_shape": list(np.array(self.actual_output).shape) if self.actual_output else None,
            "accuracy": self.accuracy
        }


class ARCBookletSolver:
    def __init__(self):
        """Initialize the ARC booklet solver with API credentials"""
        self.api_key_openai = os.getenv("OPENAI_API_KEY")
        if not self.api_key_openai:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        self.client_openai = OpenAI(api_key=self.api_key_openai)
        
        self.current_task_name = None
        self.booklet = None
        
        # Use img_tmp directory in project root
        self.temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img_tmp")
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def load_task(self, file_path: str) -> Dict[str, Any]:
        """Load an ARC-AGI task from a JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def format_grid(self, grid: List[List[int]]) -> str:
        """Format a grid for display in the prompt"""
        return '\n'.join(['[' + ', '.join(str(cell) for cell in row) + ']' for row in grid])
    
    def create_grid_image(self, grid: List[List[int]], cell_size: int = 30, label: str = "grid") -> str:
        """Create an image from a grid and return the file path"""
        img = grid_to_image(grid, cell_size)
        # Save to temp file with meaningful name including task name
        file_count = len([f for f in os.listdir(self.temp_dir) if f.endswith('.png')])
        if self.current_task_name:
            temp_path = os.path.join(self.temp_dir, f"{self.current_task_name}_{label}_{file_count:03d}.png")
        else:
            temp_path = os.path.join(self.temp_dir, f"{label}_{file_count:03d}.png")
        img.save(temp_path)
        return temp_path
    
    def encode_image(self, image_path: str) -> str:
        """Encode an image file to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def call_ai_with_image(self, text_prompt: str, image_paths: List[str]) -> str:
        """Call OpenAI with text and images"""
        
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
            response = self.client_openai.chat.completions.create(**call_params)
            
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
    
    def parse_grid_from_response(self, response: str) -> Optional[List[List[int]]]:
        """Parse a grid from the AI's response with multiple fallback strategies"""
        import re
        
        # Strategy 1: Look for "FINAL GRID:" section and parse what follows
        final_grid_match = re.search(r'FINAL GRID:\s*\n([\[\d,\s\n\]]+)', response, re.IGNORECASE)
        if final_grid_match:
            grid_text = final_grid_match.group(1)
            grid = self._parse_grid_text(grid_text)
            if grid and len(grid) > 0:
                return grid
        
        # Strategy 2: Find all bracket-enclosed sequences (original method)
        pattern = r'\[[\d,\s]+\]'
        matches = re.findall(pattern, response)
        
        grid = []
        for match in matches:
            # Remove brackets and parse numbers
            numbers = match.strip('[]').replace(' ', '').split(',')
            row = [int(n) for n in numbers if n and n.strip().isdigit()]
            if row:
                grid.append(row)
        
        if grid and len(grid) > 0:
            return grid
        
        # Strategy 3: Look for any lines that look like grid rows (numbers separated by commas or spaces)
        lines = response.split('\n')
        grid = []
        for line in lines:
            # Try to find lines with multiple numbers
            numbers = re.findall(r'\d+', line)
            if len(numbers) >= 3:  # At least 3 numbers to be considered a row
                try:
                    row = [int(n) for n in numbers]
                    # Basic validation: numbers should be in reasonable range (0-9 for ARC)
                    if all(0 <= n <= 9 for n in row):
                        grid.append(row)
                except ValueError:
                    continue
        
        # Validate the grid has consistent row lengths
        if grid and len(grid) > 1:
            row_lengths = [len(row) for row in grid]
            if len(set(row_lengths)) == 1:  # All rows same length
                return grid
        
        return None
    
    def _parse_grid_text(self, text: str) -> Optional[List[List[int]]]:
        """Helper to parse a block of text containing a grid"""
        import re
        pattern = r'\[[\d,\s]+\]'
        matches = re.findall(pattern, text)
        
        grid = []
        for match in matches:
            numbers = match.strip('[]').replace(' ', '').split(',')
            row = [int(n) for n in numbers if n and n.strip().isdigit()]
            if row:
                grid.append(row)
        
        return grid if grid else None
    
    def extract_intermediate_grids(self, response: str) -> List[List[List[int]]]:
        """Extract multiple intermediate grids from a detailed response"""
        import re
        
        # Find all bracket-enclosed sequences that look like grids
        pattern = r'\[[\d,\s\[\]]+\]'
        matches = re.findall(pattern, response)
        
        grids = []
        for match in matches:
            try:
                # Try to parse as a nested list structure
                if '[' in match and ']' in match:
                    # This looks like a multi-line grid representation
                    lines = match.strip('[]').split('], [')
                    grid = []
                    for line in lines:
                        line = line.strip('[]')
                        numbers = line.replace(' ', '').split(',')
                        row = [int(n) for n in numbers if n]
                        if row:
                            grid.append(row)
                    if grid and len(grid) > 1:  # Valid grid should have multiple rows
                        grids.append(grid)
            except (ValueError, IndexError):
                continue
        
        return grids
    
    def solve(self, task_file: str) -> Booklet:
        """
        Main solving loop that generates a booklet with iterative instruction-building
        Returns a Booklet object
        """
        # Extract task name from file path
        self.current_task_name = os.path.splitext(os.path.basename(task_file))[0]
        self.booklet = Booklet(self.current_task_name)
        
        # Load the task
        task = self.load_task(task_file)
        print(f"\nLoaded task: {task_file}")
        print(f"Task contains {len(task['train'])} training examples and {len(task['test'])} test examples")
        
        # Initialize with empty instructions
        current_instructions = ""
        
        # Iterative instruction building: process each training example one by one
        for i, example in enumerate(task['train']):
            self.booklet.add_step(f"=== ITERATION {i+1}: Processing Training Example {i+1} ===", None, self)
            
            # Create images for this example
            input_img = self.create_grid_image(example['input'], label=f"train{i+1}_input")
            output_img = self.create_grid_image(example['output'], label=f"train{i+1}_output")
            
            # Show the example
            self.booklet.add_step(f"Training Example {i+1}:", example['input'], self)
            
            # Only show expected output for the first example
            if i == 0:
                self.booklet.add_step(f"Expected Output:", example['output'], self)
            
            if i == 0:
                # First example: First analyze the input visually, then generate instructions
                
                # Step 1: Analyze input grid visually
                analyze_prompt = f"""You are solving an ARC-AGI puzzle. This is the FIRST training example INPUT.

IMPORTANT: Work from the VISUAL representation of the puzzle (the image), not the grid numbers. Think about the puzzle as a visual pattern, not as coordinates or cell values.

Input Grid:
{self.format_grid(example['input'])}

Analyze this input IMAGE carefully using VISUAL, HIGH-LEVEL descriptions WITH PRECISION:
- What shapes, objects, or patterns do you SEE in the image?
- What colors are present and how are they arranged visually?
- Are there any visual elements that stand out as different or special (reference objects)?
- Describe the overall visual structure or layout in spatial terms (top, bottom, left, right, center, corners, etc.)
- What visual/spatial relationships exist between objects? (adjacent, surrounding, inside, above, below, etc.)
- Be PRECISE about measurements: exact sizes, counts, distances, dimensions (e.g., "3 red squares", "5 cells wide", "2 cells apart")

Think visually at a HIGH LEVEL (patterns and objects, not cell coordinates), but be MATHEMATICALLY PRECISE about quantities, sizes, and distances. Use the visualization tool to better understand the visual structure."""
                
                analyze_response = self.call_ai_with_image(analyze_prompt, [input_img])
                self.booklet.add_step("Visual Analysis of Input:", None, self)
                self.booklet.add_step(analyze_response, None, self)
                
                # Step 2: Now show output and generate instructions
                instruction_prompt = f"""You are solving an ARC-AGI puzzle. You've analyzed the input. Now see the expected output and generate transformation instructions.

CRITICAL: Work from the VISUAL/IMAGE representation. Think spatially and visually, NOT in terms of grid coordinates or cell numbers. Describe transformations like you're explaining a visual pattern to someone looking at pictures.

KEY PRINCIPLE: There is ONE clear, deterministic transformation rule that explains how to get from input to output. Your goal is to identify this SINGLE transformation rule from this first example. This same rule will apply consistently to ALL training and test examples.

Your Input Analysis:
{analyze_response}

Input:
{self.format_grid(example['input'])}

Expected Output:
{self.format_grid(example['output'])}

Follow these meta-instructions to generate detailed HIGH-LEVEL, VISUAL step-by-step instructions:

1. This is a visual puzzle requiring high-level reasoning. Identify the VISUAL objects, shapes, and patterns and describe their SPATIAL locations (e.g., "the red square in the top-left", not "cell [0,0]").
2. Look for a reference object - a visual element that stands out or is different. It provides visual clues for the transformation (colors to use, spatial arrangements, etc.).
3. Determine the VISUAL transformation by comparing input and output IMAGES. The reference might indicate: color mappings, spatial operations (rotation, reflection, movement), or which visual elements to modify.
4. Describe transformations in HIGH-LEVEL VISUAL terms: "move the blue shape to the right", "fill the interior with red", "copy the pattern from the reference", NOT "set cell [x,y] to value 5".

Generate detailed HIGH-LEVEL step-by-step instructions that explain the VISUAL transformation WITH PRECISION. For each step, be explicit about:
- What VISUAL objects/patterns you're identifying and WHERE they are spatially
- What VISUAL transformation you're applying (in terms of shapes, colors, movements, spatial operations)
- What the VISUAL result looks like
- Be PRECISE with all measurements: "move exactly 3 cells right", "create a 5x5 grid", "repeat 4 times", "fill with 2-cell-wide border"

Think HIGH-LEVEL (describe patterns and transformations abstractly, not individual cell operations), but be MATHEMATICALLY EXACT about all quantities, sizes, positions, and counts.

REMEMBER: You are identifying THE transformation rule - a single, consistent rule that will work for ALL examples in this puzzle. Focus on finding the universal pattern, not describing this one example.

CRITICAL CONSISTENCY RULES:
- Do not change the shape of objects in the puzzle unless it's a part of the transformation (if you are moving, rotating or scaling an object don't change the shape)
- Once you identify a transformation, apply it EXACTLY THE SAME WAY to all relevant parts of the puzzle
- Do NOT vary or modify the transformation arbitrarily for different objects or regions
- ONLY change how you apply a transformation if there is a clear context clue in the puzzle that indicates a different approach (e.g., a reference object that specifies different behavior, positional rules, etc.)
- If a transformation works for one object, use the IDENTICAL transformation for all similar objects unless the puzzle explicitly indicates otherwise


Special note about grid sizes: YOU MUST INFER THE OUTPUT GRID SIZES FROM THE INPUT GRID SIZES! Remember the first training example's output size and what determined it.:
FOR EACH TRAINING EXAMPLE: (dont change grid size except when looking at the new training example)
Output sizes can be determined by something in the puzzle (e.g. size of the output object with no extra tiles, size of the input without the reference object, etc.)
Output sizes can also be determined by the ratio of input grid size to output grid size. 
Determine the output grid size pattern based on the input output pair of the first training sample.
Final output size must be homogenous (a rectangle, no matter what), and the pattern you use to infer input size to output size must be consistent over all the examples.
Break down the transformation into incremental steps that can be visualized. Try to generate a visualization for every step."""
                
                response = self.call_ai_with_image(instruction_prompt, [input_img, output_img])
                current_instructions = response
                self.booklet.add_step("Generated Step-by-Step Instructions:", None, self)
                self.booklet.add_step(current_instructions, None, self)
                
            else:
                # Subsequent examples: first predict without seeing output, then refine based on feedback
                predict_prompt = f"""You have step-by-step instructions from the first {i} training examples. Now apply them to training example {i+1} to predict the output.

IMPORTANT: Work VISUALLY from the IMAGE. Think in terms of shapes, colors, and spatial relationships, NOT grid coordinates or cell values. Describe what you see and what you're doing in HIGH-LEVEL VISUAL terms.

EXISTING STEP-BY-STEP INSTRUCTIONS:
{current_instructions}

NEW TRAINING EXAMPLE {i+1} INPUT:
{self.format_grid(example['input'])}

Apply your current instructions to this new input IMAGE and predict what the output should look like VISUALLY. Show your reasoning step-by-step using HIGH-LEVEL spatial and visual descriptions WITH PRECISE measurements.

CRITICAL: 
- Apply transformations CONSISTENTLY. Once you identify how to transform something visually, use the EXACT SAME visual transformation for all similar objects.
- Be PRECISE about all measurements: exact distances, sizes, counts, repetitions
- Think HIGH-LEVEL (patterns and objects) but be MATHEMATICALLY EXACT (quantities and dimensions)
- Do NOT arbitrarily change the transformation unless there's a clear visual/spatial reason in the puzzle.

At the end, output your prediction in this format:

FINAL GRID:
[row1]
[row2]
[row3]
...

CRITICAL: You MUST output a final grid. Even if you're uncertain about some details, make your best guess and output the complete grid. NEVER refuse to generate output or ask for clarification. Always commit to a prediction."""
                
                predict_response = self.call_ai_with_image(predict_prompt, [input_img])
                self.booklet.add_step(f"Applying Instructions to Example {i+1} (without seeing output):", None, self)
                self.booklet.add_step(predict_response, None, self)
                
                # Parse the predicted output
                predicted_for_training = self.parse_grid_from_response(predict_response)
                if predicted_for_training:
                    self.booklet.add_step(f"Predicted Output for Example {i+1}:", predicted_for_training, self)
                
                # Now show the actual output and ask to refine
                if predicted_for_training == example['output']:
                    feedback = "✅ Your prediction was CORRECT!"
                    self.booklet.add_step(feedback, None, self)
                else:
                    feedback = f"❌ Your prediction was INCORRECT. Here is the actual output:"
                    self.booklet.add_step(feedback, None, self)
                    self.booklet.add_step("Actual Output:", example['output'], self)
                
                # Now refine instructions based on feedback
                refine_prompt = f"""Based on the feedback, refine your step-by-step instructions.

CRITICAL: Keep instructions HIGH-LEVEL and VISUAL. Work from the IMAGE representation using spatial and visual descriptions, NOT grid coordinates or cell numbers.

KEY PRINCIPLE: There is ONE transformation rule for this entire puzzle. If your prediction was incorrect, it means your understanding of THE RULE is incomplete or wrong - not that there are different rules for different examples. Find THE SINGLE RULE that works for ALL examples.

EXISTING STEP-BY-STEP INSTRUCTIONS:
{current_instructions}

TRAINING EXAMPLE {i+1}:
Input:
{self.format_grid(example['input'])}

Actual Output:
{self.format_grid(example['output'])}

Your Prediction: {"CORRECT" if predicted_for_training == example['output'] else "INCORRECT"}

Analyze this example VISUALLY and refine your understanding of THE RULE so it works for ALL {i+1} training examples seen so far. If this example contradicts your existing instructions, it means you haven't correctly identified THE transformation rule yet. Revise your understanding to find THE SINGLE RULE that explains all examples. If a certain step is consistent with all examples, mark it with "IMPORTANT: *step*"

Provide updated detailed HIGH-LEVEL, VISUAL step-by-step instructions WITH PRECISION that work for all examples. For each step, be explicit about:
- What VISUAL objects/patterns you're identifying and their SPATIAL locations
- What VISUAL/SPATIAL transformation you're applying (described in terms of shapes, colors, movements)
- What the VISUAL result looks like
- Be PRECISE: exact counts, sizes, distances, dimensions (e.g., "move 2 cells down", "create 3x3 square", "repeat for all 5 objects")

Think HIGH-LEVEL about patterns and transformations, but be MATHEMATICALLY EXACT about all measurements and quantities.

REMEMBER: You are refining your understanding of THE SINGLE transformation rule. All training examples follow the same rule - if they seem different, you need to find the more general pattern that unifies them.

IMPORTANT: Keep your instructions CONCISE and CLEAR. Aim for under 500 words. Focus on the core transformation, not every tiny edge case. If your previous instructions were too complex, SIMPLIFY them - find the simpler pattern that unifies all examples.

CRITICAL CONSISTENCY RULES:
- Once you identify a transformation, apply it EXACTLY THE SAME WAY throughout the entire puzzle
- Do NOT vary the transformation for different objects unless there is a clear context-based reason from the puzzle itself
- If a transformation works, repeat it verbatim without arbitrary modifications
- ONLY change the transformation if the puzzle provides explicit clues (reference objects, positional patterns, color coding, etc.) that indicate different behavior

Special note about grid sizes: YOU MUST INFER THE OUTPUT GRID SIZES FROM THE INPUT GRID SIZES! Remember the first training example's output size and what determined it.
Output sizes can be determined by something in the puzzle (e.g. size of the output object with no extra tiles, size of the input without the reference object, etc.)
Output sizes can also be determined by the ratio of input grid size to output grid size. 
Determine the output grid size pattern based on the input output pair of the first training sample.
Final output size must be homogenous (a rectangle, no matter what), and the pattern you use to infer input size to output size must be consistent over all the examples.

Break down the transformation into incremental steps that can be visualized. Visualize any step that you."""
                
                response = self.call_ai_with_image(refine_prompt, [input_img, output_img])
                current_instructions = response
                self.booklet.add_step(f"Updated Step-by-Step Instructions (works for all {i+1} examples):", None, self)
                self.booklet.add_step(current_instructions, None, self)
                
                # Simplification phase - only if instructions are getting too long
                word_count = len(current_instructions.split())
                if word_count > 800:  # Instructions are getting too complex
                    simplify_prompt = f"""Your current instructions are too complex ({word_count} words). Simplify them to capture THE core transformation rule in under 500 words.

CURRENT INSTRUCTIONS:
{current_instructions}

TASK: Distill these instructions down to their ESSENCE. Remove unnecessary details, edge cases, and repetition. Focus on THE SINGLE RULE that works for all {i+1} examples. Be concise and clear.

Output the SIMPLIFIED instructions:"""
                    
                    simplified = self.call_ai_with_image(simplify_prompt, [input_img, output_img])
                    current_instructions = simplified
                    self.booklet.add_step(f"Simplified Instructions ({len(simplified.split())} words):", None, self)
                    self.booklet.add_step(current_instructions, None, self)
        
        # Final step: apply instructions to test input with detailed breakdown
        self.booklet.add_step("=== FINAL TEST: Applying Step-by-Step Instructions to Test Input ===", None, self)
        
        # Add test input to booklet with visualization
        self.booklet.add_step("Test Input:", task['test'][0]['input'], self)
        
        test_input_img = self.create_grid_image(task['test'][0]['input'], label="test_input")
        
        final_prompt = f"""Apply your final refined step-by-step instructions to the test input.

CRITICAL: Work VISUALLY from the IMAGE. Think in HIGH-LEVEL spatial and visual terms (shapes, colors, positions, patterns), NOT in grid coordinates or cell numbers. Describe what you see and do using visual language.

KEY PRINCIPLE: You have identified THE transformation rule from the training examples. This is THE SAME RULE that applies to the test input. The test may look different superficially (different colors, sizes, arrangements), but THE UNDERLYING TRANSFORMATION RULE IS IDENTICAL.

The test input IMAGE will be a little different from the examples you've seen before. There may be differences in colors, shapes, or visual layout, but THE CORE TRANSFORMATION RULE remains the same.
Look at the test input IMAGE, identify how THE SAME RULE applies to this new instance.
TEST INPUT:
{self.format_grid(task['test'][0]['input'])}

Here are THE transformation rule instructions learned from training examples. This is THE SAME RULE - apply it consistently to this new test input.
FINAL STEP-BY-STEP INSTRUCTIONS:
{current_instructions}

CRITICAL: 
- You are applying THE SAME transformation rule you learned from training examples
- The rule doesn't change for the test - apply it CONSISTENTLY and IDENTICALLY
- Be PRECISE with all measurements: exact distances, sizes, counts, repetitions
- Think HIGH-LEVEL (patterns, not cell-by-cell operations) but be MATHEMATICALLY EXACT (specific numbers for all quantities)
- Do NOT modify or adapt the rule arbitrarily - it's THE SAME RULE, just applied to a new instance
- If something seems different, it's because you're seeing THE SAME RULE in a new context, not a different rule

Apply your instructions step-by-step using HIGH-LEVEL VISUAL reasoning WITH PRECISE measurements. At the end, output your final prediction in this format:

FINAL GRID:
[row1]
[row2]
[row3]
...

CRITICAL: You MUST output a final grid. Even if you're uncertain about exact pixel placement or edge cases, make your best guess and output the complete grid. NEVER refuse to generate output or ask for clarification. Always commit to a prediction based on your best understanding of THE RULE."""
        
        final_response = self.call_ai_with_image(final_prompt, [test_input_img])
        
        # Parse the response to extract intermediate steps
        self.booklet.add_step("=== DETAILED STEP-BY-STEP APPLICATION FOR TEST INPUT ===", None, self)
        self.booklet.add_step("Step-by-step breakdown:", None, self)
        self.booklet.add_step(final_response, None, self)
        
        # Try to extract intermediate grids from the response
        intermediate_grids = self.extract_intermediate_grids(final_response)
        
        # Add each intermediate step as a separate booklet step
        for j, grid in enumerate(intermediate_grids):
            self.booklet.add_step(f"Test Intermediate Step {j+1}:", grid, self)
        
        predicted_output = self.parse_grid_from_response(final_response)
        self.booklet.final_prediction = predicted_output
        
        self.booklet.add_step("Final Test Prediction:", predicted_output, self)
        
        # Check accuracy if test output exists
        if 'output' in task['test'][0] and task['test'][0]['output']:
            self.booklet.actual_output = task['test'][0]['output']
            
            # Calculate cell-by-cell accuracy
            if predicted_output is None:
                self.booklet.accuracy = 0.0
                print("\n❌ No prediction generated")
                self.booklet.add_step("❌ FINAL RESULT: NO PREDICTION GENERATED", None, self)
                self.booklet.add_step("Expected Output:", self.booklet.actual_output, self)
            elif len(predicted_output) != len(self.booklet.actual_output) or \
                 (len(predicted_output) > 0 and len(predicted_output[0]) != len(self.booklet.actual_output[0])):
                # Shape mismatch
                self.booklet.accuracy = 0.0
                pred_shape = f"{len(predicted_output)}x{len(predicted_output[0]) if predicted_output else 0}"
                actual_shape = f"{len(self.booklet.actual_output)}x{len(self.booklet.actual_output[0])}"
                print(f"\n❌ Shape mismatch! Predicted: {pred_shape}, Expected: {actual_shape}")
                self.booklet.add_step(f"❌ FINAL RESULT: SHAPE MISMATCH (predicted {pred_shape}, expected {actual_shape})", None, self)
                self.booklet.add_step("Expected Output:", self.booklet.actual_output, self)
                self.booklet.add_step("Got:", predicted_output, self)
            else:
                # Calculate percentage of matching cells
                total_cells = len(self.booklet.actual_output) * len(self.booklet.actual_output[0])
                matching_cells = 0
                
                for i in range(len(self.booklet.actual_output)):
                    for j in range(len(self.booklet.actual_output[0])):
                        if predicted_output[i][j] == self.booklet.actual_output[i][j]:
                            matching_cells += 1
                
                self.booklet.accuracy = matching_cells / total_cells if total_cells > 0 else 0.0
                
                if self.booklet.accuracy == 1.0:
                    print(f"\n✅ SUCCESS! Predicted output matches actual output perfectly! (100%)")
                    self.booklet.add_step("✅ FINAL RESULT: PERFECT PREDICTION (100% cells correct)", None, self)
            else:
                    accuracy_pct = self.booklet.accuracy * 100
                    print(f"\n⚠️ Partial match: {matching_cells}/{total_cells} cells correct ({accuracy_pct:.1f}%)")
                    self.booklet.add_step(f"⚠️ FINAL RESULT: PARTIAL MATCH ({matching_cells}/{total_cells} cells, {accuracy_pct:.1f}% accuracy)", None, self)
                    # Show expected output when there's an error
                self.booklet.add_step("Expected Output:", self.booklet.actual_output, self)
                self.booklet.add_step("Got:", predicted_output, self)
        
        return self.booklet


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python arc_booklet_solver.py <task_json_file>")
        sys.exit(1)
    
    task_file = sys.argv[1]
    if not os.path.exists(task_file):
        print(f"Error: Task file '{task_file}' not found")
        sys.exit(1)
    
    try:
        solver = ARCBookletSolver()
        booklet = solver.solve(task_file)
        
        print(f"\n{'='*80}")
        print(f"Solving complete!")
        print(f"Steps generated: {len(booklet.steps)}")
        print(f"Accuracy: {booklet.accuracy}")
        print(f"{'='*80}")
        
        # Save booklet to JSON in test/ directory
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")
        os.makedirs(test_dir, exist_ok=True)
        output_file = os.path.join(test_dir, f"{booklet.task_name}_booklet.json")
        with open(output_file, 'w') as f:
            json.dump(booklet.to_dict(), f, indent=2)
        print(f"Booklet saved to: {output_file}")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    """Main entry point"""

    if len(sys.argv) < 2:

        print("Usage: python arc_booklet_solver.py <task_json_file>")

        sys.exit(1)

    

    task_file = sys.argv[1]

    if not os.path.exists(task_file):

        print(f"Error: Task file '{task_file}' not found")

        sys.exit(1)

    

    try:

        solver = ARCBookletSolver()

        booklet = solver.solve(task_file)

        

        print(f"\n{'='*80}")

        print(f"Solving complete!")

        print(f"Steps generated: {len(booklet.steps)}")

        print(f"Accuracy: {booklet.accuracy}")

        print(f"{'='*80}")

        

        # Save booklet to JSON in test/ directory

        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")

        os.makedirs(test_dir, exist_ok=True)

        output_file = os.path.join(test_dir, f"{booklet.task_name}_booklet.json")

        with open(output_file, 'w') as f:

            json.dump(booklet.to_dict(), f, indent=2)

        print(f"Booklet saved to: {output_file}")

        

        sys.exit(0)

        

    except Exception as e:

        print(f"Error: {e}")

        import traceback

        traceback.print_exc()

        sys.exit(1)





if __name__ == "__main__":

    main()
