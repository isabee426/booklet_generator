#!/usr/bin/env python3
"""
ARC-AGI Booklet Solver V2 - Step-by-Step Execution
Generates instructions then executes them incrementally with visualization
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


class ARCBookletSolverV2:
    def __init__(self):
        """Initialize the ARC booklet solver V2 with API credentials"""
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
    
    def parse_operations_from_response(self, response: str) -> List[str]:
        """Parse a list of operations from the AI's response"""
        # Look for numbered operations
        lines = response.split('\n')
        operations = []
        
        for line in lines:
            # Match patterns like "1. Operation", "Operation 1:", "Step 1:", etc.
            if re.match(r'^\s*(\d+[\.\:\)]|Operation\s+\d+|Step\s+\d+)', line, re.IGNORECASE):
                operations.append(line.strip())
        
        return operations if operations else [response]  # Fallback to whole response
    
    def solve(self, task_file: str) -> Booklet:
        """
        Main solving loop with iterative refinement across training examples
        Returns a Booklet object
        """
        # Extract task name from file path
        self.current_task_name = os.path.splitext(os.path.basename(task_file))[0]
        self.booklet = Booklet(self.current_task_name)
        
        # Load the task
        task = self.load_task(task_file)
        print(f"\nLoaded task: {task_file}")
        print(f"Task contains {len(task['train'])} training examples and {len(task['test'])} test examples")
        
        # Initialize rule and operations
        current_rule = None
        current_operations = []
        
        # ITERATIVE REFINEMENT: Process each training example
        for train_idx, example in enumerate(task['train']):
            print(f"\n{'='*60}")
            print(f"TRAINING EXAMPLE {train_idx + 1}")
            print(f"{'='*60}")
            
            self.booklet.add_step(f"=== TRAINING EXAMPLE {train_idx + 1} ===", None, self)
            self.booklet.add_step("Input:", example['input'], self)
            
            # Only show output for first example (others see it only during refinement)
            if train_idx == 0:
                self.booklet.add_step("Expected Output:", example['output'], self)
            
            input_img = self.create_grid_image(example['input'], label=f"train{train_idx+1}_input")
            output_img = self.create_grid_image(example['output'], label=f"train{train_idx+1}_output")
            
            if train_idx == 0:
                # FIRST EXAMPLE: First analyze input visually, then identify THE RULE
                
                # VISUAL ANALYSIS FIRST (Input only)
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
                
                # NOW IDENTIFY THE RULE (with input + output)
                rule_prompt = f"""You are solving an ARC-AGI puzzle. You've analyzed the input. Now see the expected output and identify THE transformation rule.

KEY PRINCIPLE: There is ONE transformation rule that will work for ALL examples.

Your Visual Analysis:
{analyze_response}

Input:
{self.format_grid(example['input'])}

Output:
{self.format_grid(example['output'])}

TASK: Identify THE SINGLE transformation rule. Describe it in 2-3 sentences focusing on:
- What objects/patterns to identify
- What transformation to apply
- How to determine output size

Keep it CONCISE (under 200 words) and HIGH-LEVEL. Think VISUALLY."""

                rule_response = self.call_ai_with_image(rule_prompt, [input_img, output_img])
                current_rule = rule_response
                self.booklet.add_step("THE RULE (from example 1):", None, self)
                self.booklet.add_step(current_rule, None, self)
                
                # Break into operations
                operations_prompt = f"""Break THE RULE into 3-7 operations at the right level of detail for a human to execute.

THE RULE:
{current_rule}

TASK: Think like a human solving this puzzle step-by-step. Each operation should be:

THE RIGHT LEVEL OF DETAIL (like explaining to a person):
- Specific enough to execute unambiguously
- General enough to work on different inputs
- References visual elements (colors, shapes, positions)
- Not too abstract: "Transform the grid" ❌ (too vague)
- Not too detailed: "Set cell [5,0] to 2, cell [5,1] to 2..." ❌ (too specific)
- Just right: "For each colored pixel in the input, fill its entire row with that color" ✅

VISUAL AND EXECUTABLE:
- Reference what you see: "the bottom row", "each colored square", "the reference pattern"
- Clear action: "draw", "fill", "copy", "extend", "tile", "repeat"
- Specific measurements when needed: "2 cells wide", "across entire row", "until reaching the border"
- Constraints: "without overwriting X", "only for Y", "starting from Z"

Think: "How would I explain this to someone looking at the image?"

Example (good balance of specificity):
'1. Locate the reference row at the bottom showing the color sequence
 2. For each color in the sequence, find the matching colored block in the grid
 3. Draw a horizontal bridge of that color connecting consecutive blocks, making the bridge the same width as the blocks
 4. Ensure bridges do not overwrite the block borders'

Format as numbered list. Under 300 words total."""

                operations_response = self.call_ai_with_image(operations_prompt, [input_img, output_img])
                self.booklet.add_step("OPERATIONS (from example 1):", None, self)
                self.booklet.add_step(operations_response, None, self)
                
                current_operations = self.parse_operations_from_response(operations_response)
                print(f"\n📝 Generated {len(current_operations)} operations")
                
            else:
                # SUBSEQUENT EXAMPLES: Execute operations and check per-operation feedback
                print(f"\n🔄 Executing operations on example {train_idx + 1}...")
                self.booklet.add_step(f"Testing current operations on example {train_idx + 1}...", None, self)
            
            # Execute operations step-by-step (SINGLE EXECUTION)
            self.booklet.add_step(f"=== STEP-BY-STEP EXECUTION (Example {train_idx + 1}) ===", None, self)
            
            current_grid = [row[:] for row in example['input']]  # Deep copy
            operations_failed = False
            failed_operation_idx = None
            
            for i, operation in enumerate(current_operations):
                execute_prompt = f"""Execute operation {i+1} of {len(current_operations)}.

CURRENT GRID:
{self.format_grid(current_grid)}

OPERATION:
{operation}

Apply this operation. Output RESULT GRID.

RESULT GRID:
[row1]
[row2]
...

CRITICAL: MUST output a complete grid."""

                current_img = self.create_grid_image(current_grid, label=f"train{train_idx+1}_op{i}")
                execute_response = self.call_ai_with_image(execute_prompt, [current_img])
                
                new_grid = self.parse_grid_from_response(execute_response)
                
                if new_grid:
                    current_grid = new_grid
                    self.booklet.add_step(f"After Operation {i+1}:", current_grid, self)
                    print(f"  ✅ Operation {i+1} executed")
                else:
                    print(f"  ⚠️ Warning: Could not parse grid from operation {i+1}")
                    self.booklet.add_step(f"⚠️ Operation {i+1} failed to parse", None, self)
                    operations_failed = True
                    failed_operation_idx = i
                    break
            
            # Verify final result
            if current_grid == example['output']:
                self.booklet.add_step(f"✅ All operations successful for example {train_idx + 1}! Result matches expected output.", None, self)
                print(f"✅ Execution successful for example {train_idx + 1}")
            else:
                # Show which operation likely caused the error
                self.booklet.add_step(f"❌ Result does NOT match expected output for example {train_idx + 1}", None, self)
                print(f"❌ Execution failed for example {train_idx + 1}")
                
                # Only refine if not first example
                if train_idx > 0:
                    # NOW reveal the expected output (only when operations failed)
                    self.booklet.add_step("❌ Operations failed. NOW showing expected output:", example['output'], self)
                    
                    # REFINE operations based on failure
                    refine_prompt = f"""Your current operations failed on this example. Now you can see the expected output. Refine them.

CRITICAL: Keep operations HIGH-LEVEL and GENERALIZABLE. Do not over-fit to specific examples.

CURRENT RULE:
{current_rule}

CURRENT OPERATIONS:
{chr(10).join(current_operations)}

EXAMPLE {train_idx + 1}:
Input:
{self.format_grid(example['input'])}

Expected Output (now revealed):
{self.format_grid(example['output'])}

Your operations produced:
{self.format_grid(current_grid) if current_grid else "Invalid grid"}

TASK: Refine the operations so they work for ALL {train_idx + 1} examples. 

Think like a human - find the right level of detail:
- Specific enough: Someone could execute these by looking at any valid input
- General enough: Works on variations (different colors, sizes, layouts)
- Visual: References what you SEE, not grid coordinates
- Executable: Clear actions with measurements when needed

Keep 3-7 operations. Under 300 words.

Find the more GENERAL pattern that unifies all examples. Do not add specific cases - find the abstract pattern that explains both examples.

Output REFINED OPERATIONS as numbered list."""

                    refine_response = self.call_ai_with_image(refine_prompt, [input_img, output_img])
                    self.booklet.add_step(f"REFINED OPERATIONS (works for {train_idx + 1} examples):", None, self)
                    self.booklet.add_step(refine_response, None, self)
                    
                    current_operations = self.parse_operations_from_response(refine_response)
                    print(f"\n📝 Refined to {len(current_operations)} operations")
        
        # FINAL PHASE: Apply refined operations to test input
        print(f"\n{'='*60}")
        print("APPLYING TO TEST INPUT")
        print(f"{'='*60}")
        
        self.booklet.add_step("=== TEST INPUT ===", None, self)
        test_input = task['test'][0]['input']
        self.booklet.add_step("Test Input:", test_input, self)
        
        test_img = self.create_grid_image(test_input, label="test_input")
        
        # Apply final refined operations to test
        current_grid = [row[:] for row in test_input]
        
        for i, operation in enumerate(current_operations):
            execute_prompt = f"""Execute operation {i+1} on the TEST input.

CURRENT GRID:
{self.format_grid(current_grid)}

OPERATION:
{operation}

Apply the SAME operation to this test grid. Output RESULT GRID.

RESULT GRID:
[row1]
[row2]
..."""

            current_img = self.create_grid_image(current_grid, label=f"test_op{i}")
            execute_response = self.call_ai_with_image(execute_prompt, [current_img])
            
            new_grid = self.parse_grid_from_response(execute_response)
            if new_grid:
                current_grid = new_grid
                self.booklet.add_step(f"Test After Operation {i+1}:", current_grid, self)
        
        # Final result
        self.booklet.final_prediction = current_grid
        self.booklet.add_step("FINAL TEST PREDICTION:", current_grid, self)
        
        # Calculate accuracy
        if 'output' in task['test'][0] and task['test'][0]['output']:
            self.booklet.actual_output = task['test'][0]['output']
            
            if current_grid is None:
                self.booklet.accuracy = 0.0
                print("\n❌ No prediction generated")
                self.booklet.add_step("❌ NO PREDICTION", None, self)
            elif len(current_grid) != len(self.booklet.actual_output) or \
                 len(current_grid[0]) != len(self.booklet.actual_output[0]):
                self.booklet.accuracy = 0.0
                print(f"\n❌ Shape mismatch!")
                self.booklet.add_step("❌ SHAPE MISMATCH", None, self)
            else:
                # Cell-by-cell accuracy
                total_cells = len(self.booklet.actual_output) * len(self.booklet.actual_output[0])
                matching_cells = sum(
                    1 for i in range(len(self.booklet.actual_output))
                    for j in range(len(self.booklet.actual_output[0]))
                    if current_grid[i][j] == self.booklet.actual_output[i][j]
                )
                self.booklet.accuracy = matching_cells / total_cells
                
                if self.booklet.accuracy == 1.0:
                    print(f"\n✅ PERFECT! 100% accuracy")
                    self.booklet.add_step("✅ PERFECT PREDICTION (100%)", None, self)
                else:
                    print(f"\n⚠️ Partial: {matching_cells}/{total_cells} cells ({self.booklet.accuracy*100:.1f}%)")
                    self.booklet.add_step(f"⚠️ PARTIAL MATCH ({matching_cells}/{total_cells} cells, {self.booklet.accuracy*100:.1f}%)", None, self)
                    self.booklet.add_step("Expected:", self.booklet.actual_output, self)
        
        return self.booklet


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python arc-booklets-solver-v2-stepwise.py <task_json_file>")
        sys.exit(1)
    
    task_file = sys.argv[1]
    if not os.path.exists(task_file):
        print(f"Error: Task file '{task_file}' not found")
        sys.exit(1)
    
    try:
        solver = ARCBookletSolverV2()
        booklet = solver.solve(task_file)
        
        print(f"\n{'='*80}")
        print(f"Solving complete!")
        print(f"Steps generated: {len(booklet.steps)}")
        print(f"Accuracy: {booklet.accuracy}")
        print(f"{'='*80}")
        
        # Save booklet to JSON in test/ directory
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")
        os.makedirs(test_dir, exist_ok=True)
        output_file = os.path.join(test_dir, f"{booklet.task_name}_booklet_v2.json")
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

