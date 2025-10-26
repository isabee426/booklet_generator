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
                        
                        # Add function result to messages
                        messages.append({
                            "role": "tool",
                            "content": json.dumps({"image_path": img_path, "status": "success"}),
                            "tool_call_id": tc.id
                        })
                        print(f"  ✅ Visualization created")
                
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
        """Parse a grid from the AI's response"""
        import re
        
        # Find all bracket-enclosed sequences
        pattern = r'\[[\d,\s]+\]'
        matches = re.findall(pattern, response)
        
        grid = []
        for match in matches:
            # Remove brackets and parse numbers
            numbers = match.strip('[]').replace(' ', '').split(',')
            row = [int(n) for n in numbers if n]
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
        Main solving loop that generates a booklet with iterative rule-building
        Returns a Booklet object
        """
        # Extract task name from file path
        self.current_task_name = os.path.splitext(os.path.basename(task_file))[0]
        self.booklet = Booklet(self.current_task_name)
        
        # Load the task
        task = self.load_task(task_file)
        print(f"\nLoaded task: {task_file}")
        print(f"Task contains {len(task['train'])} training examples and {len(task['test'])} test examples")
        
        # Initialize with empty rules
        current_rules = ""
        
        # Iterative rule building: process each training example one by one
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
                # First example: generate initial rules
                prompt = f"""You are solving an ARC-AGI puzzle. This is the FIRST training example.

Input:
{self.format_grid(example['input'])}

Expected Output:
{self.format_grid(example['output'])}

Follow these meta-instructions to generate detailed step-by-step rules:

1. LOCATE EACH OBJECT: Identify all objects and their locations
2. FIND THE PATTERN: Determine if there's a reference object or overall pattern  
3. APPLY THE PATTERN: Extract the transformation rule

Generate detailed step-by-step rules that explain how to transform the input to the output. For each step, be explicit about:
- What you're identifying/locating
- What transformation you're applying
- The intermediate result

Special note about grid sizes: YOU MUST INFER THE OUTPUT GRID SIZES FROM THE INPUT GRID SIZES! Remember the first training example's output size and what determined it.:
FOR EACH TRAINING EXAMPLE: (dont change grid size except when looking at the new training example)
Output sizes can be determined by something in the puzzle (e.g. size of the output object with no extra tiles, size of the input without the reference object, etc.)
Output sizes can also be determined by the ratio of input grid size to output grid size. 
Determine the output grid size rule based on the input output pair of the first training sample.
Final output size must be homogenous (a rectangle, no matter what), and the rule you use to infer input size to output size must be consistent over all the examples.
Break down the transformation into incremental steps that can be visualized. Try to generate a visualization for every step."""
                
                response = self.call_ai_with_image(prompt, [input_img, output_img])
                current_rules = response
                self.booklet.add_step("Generated Rules:", None, self)
                self.booklet.add_step(current_rules, None, self)
                
            else:
                # Subsequent examples: adapt existing rules
                prompt = f"""You have existing rules that work for the first {i} training examples. Now you need to adapt these rules to also work for training example {i+1}.

EXISTING RULES:
{current_rules}

NEW TRAINING EXAMPLE {i+1}:
Input:
{self.format_grid(example['input'])}

Expected Output:
{self.format_grid(example['output'])}

Analyze this new example and refine your rules so they work for ALL {i+1} training examples seen so far. If the new example contradicts your existing rules, modify them. If it confirms them, strengthen them.

Provide updated detailed step-by-step rules that work for all examples. For each step, be explicit about:
- What you're identifying/locating
- What transformation you're applying
- The intermediate result

Break down the transformation into incremental steps that can be visualized. Visualize any step that you."""
                
                response = self.call_ai_with_image(prompt, [input_img, output_img])
                current_rules = response
                self.booklet.add_step(f"Updated Rules (works for all {i+1} examples):", None, self)
                self.booklet.add_step(current_rules, None, self)
            
            # Test current rules on this example with detailed step-by-step breakdown
            test_prompt = f"""Apply your current rules to training example {i+1} with detailed step-by-step breakdown.

CURRENT RULES:
{current_rules}

TRAINING EXAMPLE {i+1}:
Input:
{self.format_grid(example['input'])}

Expected Output:
{self.format_grid(example['output'])}

Please break down the application of your rules into detailed incremental steps. For each step, show:
1. What you're identifying/locating
2. What transformation you're applying
3. The intermediate result

Generate the final output grid in the exact format with square brackets and comma-separated values."""
            
            test_response = self.call_ai_with_image(test_prompt, [input_img])
            
            # Parse the response to extract intermediate steps
            self.booklet.add_step(f"=== DETAILED STEP-BY-STEP APPLICATION FOR EXAMPLE {i+1} ===", None, self)
            self.booklet.add_step("Step-by-step breakdown:", None, self)
            self.booklet.add_step(test_response, None, self)
            
            # Try to extract intermediate grids from the response
            intermediate_grids = self.extract_intermediate_grids(test_response)
            
            # Add each intermediate step as a separate booklet step
            for j, grid in enumerate(intermediate_grids):
                self.booklet.add_step(f"Intermediate Step {j+1}:", grid, self)
            
            predicted_output = self.parse_grid_from_response(test_response)
            self.booklet.add_step(f"Final Predicted Output for Example {i+1}:", predicted_output, self)
            
            # Check if prediction matches expected output
            if predicted_output == example['output']:
                self.booklet.add_step(f"✅ CORRECT: Rules work for example {i+1}", None, self)
            else:
                self.booklet.add_step(f"❌ INCORRECT: Rules need refinement for example {i+1}", None, self)
                # Only show expected output when there's an error
                self.booklet.add_step("Expected Output:", example['output'], self)
                self.booklet.add_step("Got:", predicted_output, self)
        
        # Final step: apply rules to test input with detailed breakdown
        self.booklet.add_step("=== FINAL TEST: Applying Rules to Test Input ===", None, self)
        
        test_input_img = self.create_grid_image(task['test'][0]['input'], label="test_input")
        
        final_prompt = f"""Apply your final refined rules to the test input with detailed step-by-step breakdown.

FINAL RULES:
{current_rules}

TEST INPUT:
{self.format_grid(task['test'][0]['input'])}

Please break down the application of your rules into detailed incremental steps. For each step, show:
1. What you're identifying/locating
2. What transformation you're applying  
3. The intermediate result

Generate the final output grid in the exact format with square brackets and comma-separated values."""
        
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
            
            if predicted_output == self.booklet.actual_output:
                self.booklet.accuracy = 1.0
                print("\n✅ SUCCESS! Predicted output matches actual output!")
                self.booklet.add_step("✅ FINAL RESULT: CORRECT PREDICTION", None, self)
            else:
                self.booklet.accuracy = 0.0
                print("\n❌ Predicted output does not match actual output")
                self.booklet.add_step("❌ FINAL RESULT: INCORRECT PREDICTION", None, self)
                # Only show expected output when there's an error
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
