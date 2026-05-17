#!/usr/bin/env python3
"""
ARC Booklet Generator
Generates step-by-step instructional booklets with visual comparisons
Based on visual solver approach with meta-instructions
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

# Import the visualizer functions
from arc_visualizer import grid_to_image, ARC_COLORS


class BookletStep:
    """Represents a single step with model output and expected target"""
    def __init__(self, step_number: int, description: str, grid_data: List[List[int]], 
                 target_grid: Optional[List[List[int]]] = None):
        self.step_number = step_number
        self.description = description
        self.grid_data = grid_data  # Model's attempt/output
        self.target_grid = target_grid  # Expected/correct result (if known)
        self.timestamp = datetime.now()
        self.tries = 1  # Number of attempts to reach target
        self.reached_target = False  # Whether model output matches expected
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "step_number": self.step_number,
            "description": self.description,
            "grid_shape": list(np.array(self.grid_data).shape) if self.grid_data else None,
            "target_shape": list(np.array(self.target_grid).shape) if self.target_grid else None,
            "tries": self.tries,
            "reached_target": self.reached_target,
            "timestamp": self.timestamp.isoformat()
        }


class Booklet:
    """Represents a complete instructional booklet"""
    def __init__(self, task_name: str, output_dir: str):
        self.task_name = task_name
        self.output_dir = Path(output_dir) / f"{task_name}_booklet"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.steps: List[BookletStep] = []
        self.input_grid = None
        self.target_output = None
        
    def add_step(self, description: str, grid_data: List[List[int]], 
                 target_grid: Optional[List[List[int]]] = None) -> BookletStep:
        """Add a step to the booklet"""
        step = BookletStep(len(self.steps), description, grid_data, target_grid)
        
        # Check if model reached target
        if target_grid is not None and grid_data == target_grid:
            step.reached_target = True
        
        self.steps.append(step)
        return step
    
    def save_step_image(self, step: BookletStep, grid: List[List[int]], label: str):
        """Save a grid image for a step"""
        img = grid_to_image(grid, cell_size=30)
        img_path = self.output_dir / f"{label}.png"
        img.save(img_path)
        return str(img_path)
    
    def save(self):
        """Save booklet metadata and images"""
        # Save input
        if self.input_grid:
            self.save_step_image(None, self.input_grid, "input")
        
        # Save target output
        if self.target_output:
            self.save_step_image(None, self.target_output, "target_output")
        
        # Save each step
        for step in self.steps:
            # Save model output
            if step.grid_data:
                self.save_step_image(step, step.grid_data, f"step_{step.step_number:03d}")
            
            # Save expected/target if different from model output
            if step.target_grid and step.grid_data != step.target_grid:
                self.save_step_image(step, step.target_grid, f"step_{step.step_number:03d}_expected")
        
        # Save metadata
        metadata = {
            "task_name": self.task_name,
            "steps": [step.to_dict() for step in self.steps],
            "total_steps": len(self.steps),
            "generated_at": datetime.now().isoformat()
        }
        
        with open(self.output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Save README
        readme = self._generate_readme()
        with open(self.output_dir / "README.txt", 'w', encoding='utf-8') as f:
            f.write(readme)
    
    def _generate_readme(self) -> str:
        """Generate README text"""
        return f"""# ARC-AGI Task {self.task_name} - Instructional Booklet

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Files

- input.png: Original puzzle input
- target_output.png: Expected solution
- step_NNN.png: Model output for each step
- step_NNN_expected.png: Expected/correct grid for that step (if different from model output)
- metadata.json: Complete step data

## Steps

Total steps: {len(self.steps)}
Steps reaching target: {sum(1 for s in self.steps if s.reached_target)}

{chr(10).join(f"Step {s.step_number}: {s.description[:80]}{'...' if len(s.description) > 80 else ''}" for s in self.steps)}
"""


class ARCBookletGenerator:
    def __init__(self):
        """Initialize the booklet generator"""
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
        import re
        
        # Strategy 1: Look for bracketed rows [1,2,3]
        pattern = r'\[[\d,\s]+\]'
        matches = re.findall(pattern, response)
        
        grid = []
        for match in matches:
            # Handle both comma-separated and space-separated
            numbers_str = match.strip('[]').replace(',', ' ')
            numbers = [int(n) for n in numbers_str.split() if n.isdigit()]
            if numbers:
                grid.append(numbers)
        
        if grid and len(grid) > 0:
            # Validate grid is rectangular
            row_lengths = [len(row) for row in grid]
            if len(set(row_lengths)) == 1:  # All rows same length
                return grid
        
        # Strategy 2: Look for "GRID:" or "OUTPUT:" section with rows
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
        
        # Strategy 3: Look for lines of comma or space separated digits
        lines = response.split('\n')
        grid = []
        for line in lines:
            # Clean line
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            
            # Extract digits
            numbers = re.findall(r'\d', line)
            if len(numbers) >= 3:  # Assume at least 3 cells wide
                grid.append([int(n) for n in numbers])
        
        if grid and len(grid) >= 3:  # At least 3 rows
            row_lengths = [len(row) for row in grid]
            if len(set(row_lengths)) == 1:  # All rows same length
                return grid
        
        return None
    
    def generate_booklet(self, task_file: str, output_dir: str = "sample_booklets") -> Booklet:
        """Generate a step-by-step instructional booklet"""
        
        # Load task
        task = self.load_task(task_file)
        task_name = Path(task_file).stem
        
        print(f"\n{'='*80}")
        print(f"Generating Booklet for Task: {task_name}")
        print(f"{'='*80}\n")
        
        # Create booklet
        booklet = Booklet(task_name, output_dir)
        
        # Use first training example
        example = task['train'][0]
        booklet.input_grid = example['input']
        booklet.target_output = example['output']
        
        # Create temp dir for intermediate images
        temp_dir = Path("img_tmp")
        temp_dir.mkdir(exist_ok=True)
        
        # Create input/output images
        input_img_path = str(temp_dir / f"{task_name}_input.png")
        output_img_path = str(temp_dir / f"{task_name}_output.png")
        
        img_input = grid_to_image(example['input'], 30)
        img_input.save(input_img_path)
        
        img_output = grid_to_image(example['output'], 30)
        img_output.save(output_img_path)
        
        # PHASE 1: Visual Analysis
        print("Phase 1: Visual Analysis...")
        analyze_prompt = """You are analyzing an ARC-AGI puzzle input.

IMPORTANT: Work from the VISUAL representation. Think about patterns, shapes, and colors.

Analyze this input IMAGE:
- What shapes, objects, or patterns do you SEE? Are there any visual elements that stand out as different or special (reference objects)?
- What colors are present and arranged how?
- Are there reference objects or other clues in the image? If so, what do they say about how to solve the puzzle?
- Describe the visual structure spatially

Be PRECISE about measurements."""

        analysis = self.call_ai(analyze_prompt, [input_img_path])
        # Print safely (handle Unicode characters on Windows)
        try:
            print(f"Analysis: {analysis[:200]}...\n")
        except UnicodeEncodeError:
            print(f"Analysis: [Generated - {len(analysis)} chars]\n")
        
        # PHASE 2: Identify Operations
        print("Phase 2: Identifying Operations...")
        
        operations_prompt = f"""You analyzed the input. Now see the output and identify step-by-step operations.

Your Analysis:
{analysis}

Input grid:
{self.format_grid(example['input'])}

Output grid:
{self.format_grid(example['output'])}

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

        operations_text = self.call_ai(operations_prompt, [input_img_path, output_img_path])
        # Print safely (handle Unicode characters on Windows)
        try:
            print(f"Operations: {operations_text[:200]}...\n")
        except UnicodeEncodeError:
            print(f"Operations: [Generated - {len(operations_text)} chars]\n")
        
        # Parse operations
        operations = []
        for line in operations_text.split('\n'):
            if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                operations.append(line.strip())
        
        print(f"Identified {len(operations)} operations\n")
        
        # PHASE 3: Execute Operations Step-by-Step
        print("Phase 3: Executing Operations...")
        
        current_grid = [row[:] for row in example['input']]
        
        for i, operation in enumerate(operations):
            print(f"Step {i}: Executing...")
            
            # Create image of current state
            current_img_path = str(temp_dir / f"{task_name}_current_{i}.png")
            img_current = grid_to_image(current_grid, 30)
            img_current.save(current_img_path)
            
            # Ask AI to execute this operation
            execute_prompt = f"""Execute this COMPLETE operation step {i+1} of {len(operations)}.

CURRENT GRID:
{self.format_grid(current_grid)}

OPERATION (execute ALL parts of this in ONE go):
{operation}

Apply this ENTIRE operation to the current grid. If this operation has multiple parts (e.g., "for each X do Y"), complete ALL iterations in this single step.

CRITICAL EXECUTION RULES: 
- Execute the COMPLETE operation - if it says "for each", do ALL of them now
- If it has conditionals (if/else), apply them ALL in this step
- Apply ONLY this operation - nothing more
- Do NOT remove or move cells unless this operation explicitly requires it
- PRESERVE DETAILS: Don't accidentally modify objects unless this operation explicitly requires it - keep their exact shapes
- CONSISTENCY: If operation applies to multiple objects, apply it IDENTICALLY to ALL of them
- COMPLETENESS: Don't miss single-cell blocks or small details - count carefully
- EXACT SHAPES: If copying/transforming shapes, preserve their exact dimensions
- You MUST output a complete grid

RESULT GRID:
[row1]
[row2]
..."""

            response = self.call_ai(execute_prompt, [current_img_path])
            
            # Parse resulting grid
            new_grid = self.parse_grid_from_response(response)
            
            if new_grid:
                # Add step to booklet
                # For now, we don't have ideal intermediate states
                # So we compare against final output to see if we're done
                target = example['output'] if i == len(operations) - 1 else None
                step = booklet.add_step(operation, new_grid, target)
                
                current_grid = new_grid
                print(f"  [OK] Step {i} executed")
            else:
                print(f"  [WARN] Could not parse grid from step {i}")
                # Add step anyway with description
                step = booklet.add_step(operation, current_grid, None)
        
        # PHASE 4: Refine Final Output (up to 3 attempts)
        print("\n" + "="*80)
        print("PHASE 4: Refining Final Output")
        print("="*80)
        
        max_refinements = 3
        final_attempts = []
        
        for attempt in range(max_refinements):
            print(f"\nAttempt {attempt + 1}/{max_refinements}...")
            
            # Check if current grid matches target
            if current_grid == example['output']:
                print("[OK] Correct! Final output matches target")
                break
            
            # Last attempt: re-execute all steps from scratch
            if attempt == max_refinements - 1:
                print("  Final attempt - re-executing all steps from input...")
                
                re_exec_grid = [row[:] for row in example['input']]
                
                for step_idx, operation in enumerate(operations):
                    print(f"    Re-executing step {step_idx}...")
                    
                    # Create image of current state
                    re_exec_img_path = str(temp_dir / f"{task_name}_reexec_{step_idx}.png")
                    img_re_exec = grid_to_image(re_exec_grid, 30)
                    img_re_exec.save(re_exec_img_path)
                    
                    # Execute this step
                    re_exec_prompt = f"""Re-execute this step carefully.

CURRENT GRID:
{self.format_grid(re_exec_grid)}

STEP {step_idx + 1} OF {len(operations)}:
{operation}

Apply this step to the current grid carefully, paying attention to ALL details.

CRITICAL: 
- PRESERVE DETAILS: Don't miss single cells or modify shapes accidentally
- CONSISTENCY: Apply to ALL relevant objects
- COMPLETENESS: Count carefully
- Output complete grid

RESULT GRID:
[row1]
[row2]
..."""
                    
                    re_exec_response = self.call_ai(re_exec_prompt, [re_exec_img_path])
                    re_exec_new_grid = self.parse_grid_from_response(re_exec_response)
                    
                    if re_exec_new_grid:
                        re_exec_grid = re_exec_new_grid
                    else:
                        print(f"    [WARN] Could not parse grid from re-execution step {step_idx}")
                
                current_grid = re_exec_grid
                final_attempts.append({
                    "attempt": attempt + 1,
                    "grid": re_exec_grid,
                    "method": "re_execution"
                })
                print(f"  Re-executed all {len(operations)} steps")
                
                # Check if correct now
                if current_grid == example['output']:
                    print("[OK] Correct after re-execution!")
                    break
                else:
                    continue
            
            # Attempts 1-2: Just regenerate final output
            # Create refinement prompt (don't show expected output - let AI figure it out)
            refine_prompt = f"""Your previous analysis and steps to solve this puzzle:

ORIGINAL VISUAL ANALYSIS:
{analysis}

STEPS YOU IDENTIFIED:
{chr(10).join(f"{i+1}. {ops}" for i, ops in enumerate(operations))}

YOUR CURRENT FINAL OUTPUT:
{self.format_grid(current_grid)}

CURRENT OUTPUT IS WRONG. Review your original analysis and steps carefully, then regenerate the final output.

CRITICAL INSTRUCTIONS: 
- THE TRANSFORMATION RULE SHOULD BE SIMPLE AND OBVIOUS from your analysis
- DO NOT remove or move cells unless explicitly required by the transformation steps
- DO NOT arbitrarily delete or modify grid elements
- DO NOT overcomplicate - if the rule is simple, keep it simple
- Focus on faithfully applying your identified steps to the input
- Think about what might have gone wrong during execution:
  * Did you skip a step?
  * Misapply an operation?
  * Miss something from the original analysis?
  * Add unnecessary modifications?
  * Accidentally modify object shapes?
  * Miss single-cell blocks or small details?
  * Forget to apply transformation to ALL relevant objects?
- PRESERVE DETAILS: Don't accidentally change object shapes - keep exact dimensions
- CONSISTENCY: Apply transformations IDENTICALLY to all objects that should be transformed
- COMPLETENESS: Count carefully - don't miss single cells or small blocks
- Review the original analysis: what patterns, shapes, or colors did you identify?
- Apply ONLY the steps you identified - no more, no less
- Apply the steps CONSISTENTLY and COMPLETELY to EVERY relevant object
- Re-examine the input puzzle - what SIMPLE transformation rule did you discover?

Remember: The rule should be OBVIOUS. Execute it CONSISTENTLY on ALL objects without missing details or accidentally modifying shapes.

CORRECT FINAL GRID:
[row1]
[row2]
..."""
            
            # Get current state image
            current_img_path = str(temp_dir / f"{task_name}_current_final_{attempt}.png")
            img_current = grid_to_image(current_grid, 30)
            img_current.save(current_img_path)
            
            # Call AI for refinement (show input but NOT the expected output)
            refine_response = self.call_ai(refine_prompt, [input_img_path, current_img_path])
            
            # Parse refined grid
            refined_grid = self.parse_grid_from_response(refine_response)
            
            if refined_grid:
                current_grid = refined_grid
                final_attempts.append({
                    "attempt": attempt + 1,
                    "grid": refined_grid,
                    "response": refine_response[:200]
                })
                print(f"  Generated refined output (attempt {attempt + 1})")
                
                # Check if correct now
                if current_grid == example['output']:
                    print("[OK] Correct after refinement!")
                    break
            else:
                print(f"  [WARN] Could not parse refined grid")
        
        # Final check
        final_correct = current_grid == example['output']
        
        if final_correct:
            print("\n[SUCCESS] Booklet generation successful - reached target output!")
        else:
            print(f"\n[WARN] Final grid does not match target output after {max_refinements} attempts")
        
        # Update the last step with final refined grid
        if booklet.steps:
            booklet.steps[-1].grid_data = current_grid
            booklet.steps[-1].tries = len(final_attempts) + 1
            booklet.steps[-1].reached_target = final_correct
        
        # Save booklet
        booklet.save()
        print(f"\nBooklet saved to: {booklet.output_dir}")
        
        return booklet


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python arc-booklet-generator.py <task_json_file> [output_dir]")
        print("Example: python arc-booklet-generator.py ../saturn-arc/ARC-AGI-2/ARC-AGI-2/data/training/3e6067c3.json")
        sys.exit(1)
    
    task_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "sample_booklets"
    
    if not os.path.exists(task_file):
        print(f"Error: Task file '{task_file}' not found")
        sys.exit(1)
    
    try:
        generator = ARCBookletGenerator()
        booklet = generator.generate_booklet(task_file, output_dir)
        
        print(f"\n{'='*80}")
        print(f"BOOKLET GENERATED!")
        print(f"{'='*80}")
        print(f"Task: {booklet.task_name}")
        print(f"Steps: {len(booklet.steps)}")
        print(f"Location: {booklet.output_dir}")
        print(f"\nView with: streamlit run streamlit_booklet_viewer.py")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

