#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual Step Generator V7 - Generalized Steps First Architecture

V7 features:
- Consolidated prompt templates (no repetition)
- Removed unused code (detect_objects fallback)
- Streamlined prompts while preserving all functionality
- Better code organization

This version processes puzzles object-by-object:
1. Detects objects in input/output grids (model-based only)
2. For each object:
   - Crops to object bounding box
   - Finds corresponding object in output (or marks as new)
   - Generates transformation steps on cropped grids
   - Tests hypothesis (3 retries)
   - Uncropped back to full grid with modifications
3. For new objects: Shows blank crop first with position metadata
4. Maintains same output format as v2/v3

Usage:
    python scripts/visual_step_generator_v4.py --puzzle 05f2a901
"""

import os
import sys
import json
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from io import BytesIO
from collections import defaultdict

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


class VisualStepGeneratorV7:
    """Visual Step Generator V7 - Generalized Steps First, Then Apply to Training
    
    Architecture:
    1. Analyze ALL training examples to extract generalized steps
    2. Each generalized step: "for each (CONDITION, OBJECT) apply transformations"
    3. Apply generalized steps to training examples to generate training booklets
    4. Use same generalized steps for test examples
    
    Features:
    - Comprehensive puzzle analysis (input-input, output-output, input-output, reference objects)
    - Generate generalized steps BEFORE training booklets
    - Each step: (CONDITION, OBJECT) → transformations
    - Full tool awareness: negative space, scaled objects, rotate, reflect, repeat pattern
    - Line object special handling
    - Conditional transformations with 1-to-1 correlations
    - Object completion, fitting, drawing
    - Dimension awareness, reshaping, counting
    """
    
    def __init__(self, model: str = "gpt-5-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.tools = self._define_tools()
        print(f"✓ Initialized V7 with {model}")
    
    def _define_tools(self) -> List[Dict]:
        """Define function calling tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "generate_grid",
                    "description": "Generate the resulting grid after applying the action",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "grid": {
                                "type": "array",
                                "description": "2D array representing the grid state after action",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "integer"}
                                }
                            },
                            "visual_analysis": {
                                "type": "string",
                                "description": "1-2 sentences explaining what this step does"
                            }
                        },
                        "required": ["grid", "visual_analysis"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "detect_objects",
                    "description": "Detect all distinct objects in a grid. CRITICAL: Each separate filled region is a DISTINCT object, even if they share the same color. Each connected region of filled cells should be detected as its own separate object. Lines follow different rules - sequential, can change color by section, drawn in order.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "objects": {
                                "type": "array",
                                "description": "List of detected objects",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "bbox": {
                                            "type": "array",
                                            "description": "Bounding box [min_row, min_col, max_row, max_col] - 0-indexed",
                                            "items": {"type": "integer"},
                                            "minItems": 4,
                                            "maxItems": 4
                                        },
                                        "colors": {
                                            "type": "array",
                                            "description": "Primary color(s) of the object",
                                            "items": {"type": "integer"}
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "Description of what the object is"
                                        },
                                        "size": {
                                            "type": "integer",
                                            "description": "Number of cells in the object"
                                        }
                                    },
                                    "required": ["bbox", "colors", "description", "size"]
                                }
                            }
                        },
                        "required": ["objects"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "match_objects",
                    "description": "Match objects between input and output grids",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "matches": {
                                "type": "array",
                                "description": "List of matches between input and output objects",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "input_idx": {
                                            "type": "integer",
                                            "description": "Input object index (0-based), or null if new object"
                                        },
                                        "output_idx": {
                                            "type": "integer",
                                            "description": "Output object index (0-based), or null if removed",
                                            "nullable": True
                                        },
                                        "reason": {
                                            "type": "string",
                                            "description": "Brief reason for the match"
                                        }
                                    },
                                    "required": ["input_idx", "output_idx", "reason"]
                                }
                            }
                        },
                        "required": ["matches"]
                    }
                }
            }
        ]
    
    # ==================== PROMPT TEMPLATES (Consolidated) ====================
    
    def _get_base_instructions(self) -> str:
        """Base instructions used across all prompts"""
        return """⚠️ IMPORTANT:
- Work ONLY on the cropped region provided
- Use the generate_grid tool to provide your result
- Grid coordinates are 0-indexed"""
    
    def _format_grid(self, grid: List[List[int]]) -> str:
        """Format grid as text"""
        return '\n'.join(['[' + ', '.join(str(cell) for cell in row) + ']' for row in grid])
    
    def _format_training_examples(self, examples: List[Dict], max_examples: int = 3) -> Tuple[str, List[Image.Image]]:
        """Format training examples for prompts - returns (text, images)"""
        text = ""
        images = []
        for i, ex in enumerate(examples[:max_examples]):
            text += f"\nTRAINING EXAMPLE {i+1}:\n"
            text += f"Input: {self._format_grid(ex['input'])}\n"
            text += f"Output: {self._format_grid(ex['output'])}\n"
            images.append(grid_to_image(ex['input'], 40))
            images.append(grid_to_image(ex['output'], 40))
        return text, images
    
    def _format_reference_info(self, reference_info: Optional[Dict], is_test: bool = False) -> str:
        """Format reference object info for prompts - emphasize the specific part used"""
        if not reference_info:
            return ""
        
        ref_obj = reference_info['object']
        part_used = reference_info.get('part_used', {})
        part_name = part_used.get('part_name', 'full object')
        how_used = part_used.get('how_used', reference_info.get('how_to_use', ''))
        how_adapts = reference_info.get('how_adapts_steps', '')
        
        ref_type = "adapts steps to test input" if is_test else "used as template"
        
        result = f"""
⭐ REFERENCE OBJECT (constant - {ref_type}):
- Object: {ref_obj.get('description', 'object')}
- ⚠️ IMPORTANT: Focus on THIS PART of the reference: **{part_name}**
- Part Description: {part_used.get('description', 'N/A')}
- How This Part Guides Transformation: {how_used}
- Reasoning: {reference_info.get('reasoning', '')[:200]}..."""
        
        if reference_info.get('pattern_across_examples'):
            result += f"\n- Pattern: {reference_info.get('pattern_across_examples', '')}"
        
        if is_test and how_adapts:
            result += f"\n- How Adapts Steps: {how_adapts}"
        
        result += f"\n\n⚠️ CRITICAL: Pay attention to the **{part_name}** of the reference object when performing this transformation!"
        
        if is_test:
            result += f"\nIn TEST MODE: Use this reference object to ADAPT the transformation steps to this specific test input!"
            result += f"\nThe reference object tells you how to adapt the pattern from training examples to this test input."
        else:
            result += f"\nUse this specific part consistently as shown across all training examples!"
        
        return result
    
    # ==================== CORE METHODS ====================
    
    def crop_to_object(self, grid: List[List[int]], bbox: Tuple[int, int, int, int],
                      padding: int = 2, max_crop_ratio: float = 0.8) -> Tuple[List[List[int]], Dict]:
        """Crop grid to object bounding box with padding. Never crops entire grid."""
        min_r, min_c, max_r, max_c = bbox
        h, w = len(grid), len(grid[0])
        
        # Calculate original bbox size
        bbox_h = max_r - min_r + 1
        bbox_w = max_c - min_c + 1
        
        # Check if bbox covers too much of the grid (more than max_crop_ratio)
        grid_area = h * w
        bbox_area = bbox_h * bbox_w
        crop_ratio = bbox_area / grid_area if grid_area > 0 else 1.0
        
        if crop_ratio > max_crop_ratio:
            # Bbox is too large - reduce padding or reject
            print(f"  ⚠️ Warning: Object bbox covers {crop_ratio:.1%} of grid (max {max_crop_ratio:.1%})")
            print(f"    Original bbox: ({min_r}, {min_c}) to ({max_r}, {max_c}) = {bbox_h}×{bbox_w}")
            print(f"    Grid size: {h}×{w}")
            # Reduce padding to minimum
            padding = 0
            print(f"    Reducing padding to {padding} to avoid cropping entire grid")
        
        min_r = max(0, min_r - padding)
        min_c = max(0, min_c - padding)
        max_r = min(h - 1, max_r + padding)
        max_c = min(w - 1, max_c + padding)
        
        # Final check: ensure we're not cropping the entire grid
        crop_h = max_r - min_r + 1
        crop_w = max_c - min_c + 1
        final_crop_ratio = (crop_h * crop_w) / grid_area if grid_area > 0 else 1.0
        
        if final_crop_ratio > max_crop_ratio:
            # Still too large - use minimal crop around object only
            print(f"  ⚠️ Crop still too large ({final_crop_ratio:.1%}), using minimal crop")
            min_r = max(0, min_r)
            min_c = max(0, min_c)
            max_r = min(h - 1, max_r)
            max_c = min(w - 1, max_c)
            # Ensure at least some margin
            if max_r - min_r + 1 >= h * max_crop_ratio:
                margin_r = int((h - (max_r - min_r + 1)) / 2)
                min_r = max(0, min_r - margin_r)
                max_r = min(h - 1, max_r + margin_r)
            if max_c - min_c + 1 >= w * max_crop_ratio:
                margin_c = int((w - (max_c - min_c + 1)) / 2)
                min_c = max(0, min_c - margin_c)
                max_c = min(w - 1, max_c + margin_c)
        
        cropped = [row[min_c:max_c+1] for row in grid[min_r:max_r+1]]
        
        # Final validation
        final_h, final_w = len(cropped), len(cropped[0]) if cropped else 0
        final_ratio = (final_h * final_w) / grid_area if grid_area > 0 else 1.0
        
        if final_ratio > max_crop_ratio:
            print(f"  ❌ ERROR: Crop still covers {final_ratio:.1%} of grid! Using object bbox only (no padding)")
            # Last resort: use exact bbox with no padding
            min_r, min_c, max_r, max_c = bbox
            min_r = max(0, min_r)
            min_c = max(0, min_c)
            max_r = min(h - 1, max_r)
            max_c = min(w - 1, max_c)
            cropped = [row[min_c:max_c+1] for row in grid[min_r:max_r+1]]
        
        metadata = {
            'original_bbox': bbox,
            'crop_bbox': (min_r, min_c, max_r, max_c),
            'offset': (min_r, min_c),
            'size': (len(cropped), len(cropped[0]) if cropped else 0)
        }
        
        return cropped, metadata
    
    def uncrop_to_full_grid(self, cropped_grid: List[List[int]], 
                           full_grid: List[List[int]],
                           metadata: Dict,
                           input_grid: List[List[int]] = None,
                           processed_objects: List[Dict] = None,
                           background_color: int = 0) -> List[List[int]]:
        """Paste cropped grid back into full grid at original position"""
        result = [row[:] for row in full_grid]
        min_r, min_c = metadata['offset']
        crop_bbox = metadata.get('crop_bbox', metadata.get('original_bbox'))
        
        if crop_bbox:
            crop_min_r, crop_min_c, crop_max_r, crop_max_c = crop_bbox
        else:
            crop_min_r, crop_min_c = min_r, min_c
            crop_max_r = min_r + len(cropped_grid) - 1
            crop_max_c = min_c + len(cropped_grid[0]) - 1 if cropped_grid else min_c
        
        for r_idx, row in enumerate(cropped_grid):
            for c_idx, val in enumerate(row):
                full_r = min_r + r_idx
                full_c = min_c + c_idx
                if 0 <= full_r < len(result) and 0 <= full_c < len(result[0]):
                    result[full_r][full_c] = val
        
        return result
    
    def compare_grids(self, grid1: List[List[int]], grid2: List[List[int]]) -> float:
        """Compare two grids and return accuracy (0.0 to 1.0)"""
        if not grid1 or not grid2:
            return 0.0
        
        arr1 = np.array(grid1)
        arr2 = np.array(grid2)
        
        if arr1.shape != arr2.shape:
            return 0.0
        
        return float(np.mean(arr1 == arr2))
    
    def find_differing_cells(self, grid1: List[List[int]], grid2: List[List[int]]) -> List[Tuple[int, int]]:
        """Find all cell positions where grids differ"""
        if not grid1 or not grid2:
            return []
        
        diff_cells = []
        min_h = min(len(grid1), len(grid2))
        min_w = min(len(grid1[0]) if grid1 else 0, len(grid2[0]) if grid2 else 0)
        
        for r in range(min_h):
            for c in range(min_w):
                if grid1[r][c] != grid2[r][c]:
                    diff_cells.append((r, c))
        
        return diff_cells
    
    def call_api(self, prompt: str, images: List[Image.Image] = None, 
                current_grid: List[List[int]] = None, 
                input_grid: List[List[int]] = None, 
                valid_colors: set = None,
                tool_choice: Dict = None) -> Tuple[Any, Optional[List[List[int]]]]:
        """Call OpenAI API - handles function calling for object detection/matching"""
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
            
            messages = [{"role": "user", "content": content}]
            
            tool_choice_param = tool_choice if tool_choice else "auto"
            
            if "gpt-5" in self.model:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice=tool_choice_param,
                    max_completion_tokens=8000
                )
            else:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice=tool_choice_param,
                    max_tokens=4000
                )
            
            message = resp.choices[0].message
            tool_calls = message.tool_calls if hasattr(message, 'tool_calls') and message.tool_calls else []
            
            description = ""
            grid = None
            parsed_result = None
            
            if tool_calls:
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_name == "generate_grid":
                        grid = function_args["grid"]
                        description = function_args.get("visual_analysis", "")
                    elif function_name == "detect_objects":
                        parsed_result = function_args
                        description = f"Detected {len(function_args.get('objects', []))} objects"
                    elif function_name == "match_objects":
                        parsed_result = function_args
                        description = f"Matched {len(function_args.get('matches', []))} objects"
            
            if not tool_calls:
                description = message.content if message.content else ""
                return description, None
            
            if parsed_result is not None:
                return parsed_result, None
            
            return description, grid
            
        except Exception as e:
            print(f"  ❌ API Error: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", None
    
    # ==================== PUZZLE LOADING ====================
    
    def load_arc_puzzle(self, puzzle_id: str) -> Dict:
        """Load ARC puzzle from JSON file"""
        # Try multiple paths
        possible_paths = [
            Path("C:/Users/Isabe/New folder (3)/saturn-arc/ARC-AGI-2/ARC-AGI-2/data/training") / f"{puzzle_id}.json",
            Path(__file__).parent.parent.parent / "saturn-arc" / "ARC-AGI-2" / "ARC-AGI-2" / "data" / "training" / f"{puzzle_id}.json",
            Path(__file__).parent.parent / "ARC-AGI-2" / "data" / "training" / f"{puzzle_id}.json",
            Path("ARC-AGI-2/data/training") / f"{puzzle_id}.json",
        ]
        
        for puzzle_path in possible_paths:
            if puzzle_path.exists():
                print(f"  Loading puzzle from: {puzzle_path}")
                with open(puzzle_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        raise FileNotFoundError(f"Puzzle {puzzle_id}.json not found in any of the expected locations")
    
    # ==================== ANALYSIS METHODS ====================
    
    # ==================== COMPREHENSIVE ANALYSIS METHODS ====================
    
    def comprehensive_analysis(self, training_examples: List[Dict]) -> Dict:
        """Comprehensive puzzle analysis including all required segments"""
        print("\n" + "="*80)
        print("COMPREHENSIVE PUZZLE ANALYSIS")
        print("="*80)
        
        analysis = {}
        
        # 1. Input-Input Differences
        analysis['input_input'] = self.analyze_input_input_differences(training_examples)
        
        # 2. Output-Output Similarities
        analysis['output_output'] = self.analyze_output_output_similarities(training_examples)
        
        # 3. Input-Output Transitions
        analysis['input_output'] = self.analyze_input_output_transitions(training_examples)
        
        # 4. Reference Objects
        analysis['reference_objects'] = self.analyze_reference_objects(training_examples)
        
        # 5. Grid Size Changes
        analysis['grid_size'] = self.analyze_grid_size_changes(training_examples)
        
        # 6. Training Sample to Training Sample
        analysis['training_to_training'] = self.analyze_training_to_training(training_examples)
        
        return analysis
    
    def analyze_input_input_differences(self, training_examples: List[Dict]) -> Dict:
        """Analyze differences between training inputs and what they say about output differences"""
        print("\n" + "="*80)
        print("ANALYZING INPUT-INPUT DIFFERENCES")
        print("="*80)
        
        input_grids = [ex['input'] for ex in training_examples]
        output_grids = [ex['output'] for ex in training_examples]
        
        grids_text = ""
        for i, (in_grid, out_grid) in enumerate(zip(input_grids, output_grids)):
            grids_text += f"\nTRAINING {i+1}:\n"
            grids_text += f"Input: {self._format_grid(in_grid)}\n"
            grids_text += f"Output: {self._format_grid(out_grid)}\n"
        
        prompt = f"""Analyze the differences between training inputs and how they relate to output differences:

{grids_text}

ANALYSIS TASKS:
1. What stays the same across all inputs? (objects, colors, shapes, patterns, reference objects)
2. What differs between inputs? (different objects, colors, positions, sizes)
3. How do input differences relate to output differences?
4. What information do input differences provide about the transformation?
5. Do inputs build upon each other? Is there a progression or sequence?
6. What patterns emerge from comparing inputs?

Return JSON with:
- "same_across_inputs": ["list of what stays the same"]
- "differences": ["list of what differs"]
- "input_to_output_relationship": "how input differences relate to output differences"
- "progression": "do inputs build upon each other?"
- "patterns": "patterns from comparing inputs"
"""
        
        images = [grid_to_image(grid, 50) for grid in input_grids]
        result, _ = self.call_api(prompt, images)
        
        if isinstance(result, dict):
            return result
        
        return self._parse_analysis_json(result if isinstance(result, str) else str(result), "input-input differences")
    
    def analyze_output_output_similarities(self, training_examples: List[Dict]) -> Dict:
        """Analyze similarities between training outputs - anything consistent should be in test output"""
        print("\n" + "="*80)
        print("ANALYZING OUTPUT-OUTPUT SIMILARITIES")
        print("="*80)
        
        output_grids = [ex['output'] for ex in training_examples]
        
        grids_text = ""
        for i, out_grid in enumerate(output_grids):
            grids_text += f"\nOUTPUT {i+1}:\n{self._format_grid(out_grid)}\n"
        
        prompt = f"""Analyze similarities between all training outputs:

{grids_text}

CRITICAL: Anything consistent (exact same) across ALL training outputs should be in the test output.

ANALYSIS TASKS:
1. What is EXACTLY the same in all outputs? (objects, colors, shapes, positions, patterns)
2. What patterns are consistent across outputs?
3. What should definitely appear in the test output?

Return JSON with:
- "exact_same": ["list of what is exactly the same"]
- "consistent_patterns": ["list of consistent patterns"]
- "must_be_in_test": ["what must appear in test output"]
"""
        
        images = [grid_to_image(grid, 50) for grid in output_grids]
        result, _ = self.call_api(prompt, images)
        
        if isinstance(result, dict):
            return result
        
        return self._parse_analysis_json(result if isinstance(result, str) else str(result), "output-output similarities")
    
    def analyze_input_output_transitions(self, training_examples: List[Dict]) -> Dict:
        """Analyze transitions from input to output for each training example"""
        print("\n" + "="*80)
        print("ANALYZING INPUT-OUTPUT TRANSITIONS")
        print("="*80)
        
        transitions = []
        for i, ex in enumerate(training_examples):
            input_grid = ex['input']
            output_grid = ex['output']
            
            prompt = f"""Analyze the transition from input to output:

Input:
{self._format_grid(input_grid)}

Output:
{self._format_grid(output_grid)}

ANALYSIS TASKS:
1. What is similar between input and output?
2. What is common in the transition steps?
3. What changes? (color, shape, position, size)
4. What transformation types are used? (recolor, reshape, move, scale, rotate, reflect, etc.)

Return JSON with:
- "similarities": ["what stays the same"]
- "common_transitions": ["common transition steps"]
- "changes": ["what changes"]
- "transformation_types": ["list of transformation types"]
"""
            
            images = [grid_to_image(input_grid, 50), grid_to_image(output_grid, 50)]
            result, _ = self.call_api(prompt, images)
            
            if isinstance(result, dict):
                transitions.append(result)
            else:
                parsed = self._parse_analysis_json(result if isinstance(result, str) else str(result), f"input-output transition {i+1}")
                transitions.append(parsed)
        
        return {"transitions": transitions}
    
    def analyze_reference_objects(self, training_examples: List[Dict]) -> Dict:
        """Detect and analyze reference objects"""
        print("\n" + "="*80)
        print("DETECTING REFERENCE OBJECTS")
        print("="*80)
        
        input_grids = [ex['input'] for ex in training_examples]
        output_grids = [ex['output'] for ex in training_examples]
        
        grids_text = ""
        for i, (in_grid, out_grid) in enumerate(zip(input_grids, output_grids)):
            grids_text += f"\nTRAINING {i+1}:\n"
            grids_text += f"Input: {self._format_grid(in_grid)}\n"
            grids_text += f"Output: {self._format_grid(out_grid)}\n"
        
        prompt = f"""Detect reference objects in this puzzle:

{grids_text}

REFERENCE OBJECTS:
- Stay similar (location, color, shape) between inputs
- Provide clues for solving the puzzle
- Can be used for shape or color templates
- Order might matter (left to right, up to down)
- Can be dividers (bar of different color)
- If they move, explain why

ANALYSIS TASKS:
1. Identify reference objects (same across inputs/outputs)
2. How are they used? (shape template, color template, divider, order template)
3. If they move, why?
4. If no solid reference object, what provides location/color/shape clues?

Return JSON with:
- "reference_objects": [{{"description": "...", "usage": "...", "location": "...", "why_moves": "..."}}]
- "no_reference_clues": "what provides clues if no reference object"
"""
        
        images = []
        for in_grid, out_grid in zip(input_grids, output_grids):
            images.append(grid_to_image(in_grid, 50))
            images.append(grid_to_image(out_grid, 50))
        
        result, _ = self.call_api(prompt, images)
        
        if isinstance(result, dict):
            return result
        
        return self._parse_analysis_json(result if isinstance(result, str) else str(result), "reference objects")
    
    def analyze_grid_size_changes(self, training_examples: List[Dict]) -> Dict:
        """Analyze grid size changes between input and output"""
        print("\n" + "="*80)
        print("ANALYZING GRID SIZE CHANGES")
        print("="*80)
        
        size_changes = []
        for i, ex in enumerate(training_examples):
            input_grid = ex['input']
            output_grid = ex['output']
            input_h, input_w = len(input_grid), len(input_grid[0]) if input_grid else 0
            output_h, output_w = len(output_grid), len(output_grid[0]) if output_grid else 0
            
            size_changes.append({
                "training": i + 1,
                "input_size": (input_h, input_w),
                "output_size": (output_h, output_w),
                "input_larger": input_h > output_h or input_w > output_w,
                "input_smaller": input_h < output_h or input_w < output_w,
                "same_size": input_h == output_h and input_w == output_w
            })
        
        return {"size_changes": size_changes}
    
    def analyze_training_to_training(self, training_examples: List[Dict]) -> Dict:
        """Analyze training sample to training sample relationships"""
        print("\n" + "="*80)
        print("ANALYZING TRAINING SAMPLE TO TRAINING SAMPLE")
        print("="*80)
        
        input_grids = [ex['input'] for ex in training_examples]
        output_grids = [ex['output'] for ex in training_examples]
        
        grids_text = ""
        for i, (in_grid, out_grid) in enumerate(zip(input_grids, output_grids)):
            grids_text += f"\nTRAINING {i+1}:\n"
            grids_text += f"Input: {self._format_grid(in_grid)}\n"
            grids_text += f"Output: {self._format_grid(out_grid)}\n"
        
        prompt = f"""Analyze relationships between training samples:

{grids_text}

ANALYSIS TASKS:
1. What is different about the inputs? How do they provide more information about transition steps?
2. Do they build upon each other?
3. Is there a correspondence (if-then) being built upon?
4. What is the same among inputs? Are similar objects going through similar transitions?

Return JSON with:
- "input_differences": "how inputs differ and provide more information"
- "build_upon_each_other": "do they build upon each other?"
- "correspondence": "if-then correspondences"
- "same_among_inputs": "what stays the same"
- "similar_transitions": "do similar objects go through similar transitions?"
"""
        
        images = []
        for in_grid, out_grid in zip(input_grids, output_grids):
            images.append(grid_to_image(in_grid, 50))
            images.append(grid_to_image(out_grid, 50))
        
        result, _ = self.call_api(prompt, images)
        
        if isinstance(result, dict):
            return result
        
        return self._parse_analysis_json(result if isinstance(result, str) else str(result), "training-to-training")
    
    def _parse_analysis_json(self, text: Any, analysis_type: str) -> Dict:
        """Parse JSON from analysis response"""
        import re
        
        if isinstance(text, dict):
            return text
        
        if not isinstance(text, str):
            text = str(text)
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        
        # Remove trailing commas before closing braces/brackets
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"  ⚠️ Failed to parse {analysis_type}: {e}")
            print(f"    Text: {text[:200]}...")
            return {}
    
    def analyze_puzzle_pattern(self, all_training_examples: List[Dict]) -> Dict:
        """Analyze ALL training examples to find transitions and optimal ordering"""
        print("\n" + "="*80)
        print("ANALYZING TRANSITIONS AND OPTIMAL ORDERING")
        print("="*80)
        
        input_grids = [ex['input'] for ex in all_training_examples]
        output_grids = [ex['output'] for ex in all_training_examples]
        
        grids_text = ""
        for i, (in_grid, out_grid) in enumerate(zip(input_grids, output_grids)):
            grids_text += f"\nEXAMPLE {i+1}:\n"
            grids_text += f"Input ({len(in_grid)}×{len(in_grid[0])}):\n{self._format_grid(in_grid)}\n"
            grids_text += f"Output ({len(out_grid)}×{len(out_grid[0])}):\n{self._format_grid(out_grid)}\n"
        
        prompt = f"""Analyze this ARC-AGI puzzle to find ALL TRANSITIONS (what changes from input to output).

{grids_text}

IMPORTANT: Colors are numeric values in the grid:
- 0 = black (background)
- 1 = blue
- 2 = red  
- 3 = green
- 4 = yellow
- etc.

Always refer to colors as "color 1", "color 2", "color 3", etc. based on these numeric values.

CRITICAL TASKS:

1. **Find Reference Objects** (constant across all inputs/outputs):
   - Objects/regions that DON'T change between input and output
   - These are templates/anchors used to guide transformations
   - Identify by: same position, same shape, same color NUMBER across ALL examples
   - Describe colors as "color X" where X is the numeric value

2. **Find ALL Transitions** (what actually changes):
   - Cell-level changes: individual cells that change color/position
   - Region changes: connected regions that transform (color NUMBER changes, position, shape)
   - New objects: regions that appear in output but not input
   - Removed objects: regions that disappear from input to output
   - For each transition, identify: what changes, where it changes, how it changes
   - Describe color changes as "color X → color Y" using numeric values

3. **Determine Optimal Processing Order** (best portrays solution):
   - Reference objects should be identified FIRST (but not transformed - they're constant)
   - Then process transitions in order that best shows reasoning:
     * Dependencies: if transformation A depends on B, process B first
     * Visual clarity: process in order that makes pattern most obvious
     * Logical flow: what order tells the story best?
   - Avoid processing the same transition twice

Return JSON:
{{
  "reference_objects": [
    {{
      "description": "what the reference object is (include color number, e.g., 'plus shape of color 1')",
      "why_reference": "why it's constant/used as template",
      "position_pattern": "where it appears (consistent across examples)"
    }}
  ],
  "transitions": [
    {{
      "type": "cell_change|region_transform|new_object|removed_object",
      "description": "what changes (include color numbers, e.g., 'color 2 → color 4')",
      "input_location": "where in input (bbox or cell positions)",
      "output_location": "where in output",
      "transformation": "how it changes (color NUMBER change, move, etc.)",
      "depends_on": ["reference_object_1", ...]  // what this depends on
    }}
  ],
  "optimal_ordering": [
    {{
      "step_type": "identify_reference|transform|new_object|remove",
      "transition_idx": 0,  // index in transitions array, or null for reference
      "reasoning": "why this step comes at this position"
    }}
  ],
  "analysis": "overall pattern explanation (use color numbers)"
}}

Focus on finding WHAT CHANGES and in WHAT ORDER to show the reasoning best."""
        
        images = []
        for in_grid, out_grid in zip(input_grids, output_grids):
            images.append(grid_to_image(in_grid, 40))
            images.append(grid_to_image(out_grid, 40))
        
        description, _ = self.call_api(prompt, images)
        
        response_text = description if description else ""
        
        # Try multiple JSON extraction strategies
        analysis_data = None
        
        try:
            import re
            # Strategy 1: Try to find JSON block
            json_patterns = [
                r'\{[\s\S]*?\}',  # Greedy match
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested braces
                r'```json\s*(\{[\s\S]*?\})\s*```',  # Code block
                r'```\s*(\{[\s\S]*?\})\s*```',  # Generic code block
            ]
            
            for pattern in json_patterns:
                matches = re.finditer(pattern, response_text, re.DOTALL)
                for match in matches:
                    try:
                        json_str = match.group(1) if match.groups() else match.group(0)
                        json_str = json_str.strip()
                        # Remove trailing commas before closing braces/brackets
                        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        analysis_data = json.loads(json_str)
                        if isinstance(analysis_data, dict):
                            break
                    except:
                        continue
                if analysis_data:
                    break
            
            # Strategy 2: Try parsing entire response
            if not analysis_data:
                cleaned = re.sub(r'```[a-z]*\n?', '', response_text)
                cleaned = re.sub(r'```', '', cleaned)
                cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
                analysis_data = json.loads(cleaned)
            
            if analysis_data and isinstance(analysis_data, dict):
                print(f"\n✓ Transition analysis complete:")
                ref_count = len(analysis_data.get('reference_objects', []))
                trans_count = len(analysis_data.get('transitions', []))
                print(f"  Reference objects: {ref_count}")
                print(f"  Transitions found: {trans_count}")
                print(f"  Optimal ordering: {len(analysis_data.get('optimal_ordering', []))} steps")
                return analysis_data
        except Exception as e:
            print(f"  ⚠️ Parse failed: {e}")
            if response_text:
                print(f"  Response preview: {response_text[:300]}...")
            import traceback
            traceback.print_exc()
        
        print("  ⚠️ Using fallback analysis")
        return {
            "reference_objects": [],
            "transitions": [],
            "optimal_ordering": [],
            "analysis": "Analysis failed - using fallback"
        }
    
    def detect_objects_connected_components(self, grid: List[List[int]], background_color: int = 0) -> List[Dict]:
        """Detect objects using connected components (4-connected) as fallback"""
        arr = np.array(grid)
        h, w = arr.shape
        objects = []
        visited = np.zeros((h, w), dtype=bool)
        
        def get_neighbors(r, c):
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    neighbors.append((nr, nc))
            return neighbors
        
        def flood_fill(start_r, start_c, color):
            cells = []
            stack = [(start_r, start_c)]
            min_r, min_c = start_r, start_c
            max_r, max_c = start_r, start_c
            
            while stack:
                r, c = stack.pop()
                if visited[r, c] or arr[r, c] != color:
                    continue
                
                visited[r, c] = True
                cells.append((r, c))
                min_r = min(min_r, r)
                min_c = min(min_c, c)
                max_r = max(max_r, r)
                max_c = max(max_c, c)
                
                for nr, nc in get_neighbors(r, c):
                    if not visited[nr, nc] and arr[nr, nc] == color:
                        stack.append((nr, nc))
            
            if cells:
                return {
                    'bbox': [min_r, min_c, max_r, max_c],
                    'colors': [color],
                    'description': f"connected component of color {color}",
                    'size': len(cells)
                }
            return None
        
        for r in range(h):
            for c in range(w):
                if not visited[r, c] and arr[r, c] != background_color:
                    obj = flood_fill(r, c, int(arr[r, c]))
                    if obj:
                        objects.append(obj)
        
        objects.sort(key=lambda o: (o['bbox'][0], o['bbox'][1]))
        return objects
    
    def detect_objects_with_model(self, grid: List[List[int]], grid_type: str, retry_count: int = 3, use_hybrid: bool = False) -> List[Dict]:
        """Detect objects using model only - filters out single-cell objects"""
        model_objects = []
        
        # Try model detection first
        for attempt in range(retry_count):
            prompt = f"""Detect all distinct objects in this {grid_type.upper()} grid.

Grid ({len(grid)}×{len(grid[0])}):
{self._format_grid(grid)}

CRITICAL: An object is a visually distinct pattern, shape, or connected region that is MULTIPLE CELLS BIG.
- Objects are usually 2+ cells (lines, shapes, patterns)
- DO NOT detect single isolated cells as objects - these are usually just background noise
- Focus on meaningful visual patterns: lines, shapes, blocks, symbols, etc.
- Ignore scattered single cells that are just colored background

For each object provide:
- bbox: [min_row, min_col, max_row, max_col] (0-indexed coordinates)
- colors: [color_number] where color_number is the actual numeric value from the grid (0=black, 1=blue, 2=red, 3=green, 4=yellow, etc.)
- description: "what the object is" - MUST include "color X" where X is the numeric color value
- size: number_of_cells (must be 2 or more - single cells are NOT objects)

IMPORTANT: 
- In the description, always refer to colors as "color 1", "color 2", "color 3", etc. based on the numeric values in the grid.
- For example: "3×3 square of color 2" or "vertical line of color 4" or "plus shape of color 1".
- Only detect objects that are 2+ cells - ignore single isolated cells.
- If you see many single colored cells scattered around, those are background, not objects.

Use detect_objects function. bbox must be valid 0-indexed coordinates within grid bounds."""
            
            images = [grid_to_image(grid, 50)]
            result, _ = self.call_api(prompt, images, tool_choice={"type": "function", "function": {"name": "detect_objects"}})
            
            try:
                if isinstance(result, dict) and 'objects' in result:
                    objects = result['objects']
                    
                    # Validate objects and filter out single-cell objects
                    valid_objects = []
                    for obj in objects:
                        bbox = obj.get('bbox')
                        if bbox and len(bbox) == 4:
                            min_r, min_c, max_r, max_c = bbox
                            if (0 <= min_r <= max_r < len(grid) and 
                                0 <= min_c <= max_c < len(grid[0])):
                                # Calculate actual size from bbox
                                width = max_c - min_c + 1
                                height = max_r - min_r + 1
                                size = width * height
                                
                                # Filter out single-cell objects (1×1)
                                if size >= 2:
                                    # Update size if not provided or incorrect
                                    if obj.get('size', 0) < size:
                                        obj['size'] = size
                                    valid_objects.append(obj)
                                else:
                                    print(f"    Filtered out single-cell object at ({min_r}, {min_c})")
                    
                    if valid_objects:
                        print(f"  ✓ Model detected {len(valid_objects)} objects (attempt {attempt + 1}/{retry_count})")
                        model_objects = valid_objects
                        break
            except Exception as e:
                print(f"  ⚠️ Attempt {attempt + 1}/{retry_count}: {e}")
        
        # Filter out any remaining single-cell objects that might have slipped through
        if model_objects:
            filtered_objects = []
            for obj in model_objects:
                bbox = obj.get('bbox')
                if bbox and len(bbox) == 4:
                    min_r, min_c, max_r, max_c = bbox
                    width = max_c - min_c + 1
                    height = max_r - min_r + 1
                    size = width * height
                    if size >= 2:
                        filtered_objects.append(obj)
                    else:
                        print(f"    Filtered out single-cell object: {obj.get('description', 'N/A')}")
            
            if filtered_objects:
                print(f"  ✓ Final: {len(filtered_objects)} multi-cell objects detected")
                return filtered_objects
            else:
                print(f"  ⚠️ All detected objects were single-cell, retrying...")
        
        raise RuntimeError(f"Object detection failed after {retry_count} attempts - no multi-cell objects found")
    
    def match_objects_with_model(self, input_objects: List[Dict], output_objects: List[Dict],
                                 input_grid: List[List[int]], output_grid: List[List[int]]) -> Dict[int, Optional[int]]:
        """Match objects using model (no fallback)"""
        prompt = f"""Match input objects to output objects.

INPUT OBJECTS ({len(input_objects)}):
{json.dumps([{k: v for k, v in obj.items() if k != 'cells'} for obj in input_objects], indent=2)}

OUTPUT OBJECTS ({len(output_objects)}):
{json.dumps([{k: v for k, v in obj.items() if k != 'cells'} for obj in output_objects], indent=2)}

IMPORTANT: Colors are numeric values (0=black, 1=blue, 2=red, 3=green, 4=yellow, etc.).
Match objects based on their transformations, considering color changes (e.g., color 2 → color 4).

For each input object, provide corresponding output object index (or null if removed/new).
Use match_objects function."""
        
        images = [grid_to_image(input_grid, 50), grid_to_image(output_grid, 50)]
        result, _ = self.call_api(prompt, images, tool_choice={"type": "function", "function": {"name": "match_objects"}})
        
        try:
            if isinstance(result, dict) and 'matches' in result:
                matches_data = result['matches']
                matches = {}
                
                for match in matches_data:
                    input_idx = match.get('input_idx')
                    output_idx = match.get('output_idx')
                    if input_idx is not None and 0 <= input_idx < len(input_objects):
                        matches[input_idx] = output_idx if output_idx is not None and 0 <= output_idx < len(output_objects) else None
                
                if matches:
                    print(f"  ✓ Matched {len([v for v in matches.values() if v is not None])} objects")
                    return matches
        except Exception as e:
            print(f"  ⚠️ Matching failed: {e}")
        
        raise RuntimeError("Object matching failed")
    
    def find_transitions(self, input_grid: List[List[int]], output_grid: List[List[int]], 
                        puzzle_analysis: Dict) -> List[Dict]:
        """Find all transitions (changes) between input and output grids"""
        transitions = []
        
        # Get transitions from puzzle analysis if available
        analysis_transitions = puzzle_analysis.get('transitions', [])
        
        if analysis_transitions:
            # Use transitions from analysis
            for trans in analysis_transitions:
                transitions.append({
                    'type': trans.get('type', 'region_transform'),
                    'description': trans.get('description', ''),
                    'input_location': trans.get('input_location', ''),
                    'output_location': trans.get('output_location', ''),
                    'transformation': trans.get('transformation', ''),
                    'depends_on': trans.get('depends_on', [])
                })
        else:
            # Fallback: find cell-level differences
            diff_cells = []
            for r in range(min(len(input_grid), len(output_grid))):
                for c in range(min(len(input_grid[0]) if input_grid else 0, 
                                 len(output_grid[0]) if output_grid else 0)):
                    if input_grid[r][c] != output_grid[r][c]:
                        diff_cells.append((r, c, input_grid[r][c], output_grid[r][c]))
            
            if diff_cells:
                transitions.append({
                    'type': 'cell_change',
                    'description': f"{len(diff_cells)} cells change",
                    'input_location': f"{len(diff_cells)} cell positions",
                    'output_location': 'same positions',
                    'transformation': 'color changes',
                    'depends_on': []
                })
        
        return transitions
    
    def get_reference_part_for_transformation(self, ref_info: Dict, input_obj: Dict,
                                            output_obj: Optional[Dict],
                                            all_training_examples: List[Dict]) -> Dict:
        """Determine which specific part of the reference object is used for this transformation"""
        # Get reference parts from analysis if available
        ref_parts = ref_info.get('reference_parts_used', {})
        parts_list = ref_parts.get('parts_used', [])
        
        if parts_list:
            # Use the primary feature or first part
            primary = ref_parts.get('primary_feature', '')
            for part in parts_list:
                if part.get('part_name', '') == primary or not primary:
                    return part
        
        # Fallback: analyze on the fly
        ref_obj = ref_info['object']
        ref_bbox = ref_obj.get('bbox')
        
        prompt = f"""Which specific part of the REFERENCE OBJECT is used for this transformation?

REFERENCE OBJECT:
{self._format_grid(self.crop_to_object(all_training_examples[0]['input'], ref_bbox)[0]) if ref_bbox else 'N/A'}

INPUT OBJECT TO TRANSFORM:
{self._format_grid(self.crop_to_object(all_training_examples[0]['input'], input_obj.get('bbox'))[0]) if input_obj.get('bbox') else 'N/A'}

OUTPUT OBJECT (target):
{self._format_grid(self.crop_to_object(all_training_examples[0]['output'], input_obj.get('bbox'))[0]) if output_obj and input_obj.get('bbox') else 'N/A'}

IMPORTANT: Colors are numeric values (0=black, 1=blue, 2=red, 3=green, 4=yellow, etc.).
Always refer to colors as "color 1", "color 2", "color 3", etc. based on the numeric values.

Identify which specific part/feature of the reference (color number, shape, position, size, etc.) guides this transformation.

Return JSON:
{{
  "part_name": "specific part name (include color number if applicable, e.g., 'color 2 center')",
  "description": "what this part is (include color number)",
  "how_used": "how this part guides the transformation",
  "location": "where in reference"
}}"""
        
        images = []
        if ref_bbox:
            images.append(grid_to_image(self.crop_to_object(all_training_examples[0]['input'], ref_bbox)[0], 80))
        if input_obj.get('bbox'):
            images.append(grid_to_image(self.crop_to_object(all_training_examples[0]['input'], input_obj['bbox'])[0], 80))
            if output_obj:
                images.append(grid_to_image(self.crop_to_object(all_training_examples[0]['output'], input_obj['bbox'])[0], 80))
        
        description, _ = self.call_api(prompt, images)
        
        try:
            import re
            json_match = re.search(r'\{[\s\S]*?\}', description)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass
        
        # Fallback
        return {
            "part_name": "full object",
            "description": "entire reference object",
            "how_used": "used as template/pattern",
            "location": "full reference"
        }
    
    def identify_test_reference_objects(self, test_input_grid: List[List[int]],
                                       test_input_objects: List[Dict],
                                       all_training_examples: List[Dict],
                                       transformation_rule: Optional[Dict] = None) -> List[Dict]:
        """Identify reference objects in test input by comparing to training examples"""
        print(f"  Analyzing test input against {len(all_training_examples)} training examples...")
        
        # Build prompt to identify reference objects
        examples_text = ""
        example_images = []
        
        for i, example in enumerate(all_training_examples):
            input_grid = example['input']
            output_grid = example['output']
            examples_text += f"\nTRAINING EXAMPLE {i+1}:\n"
            examples_text += f"Input: {self._format_grid(input_grid)}\n"
            examples_text += f"Output: {self._format_grid(output_grid)}\n"
            example_images.append(grid_to_image(input_grid, 40))
            example_images.append(grid_to_image(output_grid, 40))
        
        test_objects_text = ""
        for i, obj in enumerate(test_input_objects):
            test_objects_text += f"  Object {i+1}: {obj.get('description', 'N/A')}, colors {obj.get('colors', [])}, bbox {obj.get('bbox')}\n"
        
        prompt = f"""Identify REFERENCE OBJECTS in the TEST INPUT.

TEST INPUT:
{self._format_grid(test_input_grid)}

TEST INPUT OBJECTS:
{test_objects_text}

TRAINING EXAMPLES:
{examples_text}

TASK: Identify which objects in the TEST INPUT are REFERENCE OBJECTS (constant objects that stay the same between input and output in training examples).

Reference objects tell you:
- How the transformation pattern should be ADAPTED to this specific test input
- What stays constant (anchors/landmarks)
- How to position/align transformations relative to these anchors

Look for objects in test input that:
1. Match objects that stay constant in training examples (same color, similar shape/pattern)
2. Are likely to remain unchanged in the output (based on training pattern)
3. Can serve as anchors for adapting the transformation steps

Return JSON:
{{
  "reference_objects": [
    {{
      "object_idx": 0,  // index in test_input_objects
      "description": "what this reference object is",
      "why_reference": "why it's constant/used as anchor",
      "how_adapts_steps": "how this reference object helps adapt steps to test input",
      "position_pattern": "where it appears/how it's positioned"
    }}
  ],
  "adaptation_guide": "how reference objects guide step adaptation for this test input"
}}

Focus on objects that help ADAPT the transformation pattern to this specific test input."""
        
        test_image = grid_to_image(test_input_grid, 50)
        all_images = [test_image]
        all_images.extend(example_images)
        
        description, _ = self.call_api(prompt, all_images)
        
        response_text = description if description else ""
        reference_objects = []
        
        try:
            import re
            json_patterns = [
                r'\{[\s\S]*?\}',
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
                r'```json\s*(\{[\s\S]*?\})\s*```',
                r'```\s*(\{[\s\S]*?\})\s*```',
            ]
            
            for pattern in json_patterns:
                matches = re.finditer(pattern, response_text, re.DOTALL)
                for match in matches:
                    try:
                        json_str = match.group(1) if match.groups() else match.group(0)
                        json_str = json_str.strip()
                        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        ref_data = json.loads(json_str)
                        if isinstance(ref_data, dict):
                            ref_list = ref_data.get('reference_objects', [])
                            for ref_info in ref_list:
                                obj_idx = ref_info.get('object_idx')
                                if obj_idx is not None and 0 <= obj_idx < len(test_input_objects):
                                    reference_objects.append({
                                        'object_idx': obj_idx,
                                        'object': test_input_objects[obj_idx],
                                        'reasoning': ref_info.get('why_reference', ''),
                                        'how_adapts_steps': ref_info.get('how_adapts_steps', ''),
                                        'pattern': ref_info.get('position_pattern', ''),
                                        'is_constant': True
                                    })
                            break
                    except:
                        continue
                if reference_objects:
                    break
        except Exception as e:
            print(f"    ⚠️ Parse failed: {e}")
        
        return reference_objects
    
    def get_reference_objects_from_analysis(self, puzzle_analysis: Dict, 
                                           input_objects: List[Dict]) -> List[Dict]:
        """Extract reference objects from puzzle analysis"""
        ref_objects = []
        analysis_refs = puzzle_analysis.get('reference_objects', [])
        
        for ref_info in analysis_refs:
            # Try to match reference description to an input object
            ref_desc = ref_info.get('description', '').lower()
            for idx, obj in enumerate(input_objects):
                obj_desc = obj.get('description', '').lower()
                # Simple matching - could be improved
                if any(word in obj_desc for word in ref_desc.split()[:3]) or \
                   any(word in ref_desc for word in obj_desc.split()[:3]):
                    ref_objects.append({
                        'object_idx': idx,
                        'object': obj,
                        'reasoning': ref_info.get('why_reference', ''),
                        'pattern': ref_info.get('position_pattern', ''),
                        'is_constant': True  # Reference objects don't change
                    })
                    break
        
        return ref_objects
    
    def analyze_reference_parts(self, ref_obj: Dict, all_training_examples: List[Dict],
                                input_grid: List[List[int]], output_grid: Optional[List[List[int]]]) -> Dict:
        """Analyze which parts/features of the reference object are used for transformations"""
        print(f"  Analyzing reference object parts...")
        
        # Build prompt to identify which parts of reference are used
        examples_text = ""
        images = []
        
        for i, example in enumerate(all_training_examples[:3]):  # Use first 3 examples
            ex_input = example['input']
            ex_output = example['output']
            examples_text += f"\nEXAMPLE {i+1}:\nInput: {self._format_grid(ex_input)}\nOutput: {self._format_grid(ex_output)}\n"
            images.append(grid_to_image(ex_input, 50))
            images.append(grid_to_image(ex_output, 50))
        
        ref_bbox = ref_obj.get('bbox')
        ref_crop = self.crop_to_object(input_grid, ref_bbox)[0] if ref_bbox else None
        
        prompt = f"""Analyze the REFERENCE OBJECT to identify which specific parts/features are used for transformations.

REFERENCE OBJECT:
{self._format_grid(ref_crop) if ref_crop else 'N/A'}

Description: {ref_obj.get('description', 'N/A')}
Bbox: {ref_bbox}
Colors: {ref_obj.get('colors', [])} (numeric values: 0=black, 1=blue, 2=red, 3=green, 4=yellow, etc.)

TRAINING EXAMPLES:
{examples_text}

TASK: Identify which specific parts/features of the reference object are used to guide transformations:
- Color properties: Refer to colors as "color 1", "color 2", etc. based on numeric values
- Shape properties (specific shapes/patterns)
- Position properties (relative positions)
- Size properties (specific sizes)
- Structural features (specific parts like edges, centers, corners, etc.)

IMPORTANT: Always refer to colors using numeric encoding: "color 1", "color 2", "color 3", etc.

Return JSON:
{{
  "parts_used": [
    {{
      "part_name": "e.g., center color 2, edge pattern color 4, corner shape",
      "description": "what this part is (include color number)",
      "how_used": "how this part guides transformations",
      "location": "where in reference object (bbox or relative position)"
    }}
  ],
  "primary_feature": "the most important feature used (include color number if applicable)",
  "usage_pattern": "how the reference guides transformations consistently"
}}"""
        
        if ref_crop:
            images.insert(0, grid_to_image(ref_crop, 80))  # Larger image of reference
        
        description, _ = self.call_api(prompt, images)
        
        try:
            import re
            json_match = re.search(r'\{[\s\S]*?\}', description)
            if json_match:
                parts_data = json.loads(json_match.group(0))
                print(f"    ✓ Identified {len(parts_data.get('parts_used', []))} reference parts")
                return parts_data
        except Exception as e:
            print(f"    ⚠️ Parts analysis failed: {e}")
        
        # Fallback
        return {
            "parts_used": [{"part_name": "full object", "description": "entire reference object", "how_used": "used as template"}],
            "primary_feature": "full object",
            "usage_pattern": "Reference object used as template"
        }
    
    def _analyze_transformation_reason(self, cropped_input: List[List[int]], cropped_output: List[List[int]],
                                      obj_properties: Dict, comprehensive_analysis: Dict,
                                      puzzle_context: str = "") -> Dict:
        """Mini-analysis to explain why and under what conditions a transformation is needed.
        Analyzes cropped input vs cropped target and provides reasoning."""
        analysis_prompt = f"""Analyze why this transformation is needed and what conditions it fulfills:

Cropped Input:
{self._format_grid(cropped_input)}

Cropped Target:
{self._format_grid(cropped_output)}

Object Properties:
- Location: {obj_properties.get('input_location', 'N/A')}
- Colors: {obj_properties.get('colors', [])}
- Shape: {obj_properties.get('shape', 'N/A')}
- Size: {obj_properties.get('size', (0, 0))}
- Object type: {obj_properties.get('object_type', 'solid_object')}

Comprehensive Puzzle Analysis Context:
{json.dumps(comprehensive_analysis, indent=2) if comprehensive_analysis else "N/A"}

Puzzle Context: {puzzle_context}

Analyze:
1. What specific changes are needed? (color, shape, position, size, etc.)
2. WHY is this transformation needed? (what pattern/rule does it follow?)
3. What CONDITIONS must be fulfilled? (e.g., "if object matches reference", "if color is X", etc.)
4. How does this relate to the overall puzzle transformation pattern?
5. What transformation type(s) would best achieve this?

Return your analysis as JSON with fields:
- "changes_needed": ["list of specific changes"]
- "reasoning": "explanation of why this transformation is needed"
- "conditions": ["list of conditions being fulfilled"]
- "pattern_relation": "how this relates to the puzzle pattern"
- "recommended_transformations": ["list of transformation types"]
- "step_description": "detailed description including reasons and conditions"
"""
        
        input_img = grid_to_image(cropped_input, 50)
        output_img = grid_to_image(cropped_output, 50) if cropped_output else None
        
        imgs = [input_img]
        if output_img:
            imgs.append(output_img)
        
        result, _ = self.call_api(analysis_prompt, imgs)
        
        if isinstance(result, dict):
            return result
        
        # Parse JSON response
        parsed = self._parse_analysis_json(result if isinstance(result, str) else str(result), "transformation analysis")
        
        # Ensure required fields
        if not isinstance(parsed, dict):
            parsed = {}
        
        parsed.setdefault("changes_needed", [])
        parsed.setdefault("reasoning", "Transformation needed to match target")
        parsed.setdefault("conditions", [])
        parsed.setdefault("pattern_relation", "Follows puzzle transformation pattern")
        parsed.setdefault("recommended_transformations", [])
        parsed.setdefault("step_description", "Transform object to match target")
        
        return parsed
    
    def generate_object_transformation(self, cropped_input: List[List[int]],
                                      cropped_output: Optional[List[List[int]]],
                                      object_num: int, total_objects: int,
                                      reference_info: Optional[Dict] = None,
                                      attempt_num: int = 1,
                                      is_test: bool = False,
                                      all_training_examples: Optional[List[Dict]] = None,
                                      booklet_pattern: Optional[Dict] = None,
                                      transformation_rule: Optional[Dict] = None,
                                      generalized_step_sequence: Optional[List[Dict]] = None,
                                      transition_determinants: Optional[Dict] = None,
                                      input_obj: Optional[Dict] = None,
                                      corresponding_training_step: Optional[Dict] = None,
                                      object_step_sequence: Optional[List[Dict]] = None,
                                      training_booklet_steps: Optional[List[Dict]] = None,
                                      transformation_analysis: Optional[Dict] = None) -> Tuple[str, List[List[int]]]:
        """Generate transformation for cropped object - consolidated prompt"""
        ref_text = self._format_reference_info(reference_info, is_test=is_test)
        grid_size = f"{len(cropped_input)}×{len(cropped_input[0])}"
        
        if is_test and cropped_output is None:
            # Test mode: use transformation rule and booklet pattern
            rule_text = ""
            pattern_text = ""
            
            if transformation_rule:
                rule_desc = transformation_rule.get('rule_description', 'N/A')
                gen = transformation_rule.get('generalization', {})
                abstract_rule = gen.get('abstract_rule', 'N/A')
                color_logic = gen.get('color_logic', 'N/A')
                shape_logic = gen.get('shape_logic', 'N/A')
                app_guide = transformation_rule.get('application_guide', {})
                
                rule_text = f"""

TRANSFORMATION RULE ANALYSIS (from training booklets - adapt using test reference objects):
Rule: {rule_desc}

Abstract Rule: {abstract_rule}

Color Logic: {color_logic}

Shape Logic: {shape_logic}

CRITICAL ADAPTATION PROCESS:
1. This is the GENERALIZED RULE from training booklets
2. Use TEST REFERENCE OBJECTS to adapt:
   - Map training reference colors → test reference colors
   - Map training reference positions → test reference positions
   - Measure spacing relative to test reference objects
   - Apply patterns centered/aligned to test reference objects
3. Apply the ADAPTED rule to transform the test object
4. Maintain the same transformation pattern, just adapted to test input

How to Apply to Test Input:
{app_guide.get('for_test_input', 'Apply rule based on input properties')}

Check First: {app_guide.get('check_first', 'Check input properties')}
Then Apply: {app_guide.get('then_apply', 'Apply transformation')}

Input Properties Analysis:
{json.dumps(transformation_rule.get('input_properties', {}), indent=2)}

Transformation Process:
{json.dumps(transformation_rule.get('transformation_process', {}), indent=2)}
"""
            
            if booklet_pattern:
                pattern_text = f"""

STEP-BY-STEP PATTERN FROM TRAINING BOOKLETS (adapt using test reference objects):
Generalization: {booklet_pattern.get('generalization', 'N/A')}

Transformation Rules:
{json.dumps(booklet_pattern.get('transformation_rules', []), indent=2)}

CRITICAL ADAPTATION PROCESS:
1. This is the PATTERN from training booklets (exact steps, colors, spacing)
2. Use TEST REFERENCE OBJECTS to adapt:
   - Map training reference positions → test reference positions
   - Map training reference colors → test reference colors
   - Measure spacing relative to test reference objects
   - Apply patterns centered/aligned to test reference objects
3. Apply the ADAPTED pattern to transform the test object
4. Maintain the same transformation pattern, just adapted to test input
"""
            
            # Determine transition based on object properties
            transition_determinant_text = ""
            if transition_determinants and input_obj:
                transitions = transition_determinants.get('transition_determinants', [])
                obj_colors = input_obj.get('colors', [])
                obj_desc = input_obj.get('description', '').lower()
                
                # Find matching transition
                matching_transition = None
                for trans in transitions:
                    props = trans.get('determining_properties', {})
                    trans_colors = props.get('colors', [])
                    trans_shapes = props.get('shapes', [])
                    
                    color_match = any(str(c) in [str(tc) for tc in trans_colors] for c in obj_colors) if trans_colors else True
                    shape_match = any(s in obj_desc for s in trans_shapes) if trans_shapes else True
                    
                    if color_match and shape_match:
                        matching_transition = trans
                        break
                
                if matching_transition:
                    trans_type = matching_transition.get('transition_type', 'N/A')
                    props = matching_transition.get('determining_properties', {})
                    trans_rule = matching_transition.get('transition_rule', {})
                    app_logic = matching_transition.get('application_logic', {})
                    
                    transition_determinant_text = f"""

TRANSITION DETERMINATION (based on object properties):
Object Properties:
- Colors: {obj_colors}
- Description: {input_obj.get('description', 'N/A')}
- Size: {input_obj.get('size', 'N/A')}

Determined Transition: {trans_type}

Determining Properties (what triggered this):
- Colors: {props.get('colors', [])}
- Shapes: {props.get('shapes', [])}
- Positions: {props.get('positions', 'N/A')}
- Relationships: {props.get('relationships', 'N/A')}

Transition Rule (exact transformation):
- Exact transformation: {trans_rule.get('exact_transformation', 'N/A')}
- Exact pattern: {trans_rule.get('exact_pattern', 'N/A')}
- Exact spacing: {trans_rule.get('exact_spacing', 'N/A')}
- Exact positions: {trans_rule.get('exact_positions', 'N/A')}

Application Logic:
- Check properties: {app_logic.get('check_properties', 'N/A')}
- Verify conditions: {app_logic.get('verify_conditions', 'N/A')}
- Apply rule: {app_logic.get('apply_rule', 'N/A')}

CRITICAL: Apply this EXACT transition rule based on the object's properties matching the determining properties.
"""
            
            # Method 1: Use training booklet steps (most direct - exact crop-transform-uncrop sequence)
            step_sequence_text = ""
            if training_booklet_steps:
                # Show the full crop-transform-uncrop sequence from training booklet
                crop_step = next((ts for ts in training_booklet_steps if ts['step_type'] == 'crop'), None)
                transform_step = next((ts for ts in training_booklet_steps if ts['step_type'] == 'transform'), None)
                uncrop_step = next((ts for ts in training_booklet_steps if ts['step_type'] == 'uncrop'), None)
                
                booklet_idx = training_booklet_steps[0]['booklet_idx'] if training_booklet_steps else 'N/A'
                
                step_sequence_text = f"""

TRAINING BOOKLET STEP SEQUENCE (follow EXACT crop-transform-uncrop sequence from booklet {booklet_idx + 1}):
"""
                if crop_step:
                    step_sequence_text += f"""
CROP STEP (Step {crop_step['step_num']}):
  Training: {crop_step['description']}
  Action: Crop to object bounding box (same as training)
"""
                if transform_step:
                    step_sequence_text += f"""
TRANSFORM STEP (Step {transform_step['step_num']}):
  Training: {transform_step['description']}
  Action: Apply the SAME transformation pattern from training, adapted to test object colors/positions
  
CRITICAL: This is the EXACT transform step from training. Follow the same transformation:
- Same color transformation pattern (e.g., color 2 → color 4)
- Same shape transformation pattern
- Same positioning relative to reference objects
- Adapt only the specific colors/positions to match test input
"""
                if uncrop_step:
                    step_sequence_text += f"""
UNCROP STEP (Step {uncrop_step['step_num']}):
  Training: {uncrop_step['description']}
  Action: Place transformed object back into full grid (same as training)
"""
                step_sequence_text += f"""
CRITICAL: Follow this EXACT crop-transform-uncrop sequence from the training booklet.
Each step must match the training pattern but adapted to the test input.
"""
            
            # Method 2: Use mapped object step sequence (fallback)
            elif object_step_sequence:
                # Find transform step in sequence
                transform_step = None
                for seq_step in object_step_sequence:
                    if seq_step.get('step_type') == 'transform':
                        transform_step = seq_step
                        break
                
                if transform_step:
                    step_sequence_text = f"""

MAPPED TRAINING STEP (follow this exact step from training booklet):
Training Description: {transform_step.get('training_description', 'N/A')}
Adaptation: {transform_step.get('adaptation', 'N/A')}
Applies To: {transform_step.get('applies_to', 'N/A')}

CRITICAL ADAPTATION PROCESS:
1. Take this EXACT training step (exact colors, spacing, patterns)
2. Use TEST REFERENCE OBJECTS to adapt:
   - Map training reference positions → test reference positions
   - Map training reference colors → test reference colors
   - Measure spacing relative to test reference objects
3. Apply the ADAPTED step to transform the test object
4. Maintain the same transformation pattern, just adapted to test input
"""
            
            # Method 2: Use corresponding training step (direct step match)
            corresponding_step_text = ""
            if corresponding_training_step:
                training_step = corresponding_training_step.get('training_step', {})
                step_desc = corresponding_training_step.get('description', 'N/A')
                step_num = corresponding_training_step.get('step_num', 'N/A')
                booklet_idx = corresponding_training_step.get('booklet_idx', 'N/A')
                
                corresponding_step_text = f"""

CORRESPONDING TRAINING STEP (exact step from training booklet):
Training Step {step_num} from Booklet {booklet_idx + 1}:
{step_desc}

CRITICAL ADAPTATION PROCESS:
1. This is the EXACT step from training booklets
2. Use TEST REFERENCE OBJECTS to adapt this step:
   - Map training reference positions → test reference positions
   - Map training reference colors → test reference colors
   - Measure spacing relative to test reference objects
3. Apply the ADAPTED step to transform the test object
4. Maintain the same transformation pattern, just adapted to test input
"""
            
            # Fallback: Use generalized step sequence
            if not step_sequence_text and generalized_step_sequence:
                # Find relevant steps for this object
                relevant_steps = []
                for seq_step in generalized_step_sequence:
                    step_type = seq_step.get('step_type', '')
                    if step_type == 'transform':
                        applies_to = seq_step.get('applies_to', '').lower()
                        obj_desc = f"object {object_num}".lower()
                        # Check if step applies to this object type
                        if not applies_to or 'object' in applies_to or step_type == 'transform':
                            relevant_steps.append(seq_step)
                
                if relevant_steps:
                    # Include specific details from steps
                    specific_details = ""
                    for step in relevant_steps[:3]:
                        if step.get('exact_colors'):
                            specific_details += f"\n  Step {step.get('step_num')} colors: {step.get('exact_colors')}"
                        if step.get('exact_spacing'):
                            specific_details += f"\n  Step {step.get('step_num')} spacing: {step.get('exact_spacing')}"
                        if step.get('exact_positions'):
                            specific_details += f"\n  Step {step.get('step_num')} positions: {step.get('exact_positions')}"
                        if step.get('exact_pattern'):
                            specific_details += f"\n  Step {step.get('step_num')} pattern: {step.get('exact_pattern')}"
                    
                    step_sequence_text = f"""

GENERALIZED STEP SEQUENCE (from training booklets - adapt using test reference objects):
{json.dumps(relevant_steps[:3], indent=2)}
{specific_details}

CRITICAL ADAPTATION PROCESS:
1. Take these EXACT steps from training booklets (exact colors, spacing, patterns)
2. Use TEST REFERENCE OBJECTS to adapt:
   - Map training reference positions → test reference positions
   - Map training reference colors → test reference colors
   - Measure spacing relative to test reference objects
   - Apply patterns centered/aligned to test reference objects
3. Apply the ADAPTED steps to transform the test object
4. Maintain the same transformation pattern, just adapted to test input

Example adaptation:
- Training: "Place object 2 cells right of reference (at position X)"
- Test: "Place object 2 cells right of TEST reference (at its position Y)"
- Same spacing pattern, adapted to test reference position
"""
            
            training_text, training_images = self._format_training_examples(all_training_examples or [], 3)
            
            # Build adaptation guidance using reference objects
            adaptation_guidance = ""
            if reference_info and is_test:
                ref_obj = reference_info.get('object', {})
                how_adapts = reference_info.get('how_adapts_steps', '')
                part_used = reference_info.get('part_used', {})
                
                adaptation_guidance = f"""

ADAPTATION USING TEST REFERENCE OBJECTS:
The GENERALIZED RULE from training booklets must be ADAPTED using the reference objects found in the test input.

Reference Object: {ref_obj.get('description', 'N/A')}
How it adapts steps: {how_adapts if how_adapts else 'Use as anchor for positioning and pattern adaptation'}
Part used: {part_used.get('part_name', 'full object') if part_used else 'full object'}

CRITICAL ADAPTATION PROCESS:
1. Take the GENERALIZED RULE from training booklets (exact colors, spacing, patterns)
2. Use the TEST REFERENCE OBJECT to adapt:
   - Colors: Map training colors to test reference object colors
   - Positions: Use test reference object position as anchor (instead of training reference position)
   - Spacing: Measure spacing relative to test reference object (instead of training reference)
   - Patterns: Apply same pattern but centered/aligned to test reference object
3. Apply the adapted rule to transform the current object

Example:
- Training rule: "Place object 2 cells to the right of reference object"
- Test adaptation: "Place object 2 cells to the right of the TEST reference object (at its current position)"
- Training rule: "Use color 2 from reference object"
- Test adaptation: "Use the same color as the TEST reference object (which may be color X in test)"
"""
            
            prompt = f"""Transform object {object_num}/{total_objects} (TEST MODE)
{ref_text}
{rule_text}
{pattern_text}
{transition_determinant_text}
{corresponding_step_text}
{step_sequence_text}
{adaptation_guidance}
CROPPED INPUT (current object to transform):
{self._format_grid(cropped_input)}

ANALYSIS TASK:
1. Identify the GENERALIZED RULE from training booklets (exact colors, spacing, patterns)
2. Use TEST REFERENCE OBJECTS to ADAPT this rule:
   - Map training reference positions → test reference positions
   - Map training reference colors → test reference colors  
   - Measure spacing relative to test reference objects
   - Apply patterns centered/aligned to test reference objects
3. Apply the ADAPTED rule to transform the current object
4. Ensure the transformation follows the same pattern as training, but adapted to test input

ADAPTATION PROCESS:
- Start with: Generalized rule from training booklets
- Adapt using: Test reference objects (positions, colors, spacing)
- Result: Rule adapted to test input, maintaining the same transformation pattern

TRAINING EXAMPLES (showing the pattern to adapt):
{training_text}

Generate transformation by:
- Taking the GENERALIZED RULE from training booklets
- ADAPTING it using the TEST REFERENCE OBJECTS (positions, colors, spacing)
- Applying the ADAPTED rule to transform this object
- Maintaining the same transformation pattern as training, but adapted to test input
{self._get_base_instructions()}
- Grid size: {grid_size}
- CRITICAL: Adapt the training rule using test reference objects, then apply the adapted rule"""
            
            images = [grid_to_image(cropped_input, 50)]
            images.extend(training_images)
        else:
            # Training mode
            # Add mini-analysis to prompt if available
            mini_analysis_text = ""
            if transformation_analysis:
                reasoning = transformation_analysis.get("reasoning", "")
                conditions = transformation_analysis.get("conditions", [])
                pattern_relation = transformation_analysis.get("pattern_relation", "")
                recommended_transforms = transformation_analysis.get("recommended_transformations", [])
                
                conditions_text = "; ".join(conditions) if conditions else "No specific conditions"
                recommended_text = ", ".join(recommended_transforms) if recommended_transforms else "unknown"
                
                mini_analysis_text = f"""

TRANSFORMATION ANALYSIS (Why this transformation is needed):
{reasoning}

CONDITIONS BEING FULFILLED:
{conditions_text}

PATTERN RELATION:
{pattern_relation}

RECOMMENDED TRANSFORMATION TYPES:
{recommended_text}

"""
            
            prompt = f"""Transform object {object_num}/{total_objects}
{ref_text}

CROPPED INPUT:
{self._format_grid(cropped_input)}

CROPPED OUTPUT (target):
{self._format_grid(cropped_output)}
{mini_analysis_text}

Generate transformation to convert input to match output.
{self._get_base_instructions()}
- Grid size: {grid_size}
- Attempt {attempt_num}/3"""
            
            images = [grid_to_image(cropped_input, 50)]
            if cropped_output:
                images.append(grid_to_image(cropped_output, 50))
        
        description, grid = self.call_api(prompt, images, cropped_input, None, None)
        return description or "", grid
    
    # ==================== BOOKLET ANALYSIS ====================
    
    def load_training_booklets(self, puzzle_id: str, num_training: int) -> List[Dict]:
        """Load all training booklets (step-by-step traces)"""
        booklets = []
        for i in range(1, num_training + 1):
            booklet_path = Path("visual_step_results") / puzzle_id / f"training_{i:02d}" / "results.json"
            if booklet_path.exists():
                try:
                    with open(booklet_path, 'r', encoding='utf-8') as f:
                        booklet_data = json.load(f)
                        booklets.append({
                            'training_num': i,
                            'steps': booklet_data.get('steps', []),
                            'input': booklet_data.get('input_grid'),
                            'output': booklet_data.get('expected_output_grid')
                        })
                        print(f"  ✓ Loaded booklet for training_{i:02d} ({len(booklet_data.get('steps', []))} steps)")
                except Exception as e:
                    print(f"  ⚠️ Failed to load booklet for training_{i:02d}: {e}")
            else:
                print(f"  ⚠️ Booklet not found for training_{i:02d}")
        return booklets
    
    def analyze_transformation_rule(self, all_training_examples: List[Dict]) -> Dict:
        """Analyze the deep transformation rule: what about input gives output, how colors/shapes are used"""
        print(f"\n{'='*80}")
        print("ANALYZING TRANSFORMATION RULE (COLORS, SHAPES, PATTERNS)")
        print(f"{'='*80}\n")
        
        # Build comprehensive analysis prompt
        examples_text = ""
        example_images = []
        
        for i, example in enumerate(all_training_examples):
            input_grid = example['input']
            output_grid = example['output']
            
            examples_text += f"\n{'='*60}\n"
            examples_text += f"TRAINING EXAMPLE {i+1}:\n"
            examples_text += f"{'='*60}\n"
            examples_text += f"Input ({len(input_grid)}×{len(input_grid[0])}):\n{self._format_grid(input_grid)}\n"
            examples_text += f"Output ({len(output_grid)}×{len(output_grid[0])}):\n{self._format_grid(output_grid)}\n"
            
            example_images.append(grid_to_image(input_grid, 50))
            example_images.append(grid_to_image(output_grid, 50))
        
        prompt = f"""Analyze ALL training examples to extract the TRANSFORMATION RULE.

{examples_text}

CRITICAL ANALYSIS TASKS:

1. **What is the RULE?** (What about the input determines the output?)
   - What property/pattern in the input causes the transformation?
   - Is it position-based? Color-based? Shape-based? Size-based?
   - What is the relationship between input and output?

2. **How are COLORS used?**
   - Which colors in input map to which colors in output?
   - Are colors preserved, changed, or used as signals?
   - Do specific colors trigger specific transformations?
   - Are colors used to identify reference objects or patterns?

3. **How are SHAPES used?**
   - Which shapes/patterns in input determine output shapes?
   - Are shapes preserved, transformed, or replicated?
   - Do specific shapes trigger specific transformations?
   - How do shapes relate to the overall pattern?

4. **What is the TRANSFORMATION PROCESS?**
   - Step-by-step: what happens to each element?
   - What gets copied? What gets transformed? What gets created?
   - What is the order of operations?
   - What conditions determine each transformation?

5. **GENERALIZATION: How does this apply to ANY input?**
   - What abstract rule can be extracted?
   - What properties must be checked in a new input?
   - How can the rule be applied to a test input?

Return JSON:
{{
  "rule_description": "clear statement of the transformation rule",
  "input_properties": {{
    "colors": {{
      "color_X": "what this color means/does in input (X is numeric value)",
      "color_mapping": "how input colors map to output colors"
    }},
    "shapes": {{
      "shape_patterns": "what shapes/patterns in input are significant",
      "shape_transformations": "how shapes are transformed"
    }},
    "positions": {{
      "position_rules": "how position affects transformation",
      "spatial_relationships": "relationships between objects"
    }},
    "triggers": [
      {{
        "condition": "what condition triggers this transformation",
        "action": "what transformation happens",
        "applies_to": "what objects/elements this applies to"
      }}
    ]
  }},
  "transformation_process": {{
    "step_by_step": [
      {{
        "step": 1,
        "action": "what happens in this step",
        "checks": "what properties are checked",
        "transforms": "what gets transformed"
      }}
    ],
    "order": "order of operations",
    "conditions": "conditional logic"
  }},
  "generalization": {{
    "abstract_rule": "abstract rule that applies to any input",
    "properties_to_check": ["list of properties to check in new input"],
    "how_to_apply": "how to apply this rule to a test input",
    "color_logic": "how colors guide the transformation",
    "shape_logic": "how shapes guide the transformation"
  }},
  "application_guide": {{
    "for_test_input": "specific guide for applying to test input",
    "check_first": "what to check first",
    "then_apply": "what to apply then"
  }}
}}

Focus on extracting a RULE that can be applied to ANY input, not just these specific examples."""
        
        description, _ = self.call_api(prompt, example_images)
        
        response_text = description if description else ""
        
        # Try multiple JSON extraction strategies
        rule_data = None
        
        try:
            import re
            json_patterns = [
                r'\{[\s\S]*?\}',
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
                r'```json\s*(\{[\s\S]*?\})\s*```',
                r'```\s*(\{[\s\S]*?\})\s*```',
            ]
            
            for pattern in json_patterns:
                matches = re.finditer(pattern, response_text, re.DOTALL)
                for match in matches:
                    try:
                        json_str = match.group(1) if match.groups() else match.group(0)
                        json_str = json_str.strip()
                        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        rule_data = json.loads(json_str)
                        if isinstance(rule_data, dict):
                            break
                    except:
                        continue
                if rule_data:
                    break
            
            if not rule_data:
                cleaned = re.sub(r'```[a-z]*\n?', '', response_text)
                cleaned = re.sub(r'```', '', cleaned)
                cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
                rule_data = json.loads(cleaned)
            
            if rule_data and isinstance(rule_data, dict):
                print(f"\n✓ Rule analysis complete:")
                print(f"  Rule: {rule_data.get('rule_description', 'N/A')[:100]}...")
                if rule_data.get('generalization'):
                    gen = rule_data['generalization']
                    print(f"  Abstract rule: {gen.get('abstract_rule', 'N/A')[:100]}...")
                return rule_data
        except Exception as e:
            print(f"  ⚠️ Parse failed: {e}")
            if response_text:
                print(f"  Response preview: {response_text[:300]}...")
        
        print("  ⚠️ Using fallback rule")
        return {
            "rule_description": "Rule extraction failed",
            "generalization": {
                "abstract_rule": "Apply same transformation pattern as training examples"
            }
        }
    
    def analyze_transition_determinants(self, all_training_examples: List[Dict],
                                       training_booklets: List[Dict]) -> Dict:
        """Analyze what determines transitions in training examples and how to apply to test input"""
        print(f"\n{'='*80}")
        print("ANALYZING TRANSITION DETERMINANTS (What causes transitions?)")
        print(f"{'='*80}\n")
        
        # Build comprehensive analysis
        examples_text = ""
        example_images = []
        
        for i, example in enumerate(all_training_examples):
            input_grid = example['input']
            output_grid = example['output']
            
            examples_text += f"\n{'='*60}\n"
            examples_text += f"TRAINING EXAMPLE {i+1}:\n"
            examples_text += f"{'='*60}\n"
            examples_text += f"Input ({len(input_grid)}×{len(input_grid[0])}):\n{self._format_grid(input_grid)}\n"
            examples_text += f"Output ({len(output_grid)}×{len(output_grid[0])}):\n{self._format_grid(output_grid)}\n"
            
            example_images.append(grid_to_image(input_grid, 40))
            example_images.append(grid_to_image(output_grid, 40))
        
        # Include booklet step sequences
        booklets_text = ""
        for i, booklet in enumerate(training_booklets):
            steps = booklet.get('steps', [])
            booklets_text += f"\nTRAINING EXAMPLE {i+1} STEPS ({len(steps)} steps):\n"
            for step in steps[:10]:  # First 10 steps
                step_type = ""
                if step.get('is_reference_step'):
                    step_type = "REFERENCE"
                elif step.get('is_crop_step'):
                    step_type = "CROP"
                elif step.get('is_cropped_view'):
                    step_type = "TRANSFORM"
                elif step.get('is_uncrop_step'):
                    step_type = "UNCROP"
                booklets_text += f"  Step {step.get('step_num', 'N/A')} [{step_type}]: {step.get('description', 'N/A')[:80]}...\n"
        
        prompt = f"""Analyze ALL training examples to determine WHAT PROPERTIES/CONDITIONS determine the transitions.

{examples_text}

STEP SEQUENCES FROM BOOKLETS:
{booklets_text}

CRITICAL ANALYSIS TASK: Determine the DECISION LOGIC for transitions.

For EACH type of transition that occurs, identify:

1. **What INPUT PROPERTIES determine this transition?**
   - What colors trigger it? (e.g., "if object is color 2")
   - What shapes trigger it? (e.g., "if object is vertical line")
   - What positions trigger it? (e.g., "if object is in top-left")
   - What relationships trigger it? (e.g., "if object is near reference object")
   - What sizes trigger it? (e.g., "if object is 3 cells")

2. **What CONDITIONS must be met?**
   - Exact conditions that must be true for transition to occur
   - Dependencies (e.g., "only if reference object exists")
   - Ordering requirements (e.g., "after reference object is identified")

3. **What is the TRANSITION RULE?**
   - Exact transformation (e.g., "color 2 → color 4")
   - Exact pattern (e.g., "replicate in 3×3 grid")
   - Exact spacing (e.g., "place 2 cells apart")

4. **How to APPLY to test input?**
   - How to check if test input has the determining properties
   - How to verify conditions are met
   - How to apply the transition rule

Return JSON:
{{
  "transition_determinants": [
    {{
      "transition_type": "color_change|replication|movement|creation|removal",
      "determining_properties": {{
        "colors": ["color X", "color Y"],  // what colors trigger this
        "shapes": ["vertical line", "U-shape"],  // what shapes trigger this
        "positions": "where objects must be (e.g., 'top-left', 'near reference')",
        "sizes": "what sizes trigger this (e.g., 'size 2-3 cells')",
        "relationships": "relationships that trigger this (e.g., 'near color 8 object')"
      }},
      "conditions": [
        "condition 1 that must be met",
        "condition 2 that must be met"
      ],
      "transition_rule": {{
        "exact_transformation": "exact transformation (e.g., 'color 2 → color 4')",
        "exact_pattern": "exact pattern (e.g., 'replicate in 3×3 grid with 2-cell spacing')",
        "exact_spacing": "exact spacing (e.g., '2 cells between replicas')",
        "exact_positions": "exact positions (e.g., 'aligned to reference object at offset (2,3)')"
      }},
      "application_logic": {{
        "check_properties": "how to check if test input has determining properties",
        "verify_conditions": "how to verify conditions are met",
        "apply_rule": "how to apply transition rule to test input"
      }}
    }}
  ],
  "decision_tree": "overall decision logic: what to check first, then what, etc.",
  "application_guide": "step-by-step guide for applying to test input"
}}

Focus on extracting the EXACT properties and conditions that determine transitions, so they can be checked and applied to any test input."""
        
        description, _ = self.call_api(prompt, example_images)
        
        response_text = description if description else ""
        determinants_data = {}
        
        try:
            import re
            json_patterns = [
                r'\{[\s\S]*?\}',
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
                r'```json\s*(\{[\s\S]*?\})\s*```',
                r'```\s*(\{[\s\S]*?\})\s*```',
            ]
            
            for pattern in json_patterns:
                matches = re.finditer(pattern, response_text, re.DOTALL)
                for match in matches:
                    try:
                        json_str = match.group(1) if match.groups() else match.group(0)
                        json_str = json_str.strip()
                        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        determinants_data = json.loads(json_str)
                        if isinstance(determinants_data, dict):
                            transitions = determinants_data.get('transition_determinants', [])
                            print(f"\n✓ Transition determinants analysis complete:")
                            print(f"  Found {len(transitions)} transition types")
                            for trans in transitions[:3]:
                                trans_type = trans.get('transition_type', 'N/A')
                                props = trans.get('determining_properties', {})
                                colors = props.get('colors', [])
                                print(f"    {trans_type}: triggered by {', '.join(colors) if colors else 'various properties'}")
                            return determinants_data
                    except:
                        continue
                if determinants_data:
                    break
        except Exception as e:
            print(f"  ⚠️ Parse failed: {e}")
            if response_text:
                print(f"  Response preview: {response_text[:300]}...")
        
        print("  ⚠️ Transition determinants analysis failed")
        return {
            "transition_determinants": [],
            "decision_tree": "Check input properties and apply transformation rule",
            "application_guide": "Apply transformation rule based on input properties"
        }
    
    def generate_generalized_step_sequence(self, training_booklets: List[Dict], 
                                          transformation_rule: Optional[Dict] = None) -> List[Dict]:
        """Generate a generalized step-by-step sequence from training booklets"""
        print(f"\n{'='*80}")
        print("GENERATING GENERALIZED STEP SEQUENCE FROM TRAINING BOOKLETS")
        print(f"{'='*80}\n")
        
        if not training_booklets:
            print("  ⚠️ No training booklets available")
            return []
        
        # Build comprehensive prompt with all booklets
        booklets_text = ""
        booklet_images = []
        
        for i, booklet in enumerate(training_booklets):
            steps = booklet.get('steps', [])
            input_grid = booklet.get('input')
            output_grid = booklet.get('output')
            
            booklets_text += f"\n{'='*60}\n"
            booklets_text += f"TRAINING EXAMPLE {i+1} BOOKLET ({len(steps)} steps):\n"
            booklets_text += f"{'='*60}\n"
            
            if input_grid:
                booklets_text += f"Input Grid ({len(input_grid)}×{len(input_grid[0])}):\n{self._format_grid(input_grid)}\n"
                booklet_images.append(grid_to_image(input_grid, 40))
            
            if output_grid:
                booklets_text += f"Output Grid ({len(output_grid)}×{len(output_grid[0])}):\n{self._format_grid(output_grid)}\n"
                booklet_images.append(grid_to_image(output_grid, 40))
            
            # Describe each step in detail
            booklets_text += f"\nStep Sequence:\n"
            for step_idx, step in enumerate(steps):
                step_type = "INITIAL" if step_idx == 0 else ""
                if step.get('is_reference_step'):
                    step_type = "REFERENCE OBJECT"
                elif step.get('is_crop_step'):
                    step_type = "CROP"
                elif step.get('is_cropped_view'):
                    step_type = "TRANSFORM"
                elif step.get('is_uncrop_step'):
                    step_type = "UNCROP"
                elif step.get('is_removal_step'):
                    step_type = "REMOVE"
                elif step.get('is_new_object'):
                    step_type = "NEW OBJECT"
                elif step.get('is_final_step'):
                    step_type = "FINAL"
                
                desc = step.get('description', 'N/A')
                obj_num = step.get('object_num')
                if obj_num:
                    desc = f"Object {obj_num}: {desc}"
                
                booklets_text += f"  Step {step.get('step_num', step_idx+1)} [{step_type}]: {desc}\n"
                
                # Include object info if available
                if step.get('object_description'):
                    booklets_text += f"    Object: {step.get('object_description')}\n"
                if step.get('reference_reasoning'):
                    booklets_text += f"    Reference reasoning: {step.get('reference_reasoning')[:80]}...\n"
        
        rule_text = ""
        if transformation_rule:
            rule_desc = transformation_rule.get('rule_description', 'N/A')
            gen = transformation_rule.get('generalization', {})
            abstract_rule = gen.get('abstract_rule', 'N/A')
            rule_text = f"""
TRANSFORMATION RULE:
Rule: {rule_desc}
Abstract Rule: {abstract_rule}
"""
        
        prompt = f"""Generate a GENERALIZED STEP-BY-STEP SEQUENCE from all training booklets.

{booklets_text}
{rule_text}

CRITICAL TASK: Extract the MOST SPECIFIC details that are CONSISTENT across ALL training booklets.

The sequence must be:
1. **Highly Specific**: Include exact colors (color 1, color 2, etc.), spacing, positions, patterns
2. **Consistent Across All**: Only include details that appear in ALL training examples
3. **Generalizable**: Apply to any test input by checking for the same specific patterns

ANALYSIS APPROACH:
- Compare ALL training booklets side-by-side
- Find what is IDENTICAL across all (colors, spacing, patterns, positions)
- Extract the EXACT specifications that are consistent
- Generalize only where necessary, but keep maximum specificity

For each step, specify EXACT details:
- Step type (reference, crop, transform, uncrop, remove, new_object, final)
- Exact colors involved (e.g., "color 2", "color 4", "color 2 → color 4")
- Exact spacing/positions (e.g., "2 cells apart", "aligned to row 5")
- Exact patterns (e.g., "3×3 grid", "every other cell", "diagonal pattern")
- Exact conditions (when this step applies - be specific)
- How to adapt (what specific properties to check in test input)

Return JSON:
{{
  "generalized_steps": [
    {{
      "step_num": 1,
      "step_type": "reference|crop|transform|uncrop|remove|new_object|final",
      "description": "highly specific description with exact colors, spacing, patterns",
      "exact_colors": ["color X", "color Y"],  // exact color numbers used
      "exact_spacing": "exact spacing pattern (e.g., '2 cells between objects')",
      "exact_positions": "exact position pattern (e.g., 'aligned to reference object at offset (2,3)')",
      "exact_pattern": "exact pattern description (e.g., '3×3 grid', 'every 2nd row')",
      "conditions": "specific conditions when this step applies",
      "adaptation": "how to check and adapt these exact specifications to test input",
      "applies_to": "what objects/elements this applies to (be specific about colors/shapes)"
    }}
  ],
  "execution_order": "order of steps",
  "specific_patterns": {{
    "color_patterns": "exact color patterns consistent across all training examples",
    "spacing_patterns": "exact spacing patterns consistent across all",
    "position_patterns": "exact position patterns consistent across all"
  }},
  "adaptation_guide": "how to check these exact specifications in test input and adapt"
}}

CRITICAL: Be as SPECIFIC as possible (exact colors, spacing, positions) while ensuring the pattern is CONSISTENT across ALL training booklets."""
        
        description, _ = self.call_api(prompt, booklet_images)
        
        response_text = description if description else ""
        step_sequence = []
        
        try:
            import re
            json_patterns = [
                r'\{[\s\S]*?\}',
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
                r'```json\s*(\{[\s\S]*?\})\s*```',
                r'```\s*(\{[\s\S]*?\})\s*```',
            ]
            
            for pattern in json_patterns:
                matches = re.finditer(pattern, response_text, re.DOTALL)
                for match in matches:
                    try:
                        json_str = match.group(1) if match.groups() else match.group(0)
                        json_str = json_str.strip()
                        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        seq_data = json.loads(json_str)
                        if isinstance(seq_data, dict):
                            step_sequence = seq_data.get('generalized_steps', [])
                            if step_sequence:
                                print(f"\n✓ Generated {len(step_sequence)} generalized steps")
                                for i, step in enumerate(step_sequence[:5]):  # Show first 5
                                    print(f"  Step {step.get('step_num', i+1)} [{step.get('step_type', 'N/A')}]: {step.get('description', 'N/A')[:60]}...")
                                if len(step_sequence) > 5:
                                    print(f"  ... and {len(step_sequence) - 5} more steps")
                                return step_sequence
                    except:
                        continue
                if step_sequence:
                    break
        except Exception as e:
            print(f"  ⚠️ Parse failed: {e}")
            if response_text:
                print(f"  Response preview: {response_text[:300]}...")
        
        print("  ⚠️ Step sequence generation failed, using fallback")
        return []
    
    def map_training_steps_to_test_objects(self, training_booklets: List[Dict], 
                                           test_input_objects: List[Dict],
                                           test_input_grid: List[List[int]]) -> Dict[int, Dict]:
        """Method 1: Map each training booklet step to corresponding test object and step
        
        For each step in training booklets, identify:
        - What object it applies to in training
        - Which test object corresponds to it
        - How to adapt the step to the test object
        """
        print(f"\n{'='*80}")
        print("MAPPING TRAINING STEPS TO TEST OBJECTS")
        print(f"{'='*80}\n")
        
        # Build mapping: test_object_idx -> training step sequence
        object_step_mapping = {}
        
        # Analyze all training booklets to find step patterns
        training_steps_by_object = {}  # training_obj_description -> list of steps
        
        for booklet_idx, booklet in enumerate(training_booklets):
            steps = booklet.get('steps', [])
            input_grid = booklet.get('input', [])
            
            # Extract object processing sequence from steps
            for step in steps:
                if step.get('object_num') and step.get('object_description'):
                    obj_desc = step.get('object_description', '')
                    step_type = ""
                    if step.get('is_reference_step'):
                        step_type = "reference"
                    elif step.get('is_crop_step'):
                        step_type = "crop"
                    elif step.get('is_cropped_view'):
                        step_type = "transform"
                    elif step.get('is_uncrop_step'):
                        step_type = "uncrop"
                    
                    if obj_desc not in training_steps_by_object:
                        training_steps_by_object[obj_desc] = []
                    
                    training_steps_by_object[obj_desc].append({
                        'step_type': step_type,
                        'step_num': step.get('step_num', 0),
                        'description': step.get('description', ''),
                        'booklet_idx': booklet_idx,
                        'object_num': step.get('object_num')
                    })
        
        # For each test object, find matching training object and its step sequence
        test_objects_text = ""
        for i, test_obj in enumerate(test_input_objects):
            test_objects_text += f"Test Object {i+1}: {test_obj.get('description', 'N/A')}, colors: {test_obj.get('colors', [])}\n"
        
        training_patterns_text = ""
        for obj_desc, steps in training_steps_by_object.items():
            training_patterns_text += f"\nTraining Object Pattern: {obj_desc}\n"
            training_patterns_text += f"  Steps: {len(steps)}\n"
            for step in steps[:5]:  # First 5 steps
                training_patterns_text += f"    Step {step['step_num']} [{step['step_type']}]: {step['description'][:60]}...\n"
        
        prompt = f"""Map each TEST OBJECT to the corresponding TRAINING OBJECT and its step sequence.

TEST INPUT OBJECTS:
{test_objects_text}

TRAINING OBJECT STEP PATTERNS:
{training_patterns_text}

TASK: For each test object, find which training object pattern it matches, and map the exact step sequence.

Return JSON:
{{
  "object_mappings": [
    {{
      "test_object_idx": 0,
      "matching_training_pattern": "description of matching training object",
      "step_sequence": [
        {{
          "step_type": "reference|crop|transform|uncrop",
          "step_order": 1,
          "training_description": "what this step does in training",
          "adaptation": "how to adapt this step to test object",
          "applies_to": "which test object this applies to"
        }}
      ],
      "reasoning": "why this test object matches this training pattern"
    }}
  ]
}}

Focus on matching objects by their properties (colors, shapes) and mapping the EXACT step sequence."""
        
        images = [grid_to_image(test_input_grid, 50)]
        for booklet in training_booklets:
            if booklet.get('input'):
                images.append(grid_to_image(booklet['input'], 40))
        
        description, _ = self.call_api(prompt, images)
        
        response_text = description if description else ""
        mapping_data = {}
        
        try:
            import re
            json_patterns = [
                r'\{[\s\S]*?\}',
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
                r'```json\s*(\{[\s\S]*?\})\s*```',
                r'```\s*(\{[\s\S]*?\})\s*```',
            ]
            
            for pattern in json_patterns:
                matches = re.finditer(pattern, response_text, re.DOTALL)
                for match in matches:
                    try:
                        json_str = match.group(1) if match.groups() else match.group(0)
                        json_str = json_str.strip()
                        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        mapping_data = json.loads(json_str)
                        if isinstance(mapping_data, dict):
                            mappings = mapping_data.get('object_mappings', [])
                            print(f"\n✓ Mapped {len(mappings)} test objects to training step sequences")
                            for mapping in mappings[:3]:
                                test_idx = mapping.get('test_object_idx', 'N/A')
                                pattern = mapping.get('matching_training_pattern', 'N/A')[:50]
                                steps = mapping.get('step_sequence', [])
                                print(f"    Test object {test_idx}: {pattern}... ({len(steps)} steps)")
                            return mapping_data
                    except:
                        continue
                if mapping_data:
                    break
        except Exception as e:
            print(f"  ⚠️ Parse failed: {e}")
        
        print("  ⚠️ Step mapping failed")
        return {"object_mappings": []}
    
    def find_corresponding_training_step(self, test_obj: Dict, test_obj_idx: int,
                                        current_step_type: str,
                                        training_booklets: List[Dict]) -> Optional[Dict]:
        """Method 2: For a specific test object and step type, find the corresponding training step
        
        This finds the exact training step that corresponds to the current test step,
        allowing us to follow the training booklet step-by-step.
        """
        # Find matching training object and its step sequence
        test_obj_desc = test_obj.get('description', '').lower()
        test_obj_colors = test_obj.get('colors', [])
        
        # Search through training booklets for matching object and step
        for booklet_idx, booklet in enumerate(training_booklets):
            steps = booklet.get('steps', [])
            
            for step in steps:
                # Check if step type matches
                step_type_match = False
                if current_step_type == 'crop' and step.get('is_crop_step'):
                    step_type_match = True
                elif current_step_type == 'transform' and step.get('is_cropped_view'):
                    step_type_match = True
                elif current_step_type == 'uncrop' and step.get('is_uncrop_step'):
                    step_type_match = True
                elif current_step_type == 'reference' and step.get('is_reference_step'):
                    step_type_match = True
                
                if step_type_match:
                    # Check if object matches
                    step_obj_desc = step.get('object_description', '').lower()
                    step_obj_num = step.get('object_num', 0)
                    
                    # Simple matching: check if descriptions are similar
                    desc_similarity = sum(1 for word in test_obj_desc.split() 
                                         if word in step_obj_desc.split()) / max(len(test_obj_desc.split()), 1)
                    
                    if desc_similarity > 0.3:  # At least 30% word overlap
                        return {
                            'training_step': step,
                            'booklet_idx': booklet_idx,
                            'step_num': step.get('step_num', 0),
                            'description': step.get('description', ''),
                            'object_num': step_obj_num,
                            'match_confidence': desc_similarity
                        }
        
        return None
    
    def analyze_booklet_pattern(self, training_booklets: List[Dict]) -> Dict:
        """Analyze all training booklets to extract the step-by-step generalization pattern"""
        print(f"\n{'='*80}")
        print("ANALYZING TRAINING BOOKLETS TO EXTRACT STEP-BY-STEP PATTERN")
        print(f"{'='*80}\n")
        
        if not training_booklets:
            print("  ⚠️ No training booklets available")
            return {}
        
        # Build comprehensive prompt with all booklets
        booklets_text = ""
        booklet_images = []
        
        for i, booklet in enumerate(training_booklets):
            steps = booklet.get('steps', [])
            input_grid = booklet.get('input')
            output_grid = booklet.get('output')
            
            booklets_text += f"\n{'='*60}\n"
            booklets_text += f"TRAINING EXAMPLE {i+1} BOOKLET ({len(steps)} steps):\n"
            booklets_text += f"{'='*60}\n"
            
            if input_grid:
                booklets_text += f"Input Grid ({len(input_grid)}×{len(input_grid[0])}):\n{self._format_grid(input_grid)}\n"
                booklet_images.append(grid_to_image(input_grid, 40))
            
            if output_grid:
                booklets_text += f"Output Grid ({len(output_grid)}×{len(output_grid[0])}):\n{self._format_grid(output_grid)}\n"
                booklet_images.append(grid_to_image(output_grid, 40))
            
            # Describe each step
            booklets_text += f"\nStep Sequence:\n"
            for step_idx, step in enumerate(steps):
                step_type = "INITIAL" if step_idx == 0 else ""
                if step.get('is_reference_step'):
                    step_type = "REFERENCE OBJECT"
                elif step.get('is_crop_step'):
                    step_type = "CROP"
                elif step.get('is_cropped_view'):
                    step_type = "TRANSFORM"
                elif step.get('is_uncrop_step'):
                    step_type = "UNCROP"
                elif step.get('is_removal_step'):
                    step_type = "REMOVE"
                elif step.get('is_new_object'):
                    step_type = "NEW OBJECT"
                elif step.get('is_final_step'):
                    step_type = "FINAL"
                
                desc = step.get('description', 'N/A')
                obj_num = step.get('object_num')
                if obj_num:
                    desc = f"Object {obj_num}: {desc}"
                
                booklets_text += f"  Step {step.get('step_num', step_idx+1)} [{step_type}]: {desc}\n"
        
        prompt = f"""Analyze ALL training booklets to extract the GENERALIZED STEP-BY-STEP PATTERN that can be applied to any test input.

{booklets_text}

TASK: Extract the abstract, generalizable pattern from these step-by-step booklets:

1. **Step Sequence Pattern**: What is the general sequence of step types?
   - Order: reference identification → crop → transform → uncrop → ...
   - Are there any conditional steps?
   - Are steps repeated? In what order?

2. **Object Processing Pattern**: How are objects processed?
   - Which objects are reference objects (constant)?
   - What order are objects processed in?
   - Are objects transformed, removed, or new ones created?

3. **Transformation Pattern**: What transformations are applied?
   - What types of transformations (color changes, moves, shape changes)?
   - How do transformations depend on reference objects?

4. **Generalization Rules**: What rules can be extracted that apply regardless of specific object positions/sizes?

Return JSON:
{{
  "step_sequence_pattern": [
    {{
      "step_type": "reference|crop|transform|uncrop|remove|new_object|final",
      "order": 1,
      "description": "what this step does in general",
      "conditions": "when this step applies (if any)"
    }}
  ],
  "object_processing_order": {{
    "reference_objects": "how reference objects are identified",
    "processing_order": "general order objects are processed",
    "removed_objects": "pattern for which objects are removed",
    "new_objects": "pattern for when/where new objects are created"
  }},
  "transformation_rules": [
    {{
      "rule": "general transformation rule",
      "applies_to": "what objects/conditions",
      "uses_reference": "how reference objects guide this"
    }}
  ],
  "generalization": "overall pattern that can be applied to any test input"
}}

Focus on extracting patterns that are GENERALIZABLE - rules that work for any input."""
        
        description, _ = self.call_api(prompt, booklet_images)
        
        response_text = description if description else ""
        
        # Try multiple JSON extraction strategies
        pattern_data = None
        
        # Strategy 1: Try to find JSON block with more lenient matching
        try:
            import re
            # Try to find JSON that starts with { and ends with }
            json_patterns = [
                r'\{[\s\S]*?\}',  # Greedy match
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested braces
                r'```json\s*(\{[\s\S]*?\})\s*```',  # Code block
                r'```\s*(\{[\s\S]*?\})\s*```',  # Generic code block
            ]
            
            for pattern in json_patterns:
                matches = re.finditer(pattern, response_text, re.DOTALL)
                for match in matches:
                    try:
                        json_str = match.group(1) if match.groups() else match.group(0)
                        # Clean up common JSON issues
                        json_str = json_str.strip()
                        # Remove trailing commas before closing braces/brackets
                        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        pattern_data = json.loads(json_str)
                        if isinstance(pattern_data, dict):
                            break
                    except:
                        continue
                if pattern_data:
                    break
        except Exception as e:
            pass
        
        # Strategy 2: Try to extract JSON from lines
        if not pattern_data:
            try:
                lines = response_text.split('\n')
                json_start = None
                json_end = None
                brace_count = 0
                
                for i, line in enumerate(lines):
                    if '{' in line and json_start is None:
                        json_start = i
                    if json_start is not None:
                        brace_count += line.count('{') - line.count('}')
                        if brace_count == 0 and json_start is not None:
                            json_end = i + 1
                            break
                
                if json_start is not None and json_end is not None:
                    json_lines = lines[json_start:json_end]
                    json_str = '\n'.join(json_lines)
                    # Clean up
                    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                    pattern_data = json.loads(json_str)
            except:
                pass
        
        # Strategy 3: Try to parse as-is with error recovery
        if not pattern_data:
            try:
                # Remove markdown code blocks if present
                cleaned = re.sub(r'```[a-z]*\n?', '', response_text)
                cleaned = re.sub(r'```', '', cleaned)
                # Remove trailing commas
                cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
                pattern_data = json.loads(cleaned)
            except:
                pass
        
        if pattern_data and isinstance(pattern_data, dict):
            print(f"\n✓ Pattern extraction complete:")
            print(f"  Step sequence: {len(pattern_data.get('step_sequence_pattern', []))} steps")
            print(f"  Transformation rules: {len(pattern_data.get('transformation_rules', []))}")
            if pattern_data.get('generalization'):
                print(f"  Generalization: {pattern_data.get('generalization', '')[:100]}...")
            return pattern_data
        
        print("  ⚠️ JSON parsing failed, using fallback pattern")
        print(f"  Response preview: {response_text[:200]}...")
        return {
            "step_sequence_pattern": [],
            "object_processing_order": {},
            "transformation_rules": [],
            "generalization": "Pattern extraction failed - using fallback"
        }
    
    # ==================== V7 GENERALIZED STEPS GENERATION ====================
    
    def generate_generalized_steps_v7(self, all_training_examples: List[Dict], 
                                     comprehensive_analysis: Dict,
                                     transformation_rule: Dict) -> List[Dict]:
        """Generate generalized steps with full tool awareness - pattern: for each (CONDITION, OBJECT) apply transformations"""
        print(f"\n{'='*80}")
        print("GENERATING GENERALIZED STEPS V7 (with full tool awareness)")
        print(f"{'='*80}\n")
        
        input_grids = [ex['input'] for ex in all_training_examples]
        output_grids = [ex['output'] for ex in all_training_examples]
        
        grids_text = ""
        for i, (in_grid, out_grid) in enumerate(zip(input_grids, output_grids)):
            grids_text += f"\nTRAINING {i+1}:\n"
            grids_text += f"Input: {self._format_grid(in_grid)}\n"
            grids_text += f"Output: {self._format_grid(out_grid)}\n"
        
        prompt = f"""Analyze ALL training examples to generate GENERALIZED STEPS.
Each step follows the pattern: "for each (CONDITION, OBJECT) apply 1 or more transformations"

{grids_text}

COMPREHENSIVE ANALYSIS CONTEXT:
{json.dumps(comprehensive_analysis, indent=2)}

TRANSFORMATION RULE:
{json.dumps(transformation_rule, indent=2) if transformation_rule else "N/A"}

CRITICAL: Generate steps that specify:
1. CONDITION: What properties/conditions identify objects to transform
   - Colors (e.g., "if object is color 2")
   - Shapes (e.g., "if object is vertical line", "if object is L-shaped")
   - Positions (e.g., "if object is in top-left", "if object is near reference")
   - Sizes (e.g., "if object is 3 cells")
   - Relationships (e.g., "if object is adjacent to color 8 object")
   - Countable properties (e.g., "if object has 2 holes", "if object height is 3")
   - Most common (e.g., "if object matches most common shape/color")

2. OBJECT: What type of object this applies to
   - Solid objects
   - Line objects (sequential, can change color by section, drawn in order)
   - Reference objects
   - Scaled objects (sized up/down versions)
   - Objects with specific properties

3. TRANSFORMATIONS: What transformations to apply (can be multiple):
   - Recolor (color X → color Y)
   - Reshape (rectangularize, make more standard shape)
   - Move (translate to new position)
   - Scale (size up/down)
   - Rotate (90°, 180°, 270°)
   - Reflect (on axis: horizontal, vertical, diagonal)
   - Repeat pattern (fill pattern, replicate)
   - Complete objects (recolor, reshape, move conditionally)
   - Fit objects together (find edges that counter each other)
   - Draw in object's color (lines vs objects)
   - Handle negative space (see cells common between parts, cells common to only one part)
   - Remove stray cells (difference between objects and stray cells)
   - Zoom in/out (including or excluding borders)
   - Center objects
   - Handle dimension/overlap (which object is frontmost)

FULL TOOL AWARENESS:
- Negative space: See cells common between parts of input, cells common to only one part
- Scaled objects: Awareness of sized up/down objects, when object is scaled version of another
- Lines: Different rules - sequential, starting/ending points, color changes by section, drawn in order
- Conditions: 1-to-1 correlations between shapes and colors, or colors and shapes
- Object completion: Recolor, reshape or move conditionally
- Object fitting: Finding edges that counter each other
- Drawing: Drawing in object's color, lines vs objects
- Line connections: Lines can connect objects (see output similarities)
- Dimension awareness: When objects overlap, which is frontmost
- Reshaping: Rectangularizing or making fuzzy-edged shapes into standard shapes
- Counting: Holes, cell length/height, multiple objects same shape/color count

Return JSON:
{{
  "generalized_steps": [
    {{
      "step_num": 1,
      "step_type": "transform|remove|new_object|reference",
      "condition": {{
        "description": "clear condition statement",
        "properties": {{
          "colors": ["color X", "color Y"],
          "shapes": ["vertical line", "L-shaped"],
          "positions": "where objects must be",
          "sizes": "what sizes trigger this",
          "relationships": "relationships that trigger this",
          "countable": "countable properties (holes, length, height)",
          "most_common": "if matches most common shape/color"
        }},
        "object_type": "solid_object|line_object|reference_object|scaled_object"
      }},
      "applies_to": "what objects this step applies to",
      "transformations": [
        {{
          "type": "recolor|reshape|move|scale|rotate|reflect|repeat_pattern|complete|fit|draw|handle_negative_space|remove_stray|zoom|center|handle_dimension",
          "description": "detailed transformation description",
          "parameters": {{
            "from_color": "color X",
            "to_color": "color Y",
            "rotation": "90|180|270",
            "reflection_axis": "horizontal|vertical|diagonal",
            "scale_factor": "2x|3x|0.5x",
            "pattern": "fill pattern description",
            "position": "where to move/place",
            "completion_type": "recolor|reshape|move",
            "fitting_type": "edge matching description",
            "drawing_type": "line|object color",
            "negative_space_type": "common_between|common_to_one",
            "zoom_type": "in|out",
            "include_borders": true|false
          }}
        }}
      ],
      "order": 1,
      "depends_on": ["step numbers this depends on"],
      "reference_objects": "how reference objects guide this step"
    }}
  ],
  "execution_order": "order of steps",
  "object_processing_rules": "rules for processing objects"
}}

Focus on extracting EXACT conditions and transformations that work across ALL training examples."""
        
        images = []
        for in_grid, out_grid in zip(input_grids, output_grids):
            images.append(grid_to_image(in_grid, 50))
            images.append(grid_to_image(out_grid, 50))
        
        result, _ = self.call_api(prompt, images)
        
        if isinstance(result, dict):
            steps = result.get('generalized_steps', [])
            if steps:
                print(f"\n✓ Generated {len(steps)} generalized steps")
                return steps
        
        # Parse JSON if needed
        parsed = self._parse_analysis_json(result if isinstance(result, str) else str(result), "generalized steps")
        if isinstance(parsed, dict):
            steps = parsed.get('generalized_steps', [])
            if steps:
                print(f"\n✓ Generated {len(steps)} generalized steps")
                return steps
        
        print("  ⚠️ Failed to generate generalized steps")
        return []
    
    def generate_transition_determinants_v7(self, all_training_examples: List[Dict],
                                           comprehensive_analysis: Dict) -> Dict:
        """Generate transition determinants: (CONDITION, OBJECT) → transformations"""
        print(f"\n{'='*80}")
        print("GENERATING TRANSITION DETERMINANTS V7")
        print(f"{'='*80}\n")
        
        input_grids = [ex['input'] for ex in all_training_examples]
        output_grids = [ex['output'] for ex in all_training_examples]
        
        grids_text = ""
        for i, (in_grid, out_grid) in enumerate(zip(input_grids, output_grids)):
            grids_text += f"\nTRAINING {i+1}:\n"
            grids_text += f"Input: {self._format_grid(in_grid)}\n"
            grids_text += f"Output: {self._format_grid(out_grid)}\n"
        
        prompt = f"""Generate TRANSITION DETERMINANTS: (CONDITION, OBJECT) → transformations

{grids_text}

COMPREHENSIVE ANALYSIS:
{json.dumps(comprehensive_analysis, indent=2)}

For EACH type of transition, specify:

1. CONDITION: Exact conditions that determine when this transition applies
   - Object properties (colors, shapes, sizes, positions)
   - Relationships (near reference, adjacent to X, etc.)
   - Countable properties (holes, length, height)
   - Most common (matches most common shape/color)
   - 1-to-1 correlations (if shape X then color Y)

2. OBJECT: What type of object this applies to
   - Solid objects, line objects, reference objects, scaled objects

3. TRANSFORMATIONS: Exact transformations to apply
   - Can be multiple transformations in sequence
   - Include all tool types: recolor, reshape, move, scale, rotate, reflect, repeat pattern, complete, fit, draw, handle negative space, etc.

Return JSON:
{{
  "transition_determinants": [
    {{
      "transition_type": "color_change|replication|movement|creation|removal|scale|rotate|reflect|complete|fit",
      "condition": {{
        "description": "clear condition statement",
        "properties": {{
          "colors": ["color X"],
          "shapes": ["vertical line"],
          "positions": "where",
          "sizes": "what sizes",
          "relationships": "relationships",
          "countable": "countable properties",
          "most_common": "if matches most common",
          "correlations": "1-to-1 correlations"
        }},
        "object_type": "solid_object|line_object|reference_object|scaled_object"
      }},
      "object_description": "what objects this applies to",
      "transformations": [
        {{
          "type": "transformation type",
          "description": "detailed description",
          "parameters": {{}}
        }}
      ],
      "transition_rule": {{
        "exact_transformation": "exact transformation",
        "exact_pattern": "exact pattern",
        "exact_spacing": "exact spacing"
      }},
      "application_logic": {{
        "check_properties": "how to check if object matches condition",
        "verify_conditions": "how to verify conditions",
        "apply_rule": "how to apply transformations"
      }}
    }}
  ],
  "decision_tree": "overall decision logic"
}}"""
        
        images = []
        for in_grid, out_grid in zip(input_grids, output_grids):
            images.append(grid_to_image(in_grid, 50))
            images.append(grid_to_image(out_grid, 50))
        
        result, _ = self.call_api(prompt, images)
        
        if isinstance(result, dict):
            return result
        
        parsed = self._parse_analysis_json(result if isinstance(result, str) else str(result), "transition determinants")
        if isinstance(parsed, dict):
            return parsed
        
        return {"transition_determinants": [], "decision_tree": "Check properties and apply transformations"}
    
    def generate_booklet_pattern_v7(self, generalized_step_sequence: List[Dict],
                                    transition_determinants: Dict) -> Dict:
        """Generate booklet pattern from generalized steps"""
        print(f"\n{'='*80}")
        print("GENERATING BOOKLET PATTERN V7")
        print(f"{'='*80}\n")
        
        steps_text = json.dumps(generalized_step_sequence, indent=2)
        determinants_text = json.dumps(transition_determinants, indent=2)
        
        prompt = f"""Generate booklet pattern from generalized steps:

GENERALIZED STEPS:
{steps_text}

TRANSITION DETERMINANTS:
{determinants_text}

Extract the overall pattern and execution order.

Return JSON:
{{
  "generalization": "overall pattern description",
  "step_sequence_pattern": [
    {{
      "step_type": "step type",
      "order": 1,
      "description": "what this step does",
      "conditions": "when this applies"
    }}
  ],
  "object_processing_order": {{
    "reference_objects": "how reference objects are identified",
    "processing_order": "order objects are processed",
    "removed_objects": "which objects are removed",
    "new_objects": "when/where new objects are created"
  }},
  "transformation_rules": [
    {{
      "rule": "general transformation rule",
      "applies_to": "what objects/conditions",
      "uses_reference": "how reference objects guide this"
    }}
  ]
}}"""
        
        result, _ = self.call_api(prompt, [])
        
        if isinstance(result, dict):
            return result
        
        parsed = self._parse_analysis_json(result if isinstance(result, str) else str(result), "booklet pattern")
        if isinstance(parsed, dict):
            return parsed
        
        return {
            "generalization": "Apply generalized steps in order",
            "step_sequence_pattern": [],
            "object_processing_order": {},
            "transformation_rules": []
        }
    
    def apply_generalized_steps_to_example(self, input_grid: List[List[int]], output_grid: List[List[int]],
                                          generalized_step_sequence: List[Dict],
                                          transition_determinants: Dict,
                                          comprehensive_analysis: Dict,
                                          puzzle_id: str, training_num: int) -> List[Dict]:
        """Apply generalized steps to a specific training example - each step processes one object"""
        print(f"\n{'='*80}")
        print(f"APPLYING GENERALIZED STEPS TO TRAINING EXAMPLE {training_num}")
        print(f"{'='*80}\n")
        
        steps = []
        step_num = 0
        current_grid = [row[:] for row in input_grid]
        
        # Initial step
        step_num += 1
        steps.append({
            "step_num": step_num,
            "description": "Start with input grid",
            "grid": [row[:] for row in current_grid],
            "is_crop_step": False
        })
        
        # Detect objects in input
        print("Detecting objects in input...")
        input_objects = self.detect_objects_with_model(input_grid, "input", retry_count=3, use_hybrid=False)
        print(f"  Found {len(input_objects)} objects")
        
        # Detect objects in output
        output_objects = []
        if output_grid:
            output_objects = self.detect_objects_with_model(output_grid, "output", retry_count=3, use_hybrid=False)
            print(f"  Found {len(output_objects)} objects in output")
        
        # Match objects
        object_matches = {}
        if output_grid and output_objects:
            object_matches = self.match_objects_with_model(input_objects, output_objects, input_grid, output_grid)
        
        # Process each generalized step
        for gen_step in generalized_step_sequence:
            step_type = gen_step.get('step_type', 'transform')
            condition = gen_step.get('condition', {})
            transformations = gen_step.get('transformations', [])
            applies_to = gen_step.get('applies_to', 'all objects')
            
            print(f"\n{'='*80}")
            print(f"APPLYING GENERALIZED STEP {gen_step.get('step_num', 'N/A')}: {step_type.upper()}")
            print(f"  Condition: {condition.get('description', 'N/A')}")
            print(f"  Applies to: {applies_to}")
            print(f"  Transformations: {len(transformations)}")
            print(f"{'='*80}\n")
            
            # Find objects that match the condition
            matching_objects = self.find_objects_matching_condition(input_objects, condition, current_grid, input_grid)
            
            if not matching_objects:
                print(f"  ⏭️ No objects match condition - skipping step")
                continue
            
            print(f"  Found {len(matching_objects)} objects matching condition")
            
            # Process each matching object
            for obj_idx in matching_objects:
                input_obj = input_objects[obj_idx]
                output_obj_idx = object_matches.get(obj_idx)
                
                # Check if object changes
                if output_grid and output_obj_idx is not None:
                    # Crop to compare
                    cropped_input_check, _ = self.crop_to_object(current_grid, input_obj['bbox'])
                    cropped_output_check, _ = self.crop_to_object(output_grid, input_obj['bbox'])
                    accuracy_check = self.compare_grids(cropped_input_check, cropped_output_check)
                    if accuracy_check >= 1.0:
                        print(f"  ⏭️ Object {obj_idx + 1} does NOT change - skipping")
                        continue
                
                print(f"\n  Processing Object {obj_idx + 1}: {input_obj.get('description', 'N/A')}")
                
                # Apply transformations to this object
                object_steps, updated_grid = self.apply_transformations_to_object(
                    obj_idx, input_obj, current_grid, output_grid, 
                    transformations, condition, gen_step, step_num
                )
                
                steps.extend(object_steps)
                step_num += len(object_steps)
                
                # Update current grid from last step
                current_grid = updated_grid
        
        # Final step
        step_num += 1
        final_accuracy = None
        if output_grid:
            final_accuracy = self.compare_grids(current_grid, output_grid)
        
        steps.append({
            "step_num": step_num,
            "description": "Final output",
            "grid": [row[:] for row in current_grid],
            "is_final_step": True,
            "accuracy": final_accuracy
        })
        
        return steps
    
    def find_objects_matching_condition(self, input_objects: List[Dict], condition: Dict,
                                       current_grid: List[List[int]], input_grid: List[List[int]]) -> List[int]:
        """Find objects that match the condition"""
        matching = []
        props = condition.get('properties', {})
        object_type = condition.get('object_type', 'solid_object')
        
        for i, obj in enumerate(input_objects):
            matches = True
            
            # Check colors
            if props.get('colors'):
                obj_colors = obj.get('colors', [])
                required_colors = props['colors']
                if not any(c in obj_colors for c in required_colors):
                    matches = False
            
            # Check shapes
            if props.get('shapes') and matches:
                obj_desc = obj.get('description', '').lower()
                required_shapes = [s.lower() for s in props['shapes']]
                if not any(shape in obj_desc for shape in required_shapes):
                    matches = False
            
            # Check object type
            if object_type != 'solid_object' and matches:
                # For now, assume all are solid objects unless specified
                pass
            
            if matches:
                matching.append(i)
        
        return matching
    
    def apply_transformations_to_object(self, obj_idx: int, input_obj: Dict,
                                      current_grid: List[List[int]], output_grid: List[List[int]],
                                      transformations: List[Dict], condition: Dict,
                                      gen_step: Dict, start_step_num: int) -> Tuple[List[Dict], List[List[int]]]:
        """Apply transformations to a single object - returns crop-transform-uncrop steps"""
        steps = []
        step_num = start_step_num
        
        bbox = input_obj.get('bbox')
        if not bbox:
            return steps
        
        # Crop to object
        cropped_input, crop_metadata = self.crop_to_object(current_grid, bbox)
        
        cropped_output = None
        if output_grid:
            cropped_output, _ = self.crop_to_object(output_grid, bbox)
        
        # Crop step
        step_num += 1
        steps.append({
            "step_num": step_num,
            "description": f"CROP: Object {obj_idx + 1} ({input_obj.get('description', 'object')})",
            "grid": [row[:] for row in current_grid],
            "object_num": obj_idx + 1,
            "is_crop_step": True,
            "crop_metadata": crop_metadata,
            "cropped_input": cropped_input,
            "cropped_output": cropped_output,
            "bbox": bbox
        })
        
        crop_step_num = step_num
        
        # Apply transformations
        transformed_grid = cropped_input
        for trans in transformations:
            trans_type = trans.get('type', 'transform')
            trans_desc = trans.get('description', '')
            trans_params = trans.get('parameters', {})
            
            # Build transformation prompt
            transform_prompt = f"""Apply transformation to this object:

Cropped Input:
{self._format_grid(transformed_grid)}

Cropped Target:
{self._format_grid(cropped_output) if cropped_output else 'N/A'}

Transformation Type: {trans_type}
Description: {trans_desc}
Parameters: {json.dumps(trans_params, indent=2)}

Generalized Step Condition: {condition.get('description', 'N/A')}

Apply the transformation. Use generate_grid tool."""
            
            transform_img = grid_to_image(transformed_grid, 50)
            target_img = grid_to_image(cropped_output, 50) if cropped_output else None
            
            imgs = [transform_img]
            if target_img:
                imgs.append(target_img)
            
            description, transformed_grid = self.call_api(
                transform_prompt, imgs, transformed_grid,
                tool_choice={"type": "function", "function": {"name": "generate_grid"}}
            )
            
            if not transformed_grid:
                transformed_grid = cropped_input
        
        # Transform step
        step_num += 1
        accuracy = None
        if cropped_output:
            accuracy = self.compare_grids(transformed_grid, cropped_output)
        
        steps.append({
            "step_num": step_num,
            "description": f"TRANSFORM: {', '.join([t.get('type', 'transform') for t in transformations])} - Object {obj_idx + 1}",
            "grid": transformed_grid,
            "object_num": obj_idx + 1,
            "is_crop_step": False,
            "is_cropped_view": True,
            "accuracy": accuracy,
            "parent_crop_step": crop_step_num,
            "cropped_ground_truth": cropped_output,
            "cropped_output": cropped_output
        })
        
        # Uncrop - need to pass current_grid properly
        # Get the current grid state (should be passed in, but for now use the one from parent)
        # Note: This is a simplified version - in full implementation, current_grid should be maintained
        uncropped_grid = self.uncrop_to_full_grid(transformed_grid, current_grid, crop_metadata)
        
        step_num += 1
        uncrop_accuracy = None
        if output_grid:
            uncrop_accuracy = self.compare_grids(uncropped_grid, output_grid)
        
        steps.append({
            "step_num": step_num,
            "description": f"UNCROP: Object {obj_idx + 1} back to grid",
            "grid": [row[:] for row in uncropped_grid],
            "object_num": obj_idx + 1,
            "is_crop_step": False,
            "is_uncrop_step": True,
            "accuracy": uncrop_accuracy,
            "parent_crop_step": crop_step_num
        })
        
        return steps, uncropped_grid  # Return both steps and updated grid
    
    # ==================== MAIN RUN METHOD ====================
    
    def run(self, puzzle_id: str, training_num: int = 1, shared_analysis: str = None, is_test: bool = False):
        """Main run function - object-centric approach with iterative improvement"""
        print(f"\n{'='*80}")
        example_type = "TESTING" if is_test else "TRAINING"
        print(f"VISUAL STEP GENERATOR V7: {puzzle_id} ({example_type.lower()}_{training_num:02d})")
        print(f"{'='*80}\n")
        
        # Load puzzle data
        arc = self.load_arc_puzzle(puzzle_id)
        
        all_training_examples = arc['train']
        
        # V7 ARCHITECTURE: Generate generalized steps FIRST (before training booklets)
        # Check if generalized steps already exist, otherwise generate them
        generalized_steps = self._load_or_generate_generalized_steps(puzzle_id, all_training_examples)
        
        # Extract generalized step components
        transformation_rule = generalized_steps.get('transformation_rule')
        booklet_pattern = generalized_steps.get('booklet_pattern')
        generalized_step_sequence = generalized_steps.get('generalized_step_sequence', [])
        transition_determinants = generalized_steps.get('transition_determinants')
        object_step_mapping = None
        
        # Get input/output grids
        if is_test:
            test_examples = arc.get('test', [])
            if training_num < 1 or training_num > len(test_examples):
                raise ValueError(f"Test example {training_num} not found. Available: 1-{len(test_examples)}")
            example = test_examples[training_num - 1]
            input_grid = example['input']
            output_grid = None
        else:
            if training_num < 1 or training_num > len(all_training_examples):
                raise ValueError(f"Training example {training_num} not found. Available: 1-{len(all_training_examples)}")
            example = all_training_examples[training_num - 1]
            input_grid = example['input']
            output_grid = example['output']
        
        print(f"Input: {len(input_grid)}×{len(input_grid[0])}")
        if output_grid:
            print(f"Output: {len(output_grid)}×{len(output_grid[0])}")
        print(f"Analyzing {len(all_training_examples)} training examples\n")
        
        # Comprehensive analysis
        if shared_analysis and isinstance(shared_analysis, dict):
            comprehensive_analysis = shared_analysis
            puzzle_analysis = {"analysis": str(shared_analysis)}
        else:
            # Perform comprehensive analysis
            comprehensive_analysis = self.comprehensive_analysis(all_training_examples)
            puzzle_analysis = {"analysis": json.dumps(comprehensive_analysis, indent=2)}
            # Also run analyze_puzzle_pattern for backward compatibility
            puzzle_pattern = self.analyze_puzzle_pattern(all_training_examples)
            puzzle_analysis.update(puzzle_pattern)
        
        # Store comprehensive analysis for mini-analysis access
        self._current_comprehensive_analysis = comprehensive_analysis
        shared_analysis = puzzle_analysis.get('analysis', '')
        
        # V7: Apply generalized steps to training examples to generate training booklets
        if not is_test:
            print(f"\nV7: Applying generalized steps to training example {training_num}...")
            # Apply generalized steps to this training example
            steps = self.apply_generalized_steps_to_example(
                input_grid, output_grid, generalized_step_sequence, 
                transition_determinants, comprehensive_analysis, puzzle_id, training_num
            )
            # Save training booklet
            self._save_results(puzzle_id, training_num, shared_analysis or "", 
                              steps, input_grid, output_grid, is_test=False)
            print(f"\n{'='*80}")
            print(f"✅ COMPLETE: Generated {len(steps)} steps")
            print(f"{'='*80}\n")
            return
        
        # Merge transformation rule and booklet pattern into puzzle analysis if available
        if transformation_rule:
            puzzle_analysis['transformation_rule'] = transformation_rule
            print(f"\n✓ Using transformation rule:")
            print(f"  Rule: {transformation_rule.get('rule_description', 'N/A')[:100]}...")
            if transformation_rule.get('generalization'):
                gen = transformation_rule['generalization']
                print(f"  Abstract rule: {gen.get('abstract_rule', 'N/A')[:100]}...")
        
        if booklet_pattern:
            puzzle_analysis['booklet_pattern'] = booklet_pattern
            print(f"\n✓ Using step-by-step pattern from training booklets:")
            print(f"  Pattern: {booklet_pattern.get('generalization', 'N/A')[:100]}...")
        
        if generalized_step_sequence:
            puzzle_analysis['generalized_step_sequence'] = generalized_step_sequence
            print(f"\n✓ Using generalized step sequence:")
            print(f"  {len(generalized_step_sequence)} steps generated")
            step_types = [s.get('step_type', 'N/A') for s in generalized_step_sequence[:5]]
            print(f"  Execution order: {', '.join(step_types)}...")
        
        if transition_determinants:
            puzzle_analysis['transition_determinants'] = transition_determinants
            print(f"\n✓ Using transition determinants:")
            transitions = transition_determinants.get('transition_determinants', [])
            print(f"  {len(transitions)} transition types identified")
            if transition_determinants.get('decision_tree'):
                print(f"  Decision tree: {transition_determinants.get('decision_tree', 'N/A')[:80]}...")
        
        # Test mode: Apply generalized steps to test input
        if is_test:
            print(f"\nV7 TEST MODE: Applying generalized steps to test example {training_num}...")
            steps = self.apply_generalized_steps_to_example(
                input_grid, None, generalized_step_sequence, 
                transition_determinants, comprehensive_analysis, puzzle_id, training_num
            )
            # Save test results
            self._save_results(puzzle_id, training_num, shared_analysis or "", 
                              steps, input_grid, None, is_test=True)
            print(f"\n{'='*80}")
            print(f"✅ COMPLETE: Generated {len(steps)} steps")
            print(f"{'='*80}\n")
            return
        
        # Detect objects using model only (filters out single-cell objects)
        print("Detecting objects (model only - multi-cell objects)...")
        input_objects = self.detect_objects_with_model(input_grid, "input", retry_count=3, use_hybrid=False)
        
        if output_grid:
            output_objects = self.detect_objects_with_model(output_grid, "output", retry_count=3, use_hybrid=False)
        else:
            output_objects = []
        
        # For test mode: Map training steps to test objects (Method 1) - OLD CODE, should not reach here
        if is_test and False:  # Disabled - using new V7 approach above
            print("\nTEST MODE: Mapping training steps to test objects...")
            object_step_mapping = self.map_training_steps_to_test_objects(
                training_booklets, input_objects, input_grid
            )
        
        # For test mode: identify reference objects by comparing test input to training examples
        test_reference_objects = []
        if is_test:
            print("\nTEST MODE: Identifying reference objects from training examples...")
            test_reference_objects = self.identify_test_reference_objects(
                input_grid, input_objects, all_training_examples, transformation_rule
            )
            if test_reference_objects:
                print(f"  ✓ Found {len(test_reference_objects)} reference objects in test input")
                for ref in test_reference_objects:
                    ref_obj = ref.get('object', {})
                    print(f"    - {ref_obj.get('description', 'N/A')}: {ref.get('reasoning', 'N/A')[:60]}...")
            else:
                print(f"  ⚠️ No reference objects identified")
        
        print(f"  Input objects: {len(input_objects)}")
        for i, obj in enumerate(input_objects):
            colors = obj.get('colors', [])
            color_str = f"color {colors[0]}" if colors else "N/A"
            print(f"    Object {i+1}: {obj.get('description', 'N/A')}, {color_str}, size {obj.get('size', 'N/A')}")
        
        if output_grid:
            print(f"  Output objects: {len(output_objects)}")
            for i, obj in enumerate(output_objects):
                colors = obj.get('colors', [])
                color_str = f"color {colors[0]}" if colors else "N/A"
                print(f"    Object {i+1}: {obj.get('description', 'N/A')}, {color_str}, size {obj.get('size', 'N/A')}")
        
        # Match objects (for training) or predict matches (for test)
        if output_grid:
            print("\nMatching objects...")
            object_matches = self.match_objects_with_model(input_objects, output_objects, input_grid, output_grid)
            
            print(f"  Matches:")
            for input_idx, output_idx in object_matches.items():
                if output_idx is not None:
                    print(f"    Input {input_idx+1} → Output {output_idx+1}")
                else:
                    print(f"    Input {input_idx+1} → (removed)")
            
            matched_output_indices = set(v for v in object_matches.values() if v is not None)
            new_output_objects = [i for i in range(len(output_objects)) if i not in matched_output_indices]
            
            if new_output_objects:
                print(f"  New objects: {[i+1 for i in new_output_objects]}")
        else:
            # Test mode: predict which objects will transform based on training pattern
            # All input objects should be processed (they may transform or be removed)
            print("\nTest mode: All input objects will be processed based on training pattern")
            object_matches = {i: None for i in range(len(input_objects))}  # Will be determined during transformation
            new_output_objects = []  # Will be detected if needed based on pattern
        
        # Get reference objects and transitions from analysis
        if is_test and test_reference_objects:
            # Use test-specific reference objects
            reference_objects = test_reference_objects
            print(f"\n  Using {len(reference_objects)} test reference objects to adapt steps")
        else:
            # Use reference objects from puzzle analysis (training mode)
            reference_objects = self.get_reference_objects_from_analysis(puzzle_analysis, input_objects)
        
        transitions = self.find_transitions(input_grid, output_grid, puzzle_analysis) if output_grid else []
        optimal_ordering_steps = puzzle_analysis.get('optimal_ordering', [])
        
        print(f"\n  Reference objects found: {len(reference_objects)}")
        for ref in reference_objects:
            print(f"    - {ref.get('object', {}).get('description', 'N/A')}: {ref.get('reasoning', 'N/A')[:60]}...")
        
        print(f"  Transitions found: {len(transitions)}")
        for i, trans in enumerate(transitions):
            print(f"    - Transition {i+1}: {trans.get('type', 'N/A')} - {trans.get('description', 'N/A')[:60]}...")
        
        # Map optimal ordering to actual object indices
        # If optimal_ordering_steps is provided, use it; otherwise fall back to sequential
        if optimal_ordering_steps and len(optimal_ordering_steps) > 0:
            # Extract object indices from optimal ordering steps
            object_processing_order = []
            reference_indices = set()
            
            for step_info in optimal_ordering_steps:
                step_type = step_info.get('step_type', '')
                if step_type == 'identify_reference':
                    # Reference objects are identified but not transformed
                    trans_idx = step_info.get('transition_idx')
                    if trans_idx is None:
                        # This is a reference identification step
                        continue
                elif step_type in ['transform', 'new_object', 'remove']:
                    trans_idx = step_info.get('transition_idx')
                    if trans_idx is not None and trans_idx < len(transitions):
                        # Map transition to object - this is simplified, could be improved
                        # For now, we'll process objects in the order they appear
                        pass
            
            # Fallback: use sequential if mapping fails
            if not object_processing_order:
                object_processing_order = list(range(len(input_objects)))
        else:
            # Fallback to sequential ordering
            object_processing_order = list(range(len(input_objects)))
        
        # Build reference object indices set
        reference_indices = {ref['object_idx'] for ref in reference_objects}
        
        print(f"\n  Processing order: {object_processing_order}")
        print(f"  Reference objects (won't be transformed): {sorted(reference_indices)}")
        
        # Initialize grid state
        current_full_grid = [row[:] for row in input_grid]
        all_step_results = []
        step_num = 0
        processed_object_indices = set()
        successfully_transformed = set()  # Track objects with successful transformations
        reference_identified = set()  # Track which reference objects have been identified
        
        step_num += 1
        all_step_results.append({
            "step_num": step_num,
            "description": "Start with input grid (all objects in original state)",
            "grid": current_full_grid,
            "object_num": None,
            "is_crop_step": False
        })
        
        # First: Identify all reference objects (they don't change, just identify them)
        # Analyze which parts/features of each reference object are used for transformations
        # In test mode, reference objects help adapt steps to the test input
        for ref_info in reference_objects:
            ref_idx = ref_info['object_idx']
            if ref_idx not in reference_identified:
                ref_obj = ref_info['object']
                print(f"\n{'='*80}")
                if is_test:
                    print(f"IDENTIFYING REFERENCE OBJECT {ref_idx + 1} (TEST MODE - adapts steps)")
                else:
                    print(f"IDENTIFYING REFERENCE OBJECT {ref_idx + 1}")
                print(f"  Description: {ref_obj.get('description', 'N/A')}")
                print(f"  Reasoning: {ref_info.get('reasoning', 'N/A')[:100]}...")
                if is_test and ref_info.get('how_adapts_steps'):
                    print(f"  How adapts steps: {ref_info.get('how_adapts_steps', 'N/A')[:100]}...")
                print(f"{'='*80}\n")
                
                # Analyze which parts of the reference object are used for transformations
                ref_parts_used = self.analyze_reference_parts(ref_obj, all_training_examples, input_grid, output_grid)
                
                # Store reference parts in ref_info for later use
                ref_info['reference_parts_used'] = ref_parts_used
                
                step_num += 1
                ref_bbox = ref_obj.get('bbox')
                ref_crop, ref_metadata = self.crop_to_object(input_grid, ref_bbox) if ref_bbox else (None, None)
                
                # Highlight primary part used
                primary_part = ref_parts_used.get('primary_feature', 'full object')
                parts_list = ref_parts_used.get('parts_used', [])
                primary_part_info = next((p for p in parts_list if p.get('part_name') == primary_part), parts_list[0] if parts_list else {})
                
                ref_description = f"⭐ REFERENCE OBJECT: {ref_obj.get('description', 'object')}"
                if is_test:
                    ref_description += " (constant - adapts steps to test input)"
                else:
                    ref_description += " (constant - used as template)"
                
                all_step_results.append({
                    "step_num": step_num,
                    "description": ref_description,
                    "grid": [row[:] for row in current_full_grid],
                    "object_num": ref_idx + 1,
                    "is_reference_step": True,
                    "reference_reasoning": ref_info.get('reasoning', ''),
                    "reference_pattern": ref_info.get('pattern', ''),
                    "how_adapts_steps": ref_info.get('how_adapts_steps', '') if is_test else None,
                    "reference_crop": ref_crop,
                    "reference_metadata": ref_metadata,
                    "reference_parts_used": ref_parts_used,  # Which parts are used for transformations
                    "primary_part": primary_part_info.get('part_name', 'full object'),
                    "primary_part_description": primary_part_info.get('description', ''),
                    "primary_part_how_used": primary_part_info.get('how_used', ''),
                    "is_constant": True,  # Reference objects don't change
                    "is_test_mode": is_test
                })
                
                reference_identified.add(ref_idx)
                processed_object_indices.add(ref_idx)  # Mark as processed (but not transformed)
        
        # Iterative processing loop - keep going until output is reached
        max_iterations = 10  # Increased max iterations
        iteration = 0
        all_objects_processed = False
        last_accuracy = 0.0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration}/{max_iterations}")
            print(f"{'='*80}\n")
            
            if output_grid:
                current_accuracy = self.compare_grids(current_full_grid, output_grid)
                print(f"Current accuracy: {current_accuracy:.1%}")
                
                # Check if we've reached the output
                if current_accuracy >= 0.99:
                    print(f"✓ Grid matches output! Stopping.")
                    break
                
                # Check if we're making progress
                if current_accuracy <= last_accuracy + 0.001:  # Less than 0.1% improvement
                    print(f"  ⚠️ Accuracy not improving ({last_accuracy:.1%} → {current_accuracy:.1%})")
                    # Still continue to try more objects
                
                last_accuracy = current_accuracy
            else:
                # Test mode: check if we've processed all objects
                if len(processed_object_indices) >= len(input_objects):
                    print(f"  All input objects processed (test mode)")
                    # Continue to check for new objects based on pattern
            
            # Determine which objects to process (skip reference objects - they don't change)
            if iteration == 1:
                # Process non-reference objects in optimal order
                objects_to_process = [i for i in object_processing_order if i not in reference_indices]
            else:
                # Process unprocessed non-reference objects
                unprocessed = [i for i in range(len(input_objects)) 
                              if i not in processed_object_indices and i not in reference_indices]
                if unprocessed:
                    objects_to_process = unprocessed
                    print(f"Processing {len(unprocessed)} unprocessed objects...")
                else:
                    # All input objects processed
                    if output_grid:
                        # Training mode: check if grid is complete
                        # Find cells that still differ from output
                        diff_cells = []
                        for r in range(len(current_full_grid)):
                            for c in range(len(current_full_grid[0])):
                                if (r < len(output_grid) and c < len(output_grid[0]) and 
                                    current_full_grid[r][c] != output_grid[r][c]):
                                    diff_cells.append((r, c))
                        
                        if diff_cells:
                            print(f"All input objects processed but {len(diff_cells)} cells still differ from output.")
                            
                            # Check which objects need retrying (only those whose regions don't match)
                            objects_to_retry = []
                            for obj_idx in range(len(input_objects)):
                                if obj_idx in successfully_transformed:
                                    # Check if this object's region still matches output
                                    input_obj = input_objects[obj_idx]
                                    bbox = input_obj.get('bbox')
                                    if bbox:
                                        min_r, min_c, max_r, max_c = bbox
                                        # Check if this region matches output
                                        region_matches = True
                                        for r in range(min_r, max_r + 1):
                                            for c in range(min_c, max_c + 1):
                                                if (0 <= r < len(current_full_grid) and 0 <= c < len(current_full_grid[0]) and
                                                    0 <= r < len(output_grid) and 0 <= c < len(output_grid[0])):
                                                    if current_full_grid[r][c] != output_grid[r][c]:
                                                        region_matches = False
                                                        break
                                            if not region_matches:
                                                break
                                        
                                        if not region_matches:
                                            objects_to_retry.append(obj_idx)
                                            successfully_transformed.discard(obj_idx)
                                else:
                                    # Not successfully transformed yet, retry it
                                    objects_to_retry.append(obj_idx)
                            
                            if objects_to_retry:
                                print(f"  Retrying {len(objects_to_retry)} objects with incorrect regions...")
                                processed_object_indices = set(obj for obj in processed_object_indices if obj not in objects_to_retry)
                                objects_to_process = objects_to_retry
                            else:
                                # All objects are correct but grid still incomplete - might be new objects
                                print(f"  All objects correct, checking for new objects...")
                                break
                        else:
                            # No differences found
                            print(f"  No cell differences found. Stopping.")
                            break
                    else:
                        # Test mode: all input objects processed, check if pattern suggests new objects
                        # Based on training examples, determine if new objects should be created
                        print(f"  All input objects processed. Checking if pattern requires new objects...")
                        # Will be handled in new objects section
                        break
            
            if not objects_to_process:
                break
            
            # Process objects (skip reference objects - they're already identified and don't change)
            for order_idx, obj_idx in enumerate(objects_to_process):
                # Skip reference objects - they don't change
                if obj_idx in reference_indices:
                    print(f"\n  ⏭️ Skipping Object {obj_idx + 1} - reference object (constant)")
                    continue
                
                # Skip if already successfully transformed (unless explicitly retrying)
                if obj_idx in successfully_transformed and obj_idx not in processed_object_indices:
                    # Check if this object's region already matches output
                    if output_grid:
                        input_obj = input_objects[obj_idx]
                        bbox = input_obj.get('bbox')
                        if bbox:
                            min_r, min_c, max_r, max_c = bbox
                            region_matches = True
                            for r in range(min_r, max_r + 1):
                                for c in range(min_c, max_c + 1):
                                    if (0 <= r < len(current_full_grid) and 0 <= c < len(current_full_grid[0]) and
                                        0 <= r < len(output_grid) and 0 <= c < len(output_grid[0])):
                                        if current_full_grid[r][c] != output_grid[r][c]:
                                            region_matches = False
                                            break
                                if not region_matches:
                                    break
                            
                            if region_matches:
                                print(f"\n  ⏭️ Skipping Object {obj_idx + 1} - already correctly transformed")
                                processed_object_indices.add(obj_idx)
                                continue
                
                input_obj = input_objects[obj_idx]
                print(f"\n{'='*80}")
                print(f"PROCESSING OBJECT {obj_idx + 1}/{len(input_objects)} (order {order_idx + 1}/{len(objects_to_process)})")
                print(f"  Description: {input_obj.get('description', 'N/A')}")
                print(f"{'='*80}\n")
                
                output_obj_idx = object_matches.get(obj_idx) if output_grid else None
                
                # In test mode, process all objects (they may transform based on training pattern)
                if output_obj_idx is not None or is_test:
                    # Matched object or test mode
                    if output_grid and output_obj_idx is not None:
                        output_obj = output_objects[output_obj_idx]
                        print(f"  Matched to output object {output_obj_idx + 1}: {output_obj.get('description', 'N/A')}")
                    elif is_test:
                        print(f"  Test mode: determining transition based on input properties")
                        
                        # Determine what transition should happen based on object properties
                        if transition_determinants:
                            transitions = transition_determinants.get('transition_determinants', [])
                            obj_colors = input_obj.get('colors', [])
                            obj_desc = input_obj.get('description', '').lower()
                            
                            # Find matching transition based on properties
                            matching_transition = None
                            for trans in transitions:
                                props = trans.get('determining_properties', {})
                                trans_colors = props.get('colors', [])
                                trans_shapes = props.get('shapes', [])
                                
                                # Check if object matches transition properties
                                color_match = any(c in trans_colors for c in obj_colors) if trans_colors else True
                                shape_match = any(s in obj_desc for s in trans_shapes) if trans_shapes else True
                                
                                if color_match and shape_match:
                                    matching_transition = trans
                                    break
                            
                            if matching_transition:
                                trans_type = matching_transition.get('transition_type', 'N/A')
                                trans_rule = matching_transition.get('transition_rule', {})
                                print(f"    Determined transition: {trans_type}")
                                if trans_rule.get('exact_transformation'):
                                    print(f"    Rule: {trans_rule.get('exact_transformation', 'N/A')[:60]}...")
                                if trans_rule.get('exact_pattern'):
                                    print(f"    Pattern: {trans_rule.get('exact_pattern', 'N/A')[:60]}...")
                        
                        if generalized_step_sequence:
                            # Find current step in sequence
                            current_step_info = None
                            for seq_step in generalized_step_sequence:
                                if seq_step.get('step_type') == 'transform' and seq_step.get('applies_to'):
                                    # Check if this step applies to current object
                                    applies_to = seq_step.get('applies_to', '').lower()
                                    obj_desc = input_obj.get('description', '').lower()
                                    if any(word in obj_desc for word in applies_to.split()[:3]) or \
                                       any(word in applies_to for word in obj_desc.split()[:3]):
                                        current_step_info = seq_step
                                        break
                            
                            if current_step_info:
                                print(f"    Following step {current_step_info.get('step_num', 'N/A')}: {current_step_info.get('description', 'N/A')[:60]}...")
                                print(f"    Adaptation: {current_step_info.get('adaptation', 'N/A')[:60]}...")
                            else:
                                print(f"    Using generalized sequence ({len(generalized_step_sequence)} steps)")
                        
                        if transformation_rule:
                            rule_desc = transformation_rule.get('rule_description', 'N/A')
                            print(f"    Rule: {rule_desc[:80]}...")
                        if booklet_pattern:
                            print(f"    Pattern: {booklet_pattern.get('generalization', 'N/A')[:80]}...")
                    
                    # Use already-identified reference objects (don't detect again)
                    reference_info = None
                    if reference_objects:
                        # Find the most relevant reference object for this transformation
                        # Use the first reference object as default, or find one that's related
                        ref_info = reference_objects[0]  # Could be improved to find best match
                        ref_idx = ref_info['object_idx']
                        
                        # Get which part of reference is used for this specific transformation
                        ref_part_used = self.get_reference_part_for_transformation(
                            ref_info, input_obj, output_obj if output_grid and output_obj_idx is not None else None,
                            all_training_examples
                        )
                        
                        print(f"\n  ⭐ Using reference object {ref_idx + 1}: {ref_info.get('object', {}).get('description', 'N/A')}")
                        print(f"    Part used: {ref_part_used.get('part_name', 'full object')}")
                        print(f"    How: {ref_part_used.get('how_used', 'N/A')[:80]}...")
                        if is_test and ref_info.get('how_adapts_steps'):
                            print(f"    Adapts steps: {ref_info.get('how_adapts_steps', 'N/A')[:80]}...")
                        
                        reference_info = {
                            'object_idx': ref_idx,
                            'object': ref_info['object'],
                            'reasoning': ref_info.get('reasoning', ''),
                            'how_to_use': ref_part_used.get('how_used', 'Use as template'),
                            'pattern_across_examples': ref_info.get('pattern', ''),
                            'part_used': ref_part_used,  # Specific part used for this transformation
                            'how_adapts_steps': ref_info.get('how_adapts_steps', '') if is_test else None  # How this adapts steps in test mode
                        }
                    
                    # CRITICAL: Check if object actually changes before processing
                    if output_grid and output_obj_idx is not None:
                        # Crop both input and output to compare
                        grid_to_crop = current_full_grid if obj_idx in processed_object_indices else input_grid
                        cropped_input_check, _ = self.crop_to_object(grid_to_crop, input_obj['bbox'])
                        cropped_output_check, _ = self.crop_to_object(output_grid, input_obj['bbox'])
                        
                        # Compare cropped regions - if identical (100% match), skip crop-transform-uncrop
                        accuracy_check = self.compare_grids(cropped_input_check, cropped_output_check)
                        if accuracy_check >= 1.0:  # Perfect match - object doesn't change
                            print(f"  ✓ Object {obj_idx + 1} does NOT change (accuracy: {accuracy_check:.1%}) - skipping crop-transform-uncrop")
                            processed_object_indices.add(obj_idx)
                            successfully_transformed.add(obj_idx)
                            continue
                    
                    # Crop to object - use current grid state if retrying, otherwise use input
                    # This ensures we're working with the current state of transformations
                    grid_to_crop = current_full_grid if obj_idx in processed_object_indices else input_grid
                    cropped_input, input_metadata = self.crop_to_object(grid_to_crop, input_obj['bbox'])
                    
                    if output_grid:
                        cropped_output, output_metadata = self.crop_to_object(output_grid, input_obj['bbox'])
                        print(f"  Cropped: {len(cropped_input)}×{len(cropped_input[0])} (from {'current' if obj_idx in processed_object_indices else 'input'} grid)")
                    else:
                        cropped_output = None
                        output_metadata = input_metadata
                        print(f"  Cropped: {len(cropped_input)}×{len(cropped_input[0])} (test mode)")
                    
                    # Crop step - only add if we don't already have a crop step for this object in this iteration
                    # Check if we already have a crop step for this object in recent steps
                    existing_crop_step = None
                    for existing_step in reversed(all_step_results):
                        if (existing_step.get('is_crop_step') and 
                            existing_step.get('object_num') == obj_idx + 1 and
                            existing_step.get('current_object') == obj_idx):
                            existing_crop_step = existing_step
                            break
                    
                    if existing_crop_step:
                        # Reuse existing crop step number
                        crop_step_num = existing_crop_step['step_num']
                        print(f"  ↻ Reusing crop step {crop_step_num} for object {obj_idx + 1}")
                    else:
                        # Create new crop step
                        step_num += 1
                        crop_step_num = step_num
                        all_step_results.append({
                            "step_num": crop_step_num,
                            "description": f"CROP: Object {obj_idx + 1} ({input_obj.get('description', 'object')})",
                            "grid": [row[:] for row in current_full_grid],
                            "object_num": obj_idx + 1,
                            "is_crop_step": True,
                            "crop_metadata": input_metadata,
                            "object_description": input_obj.get('description', ''),
                            "processed_objects": list(processed_object_indices),
                            "current_object": obj_idx,
                            "cropped_input": cropped_input,
                            "cropped_output": cropped_output
                        })
                    
                    # Find corresponding training step for this object and step type (Method 2)
                    corresponding_training_step = None
                    if is_test and training_booklets:
                        # For transform step, find corresponding training transform step
                        corresponding_training_step = self.find_corresponding_training_step(
                            input_obj, obj_idx, 'transform', training_booklets
                        )
                        if corresponding_training_step:
                            print(f"    📋 Following training step {corresponding_training_step['step_num']} "
                                  f"from booklet {corresponding_training_step['booklet_idx']+1}: "
                                  f"{corresponding_training_step['description'][:60]}...")
                    
                    # Get step sequence for this object (Method 1)
                    object_step_sequence = None
                    if is_test and object_step_mapping:
                        mappings = object_step_mapping.get('object_mappings', [])
                        for mapping in mappings:
                            if mapping.get('test_object_idx') == obj_idx:
                                object_step_sequence = mapping.get('step_sequence', [])
                                if object_step_sequence:
                                    print(f"    📋 Using mapped step sequence ({len(object_step_sequence)} steps)")
                                    # Show the full crop-transform-uncrop sequence
                                    for seq_step in object_step_sequence:
                                        step_type = seq_step.get('step_type', 'N/A')
                                        step_order = seq_step.get('step_order', 'N/A')
                                        print(f"      Step {step_order} [{step_type}]: {seq_step.get('training_description', 'N/A')[:60]}...")
                                        if seq_step.get('adaptation'):
                                            print(f"        Adaptation: {seq_step.get('adaptation', 'N/A')[:60]}...")
                                break
                    
                    # Also find the exact training booklet step sequence for this object
                    training_booklet_steps = None
                    if is_test and training_booklets:
                        # Find which training booklet has a similar object and extract its full step sequence
                        test_obj_desc = input_obj.get('description', '').lower()
                        test_obj_colors = input_obj.get('colors', [])
                        
                        # First, find matching objects by description similarity
                        best_match = None
                        best_similarity = 0
                        
                        for booklet_idx, booklet in enumerate(training_booklets):
                            steps = booklet.get('steps', [])
                            
                            # Group steps by object number to find complete sequences
                            steps_by_object = {}
                            for step in steps:
                                step_obj_num = step.get('object_num')
                                if step_obj_num:
                                    if step_obj_num not in steps_by_object:
                                        steps_by_object[step_obj_num] = []
                                    steps_by_object[step_obj_num].append(step)
                            
                            # For each object in training booklet, check if it matches test object
                            for obj_num, obj_steps in steps_by_object.items():
                                # Get object description from first step
                                obj_desc = obj_steps[0].get('object_description', '').lower() if obj_steps else ''
                                
                                # Check similarity
                                desc_similarity = sum(1 for word in test_obj_desc.split() 
                                                     if word in obj_desc.split()) / max(len(test_obj_desc.split()), 1)
                                
                                # Also check color similarity
                                obj_colors = obj_steps[0].get('colors', []) if obj_steps else []
                                color_match = any(c in obj_colors for c in test_obj_colors) if obj_colors and test_obj_colors else True
                                
                                similarity_score = desc_similarity + (0.3 if color_match else 0)
                                
                                if similarity_score > best_similarity:
                                    best_similarity = similarity_score
                                    best_match = {
                                        'booklet_idx': booklet_idx,
                                        'object_num': obj_num,
                                        'steps': obj_steps
                                    }
                        
                        # Extract full crop-transform-uncrop sequence from best match
                        if best_match and best_similarity > 0.3:
                            obj_steps = best_match['steps']
                            matching_steps = []
                            
                            for step in obj_steps:
                                # Collect all steps for this object (crop, transform, uncrop)
                                if step.get('is_crop_step') or step.get('is_cropped_view') or step.get('is_uncrop_step'):
                                    matching_steps.append({
                                        'step': step,
                                        'step_num': step.get('step_num', 0),
                                        'step_type': 'crop' if step.get('is_crop_step') else 
                                                    'transform' if step.get('is_cropped_view') else 
                                                    'uncrop' if step.get('is_uncrop_step') else 'unknown',
                                        'description': step.get('description', ''),
                                        'booklet_idx': best_match['booklet_idx'],
                                        'object_num': best_match['object_num']
                                    })
                            
                            if matching_steps:
                                # Sort by step number to get crop -> transform -> uncrop order
                                matching_steps.sort(key=lambda x: x['step_num'])
                                training_booklet_steps = matching_steps
                                print(f"    📚 Found training booklet step sequence ({len(matching_steps)} steps) from booklet {best_match['booklet_idx'] + 1}, object {best_match['object_num']}:")
                                for ts in matching_steps:
                                    print(f"      Step {ts['step_num']} [{ts['step_type']}]: {ts['description'][:60]}...")
                    
                    # CRITICAL: Ensure cropped_output is set when output_grid exists
                    if output_grid and not cropped_output:
                        print(f"  ⚠️ WARNING: output_grid exists but cropped_output is None - cropping output now")
                        cropped_output, output_metadata = self.crop_to_object(output_grid, input_obj['bbox'])
                        print(f"  ✓ Cropped output: {len(cropped_output)}×{len(cropped_output[0]) if cropped_output else 0}")
                    
                    # MINI-ANALYSIS: Analyze why this transformation is needed
                    obj_properties = {
                        'input_location': input_obj.get('bbox', []),
                        'colors': input_obj.get('colors', []),
                        'shape': input_obj.get('description', ''),
                        'size': input_obj.get('size', 0),
                        'object_type': input_obj.get('object_type', 'solid_object')
                    }
                    
                    # Get comprehensive analysis if available (from run method)
                    comprehensive_analysis = getattr(self, '_current_comprehensive_analysis', {})
                    
                    print(f"    Performing mini-analysis to explain transformation reasoning...")
                    transformation_analysis = self._analyze_transformation_reason(
                        cropped_input, cropped_output, obj_properties, comprehensive_analysis,
                        puzzle_context=f"Object {obj_idx + 1}, {'test' if is_test else 'training'} mode"
                    )
                    
                    reasoning = transformation_analysis.get("reasoning", "Transformation needed to match target")
                    conditions = transformation_analysis.get("conditions", [])
                    pattern_relation = transformation_analysis.get("pattern_relation", "")
                    recommended_transforms = transformation_analysis.get("recommended_transformations", [])
                    analysis_description = transformation_analysis.get("step_description", "")
                    
                    conditions_text = "; ".join(conditions) if conditions else "No specific conditions"
                    recommended_text = ", ".join(recommended_transforms) if recommended_transforms else "unknown"
                    
                    print(f"    Analysis: {reasoning[:100]}...")
                    print(f"    Conditions: {conditions_text[:100]}...")
                    print(f"    Recommended: {recommended_text}")
                    
                    # Generate transformation
                    success = False
                    for attempt in range(1, 4):
                        print(f"\n  Attempt {attempt}/3...")
                        description, transformed_crop = self.generate_object_transformation(
                            cropped_input, cropped_output, obj_idx + 1, len(input_objects), 
                            reference_info, attempt, is_test=is_test, all_training_examples=all_training_examples,
                            booklet_pattern=booklet_pattern if is_test else None,
                            transformation_rule=transformation_rule if is_test else None,
                            generalized_step_sequence=generalized_step_sequence if is_test else None,
                            transition_determinants=transition_determinants if is_test else None,
                            input_obj=input_obj,
                            corresponding_training_step=corresponding_training_step if is_test else None,
                            object_step_sequence=object_step_sequence if is_test else None,
                            training_booklet_steps=training_booklet_steps if is_test else None,
                            transformation_analysis=transformation_analysis  # Pass mini-analysis
                        )
                        
                        if transformed_crop:
                            accuracy = None
                            if output_grid and cropped_output:
                                accuracy = self.compare_grids(transformed_crop, cropped_output)
                                print(f"    Accuracy: {accuracy:.1%}")
                                
                                if accuracy >= 0.95:
                                    success = True
                            else:
                                # Test mode: accept transformation if generated
                                print(f"    Generated transformation (test mode - applying training pattern)")
                                success = True
                            
                            if success:
                                # Check if this transformation is actually needed (not a duplicate)
                                # Only check duplicates in training mode
                                if output_grid and cropped_output:
                                    # Check if current grid region already matches output
                                    current_crop = self.crop_to_object(current_full_grid, input_obj['bbox'])[0]
                                    current_region_accuracy = self.compare_grids(current_crop, cropped_output)
                                    
                                    if current_region_accuracy >= 0.95:
                                        print(f"  ⏭️ Skipping duplicate - region already matches output ({current_region_accuracy:.1%})")
                                        processed_object_indices.add(obj_idx)
                                        successfully_transformed.add(obj_idx)
                                        
                                        # Still check overall accuracy
                                        overall_accuracy = self.compare_grids(current_full_grid, output_grid)
                                        if overall_accuracy >= 0.99:
                                            all_objects_processed = True
                                            break
                                        continue
                                # Test mode: always proceed with transformation
                                
                                # Save cropped transformation
                                step_num += 1
                                
                                # Build description with mini-analysis reasoning
                                transform_desc = description or f"Transform object {obj_idx + 1}"
                                if transformation_analysis:
                                    reasoning_snippet = transformation_analysis.get("reasoning", "")[:80]
                                    conditions_snippet = "; ".join(transformation_analysis.get("conditions", []))[:60]
                                    recommended = ", ".join(transformation_analysis.get("recommended_transformations", []))
                                    if recommended:
                                        transform_desc = f"TRANSFORM [{recommended}]: {transform_desc} | Why: {reasoning_snippet}... | Conditions: {conditions_snippet}..."
                                
                                all_step_results.append({
                                    "step_num": step_num,
                                    "description": transform_desc,
                                    "transformation_reasoning": transformation_analysis.get("reasoning", "") if transformation_analysis else None,
                                    "transformation_conditions": transformation_analysis.get("conditions", []) if transformation_analysis else None,
                                    "pattern_relation": transformation_analysis.get("pattern_relation", "") if transformation_analysis else None,
                                    "grid": transformed_crop,
                                    "object_num": obj_idx + 1,
                                    "is_crop_step": False,
                                    "is_cropped_view": True,
                                    "accuracy": accuracy,
                                    "parent_crop_step": crop_step_num,
                                    "cropped_ground_truth": cropped_output if output_grid else None,
                                    "cropped_output": cropped_output  # Also save as cropped_output for streamlit
                                })
                                
                                # Uncropped back to full grid
                                current_full_grid = self.uncrop_to_full_grid(
                                    transformed_crop, current_full_grid, input_metadata,
                                    input_grid=input_grid,
                                    processed_objects=list(processed_object_indices)
                                )
                                
                                processed_object_indices.add(obj_idx)
                                if accuracy and accuracy >= 0.95:
                                    successfully_transformed.add(obj_idx)
                                elif is_test:
                                    # In test mode, mark as successfully transformed if we got a result
                                    successfully_transformed.add(obj_idx)
                                
                                step_num += 1
                                uncrop_accuracy = None
                                if output_grid:
                                    uncrop_accuracy = self.compare_grids(current_full_grid, output_grid)
                                    print(f"  After uncrop, accuracy: {uncrop_accuracy:.1%}")
                                else:
                                    print(f"  After uncrop (test mode)")
                                
                                all_step_results.append({
                                    "step_num": step_num,
                                    "description": f"UNCROP: Object {obj_idx + 1} back to full grid",
                                    "grid": [row[:] for row in current_full_grid],
                                    "object_num": obj_idx + 1,
                                    "is_crop_step": False,
                                    "is_uncrop_step": True,
                                    "accuracy": uncrop_accuracy if uncrop_accuracy is not None else (accuracy if accuracy else None),
                                    "parent_crop_step": crop_step_num,
                                    "processed_objects": list(processed_object_indices),
                                    "uncrop_ground_truth": output_grid if output_grid else None
                                })
                                
                                if output_grid and uncrop_accuracy and uncrop_accuracy >= 0.99:
                                    print(f"\n{'='*80}")
                                    print(f"✓ SUCCESS: Grid matches output!")
                                    print(f"  Accuracy: {uncrop_accuracy:.1%}")
                                    print(f"{'='*80}\n")
                                    all_objects_processed = True
                                    break
                                
                                success = True
                                break
                    
                    if not success:
                        print(f"  ⚠️ Failed after 3 attempts")
                        if output_grid:
                            current_full_grid = self.uncrop_to_full_grid(
                                cropped_output, current_full_grid, input_metadata,
                                input_grid=input_grid,
                                processed_objects=list(processed_object_indices)
                            )
                            processed_object_indices.add(obj_idx)
                            
                            step_num += 1
                            all_step_results.append({
                                "step_num": step_num,
                                "description": f"Transform object {obj_idx + 1} (using GT)",
                                "grid": [row[:] for row in current_full_grid],
                                "object_num": obj_idx + 1,
                                "is_crop_step": False,
                                "used_ground_truth": True,
                                "processed_objects": list(processed_object_indices)
                            })
                        else:
                            print(f"  ⚠️ Skipping (test mode)")
                            processed_object_indices.add(obj_idx)
                
                elif output_obj_idx is None and output_grid:
                    # Removed object
                    print(f"  ⚠️ REMOVED - clearing from grid")
                    input_obj = input_objects[obj_idx]
                    bbox = input_obj.get('bbox')
                    
                    if bbox:
                        min_r, min_c, max_r, max_c = bbox
                        for r in range(min_r, max_r + 1):
                            for c in range(min_c, max_c + 1):
                                if 0 <= r < len(current_full_grid) and 0 <= c < len(current_full_grid[0]):
                                    if current_full_grid[r][c] == input_obj.get('color'):
                                        current_full_grid[r][c] = 0
                        
                        processed_object_indices.add(obj_idx)
                        
                        step_num += 1
                        all_step_results.append({
                            "step_num": step_num,
                            "description": f"REMOVE: Object {obj_idx + 1} ({input_obj.get('description', 'object')})",
                            "grid": [row[:] for row in current_full_grid],
                            "object_num": obj_idx + 1,
                            "is_crop_step": False,
                            "is_removal_step": True,
                            "processed_objects": list(processed_object_indices)
                        })
                        
                        if output_grid:
                            removal_accuracy = self.compare_grids(current_full_grid, output_grid)
                            print(f"  After removal, accuracy: {removal_accuracy:.1%}")
                            
                            if removal_accuracy >= 0.99:
                                print(f"\n{'='*80}")
                                print(f"✓ SUCCESS: Grid matches output!")
                                print(f"{'='*80}\n")
                                all_objects_processed = True
                else:
                    print(f"  ⚠️ Unexpected state - marking as processed")
                    processed_object_indices.add(obj_idx)
                
                if all_objects_processed:
                    break
            
            if all_objects_processed:
                break
        
        # Handle new objects (training mode) or predict new objects (test mode)
        if not all_objects_processed:
            if output_grid and new_output_objects:
                # Training mode: create new objects from output
                for output_obj_idx in new_output_objects:
                    output_obj = output_objects[output_obj_idx]
                    print(f"\n  NEW OBJECT: {output_obj.get('description', 'N/A')}")
                    
                    step_num += 1
                    cropped_output, output_metadata = self.crop_to_object(output_grid, output_obj['bbox'])
                    
                    all_step_results.append({
                        "step_num": step_num,
                        "description": f"CROP: New object region ({output_obj.get('description', 'object')})",
                        "grid": [row[:] for row in current_full_grid],
                        "object_num": None,
                        "is_crop_step": True,
                        "is_new_object": True,
                        "crop_metadata": output_metadata,
                        "object_description": output_obj.get('description', ''),
                        "processed_objects": list(processed_object_indices)
                    })
                    
                    blank_crop = [[0] * len(cropped_output[0]) for _ in range(len(cropped_output))]
                    
                    for attempt in range(1, 4):
                        description, generated_crop = self.generate_object_transformation(
                            blank_crop, cropped_output, len(input_objects) + 1, 
                            len(input_objects) + len(new_output_objects), None, attempt,
                            is_test=False, all_training_examples=all_training_examples
                        )
                        
                        if generated_crop:
                            accuracy = self.compare_grids(generated_crop, cropped_output)
                            print(f"    Accuracy: {accuracy:.1%}")
                            if accuracy >= 0.95:
                                step_num += 1
                                all_step_results.append({
                                    "step_num": step_num,
                                    "description": description or f"Create new object",
                                    "grid": generated_crop,
                                    "object_num": None,
                                    "is_crop_step": False,
                                    "is_cropped_view": True,
                                    "is_new_object": True,
                                    "accuracy": accuracy,
                                    "parent_crop_step": step_num - 1,
                                    "cropped_ground_truth": cropped_output
                                })
                                
                                current_full_grid = self.uncrop_to_full_grid(
                                    generated_crop, current_full_grid, output_metadata,
                                    input_grid=input_grid,
                                    processed_objects=list(processed_object_indices)
                                )
                                
                                step_num += 1
                                uncrop_accuracy = None
                                if output_grid:
                                    uncrop_accuracy = self.compare_grids(current_full_grid, output_grid)
                                    print(f"  After uncrop, accuracy: {uncrop_accuracy:.1%}")
                                
                                all_step_results.append({
                                    "step_num": step_num,
                                    "description": f"UNCROP: New object back to full grid",
                                    "grid": [row[:] for row in current_full_grid],
                                    "object_num": None,
                                    "is_crop_step": False,
                                    "is_uncrop_step": True,
                                    "is_new_object": True,
                                    "accuracy": uncrop_accuracy if uncrop_accuracy is not None else accuracy,
                                    "parent_crop_step": step_num - 2,
                                    "processed_objects": list(processed_object_indices),
                                    "uncrop_ground_truth": output_grid if output_grid else None
                                })
                                
                                if output_grid and uncrop_accuracy and uncrop_accuracy >= 0.99:
                                    print(f"\n{'='*80}")
                                    print(f"✓ SUCCESS: Grid matches output!")
                                    print(f"{'='*80}\n")
                                    all_objects_processed = True
                                
                                break
        
        # Final verification (training mode) or final step (test mode)
        if output_grid:
            # Training mode: verify completion
            final_accuracy = self.compare_grids(current_full_grid, output_grid)
            print(f"\n{'='*80}")
            print(f"FINAL VERIFICATION")
            print(f"  Accuracy: {final_accuracy:.1%}")
            
            if final_accuracy < 0.99:
                # Analyze what cells are different and determine if final transition is needed
                print(f"\n  Analyzing differences before final step...")
                diff_cells = []
                diff_by_color = {}
                diff_by_position = {"rows": set(), "cols": set()}
                
                for r in range(min(len(current_full_grid), len(output_grid))):
                    for c in range(min(len(current_full_grid[0]) if current_full_grid else 0,
                                     len(output_grid[0]) if output_grid else 0)):
                        if current_full_grid[r][c] != output_grid[r][c]:
                            diff_cells.append({
                                "row": r,
                                "col": c,
                                "current": current_full_grid[r][c],
                                "expected": output_grid[r][c]
                            })
                            # Track color changes
                            color_change = f"{current_full_grid[r][c]}→{output_grid[r][c]}"
                            diff_by_color[color_change] = diff_by_color.get(color_change, 0) + 1
                            diff_by_position["rows"].add(r)
                            diff_by_position["cols"].add(c)
                
                print(f"  Found {len(diff_cells)} differing cells")
                if diff_by_color:
                    print(f"  Color changes:")
                    for change, count in sorted(diff_by_color.items(), key=lambda x: x[1], reverse=True)[:5]:
                        print(f"    {change}: {count} cells")
                
                # Determine if there's a pattern/transition needed
                needs_transition = False
                transition_description = ""
                
                if len(diff_cells) > 0:
                    # Check if differences form a pattern
                    if len(diff_by_color) == 1:
                        # Single color change pattern
                        color_change = list(diff_by_color.keys())[0]
                        needs_transition = True
                        transition_description = f"Color change transition: {color_change} ({len(diff_cells)} cells)"
                    elif len(diff_by_color) <= 3:
                        # Few color changes - might be a simple transition
                        needs_transition = True
                        transition_description = f"Multiple color transitions: {', '.join(diff_by_color.keys())}"
                    
                    # Check if differences are localized (suggesting a specific object/region needs fixing)
                    if len(diff_by_position["rows"]) <= 5 and len(diff_by_position["cols"]) <= 5:
                        needs_transition = True
                        if not transition_description:
                            transition_description = f"Localized region needs correction ({len(diff_cells)} cells)"
                    
                    # Check if it's a systematic pattern
                    if len(diff_cells) > len(current_full_grid) * len(current_full_grid[0]) * 0.1:
                        needs_transition = True
                        if not transition_description:
                            transition_description = f"Systematic pattern correction needed ({len(diff_cells)} cells)"
                
                if needs_transition:
                    print(f"  ✓ Final transition needed: {transition_description}")
                else:
                    print(f"  ⚠️ Differences detected but no clear transition pattern")
                
                step_num += 1
                
                # Build detailed prompt with difference analysis
                diff_summary = f"""
DIFFERENCE ANALYSIS:
- Total differing cells: {len(diff_cells)}
- Color changes: {', '.join([f"{k} ({v} cells)" for k, v in sorted(diff_by_color.items(), key=lambda x: x[1], reverse=True)[:3]])}
- Affected rows: {sorted(diff_by_position['rows'])[:10]}{'...' if len(diff_by_position['rows']) > 10 else ''}
- Affected columns: {sorted(diff_by_position['cols'])[:10]}{'...' if len(diff_by_position['cols']) > 10 else ''}

Sample differences (first 10):
"""
                for i, diff in enumerate(diff_cells[:10]):
                    diff_summary += f"  Cell ({diff['row']}, {diff['col']}): color {diff['current']} → color {diff['expected']}\n"
                
                if needs_transition:
                    diff_summary += f"\nTRANSITION NEEDED: {transition_description}\n"
                
                final_prompt = f"""FINAL COMPLETION STEP

CURRENT STATE:
{self._format_grid(current_full_grid)}

EXPECTED OUTPUT:
{self._format_grid(output_grid)}

{diff_summary}

TASK: Apply the final transition to correct the differences.
- Focus on the specific cells that differ
- Apply the identified transition pattern
- Ensure all cells match expected output

{self._get_base_instructions()}
- Grid size: {len(output_grid)}×{len(output_grid[0])}
- Accuracy target: 100%"""
                
                final_images = [
                    grid_to_image(current_full_grid, 50),
                    grid_to_image(output_grid, 50)
                ]
                
                description, final_grid = self.call_api(final_prompt, final_images, current_full_grid, None, None)
                
                if final_grid:
                    final_final_accuracy = self.compare_grids(final_grid, output_grid)
                    print(f"  Final step accuracy: {final_final_accuracy:.1%}")
                    
                    all_step_results.append({
                        "step_num": step_num,
                        "description": description or "Final completion step",
                        "grid": final_grid,
                        "object_num": None,
                        "is_crop_step": False,
                        "is_final_step": True,
                        "accuracy": final_final_accuracy
                    })
                    current_full_grid = final_grid
                else:
                    print(f"  ⚠️ Using expected output")
                    all_step_results.append({
                        "step_num": step_num,
                        "description": "Final completion (using expected output)",
                        "grid": [row[:] for row in output_grid],
                        "object_num": None,
                        "is_crop_step": False,
                        "is_final_step": True,
                        "used_ground_truth": True,
                        "accuracy": 1.0
                    })
                    current_full_grid = [row[:] for row in output_grid]
            else:
                print(f"  ✓ Grid is complete!")
        else:
            # Test mode: add final step to complete the transformation
            print(f"\n{'='*80}")
            print(f"FINAL STEP (TEST MODE)")
            print(f"{'='*80}\n")
            
            step_num += 1
            final_prompt = f"""FINAL COMPLETION STEP (TEST MODE)

Apply the transformation pattern learned from training examples to complete the test input.

CURRENT STATE:
{self._format_grid(current_full_grid)}

TRAINING PATTERN:
Based on the training examples, apply any final adjustments needed to complete the transformation following the same pattern.

{self._get_base_instructions()}
- Grid size: {len(current_full_grid)}×{len(current_full_grid[0])}
- Follow the same transformation pattern as training examples
- Complete any remaining transformations"""
            
            training_images = []
            for ex in all_training_examples[:2]:  # Show 2 training examples for reference
                training_images.append(grid_to_image(ex['input'], 40))
                training_images.append(grid_to_image(ex['output'], 40))
            
            final_images = [grid_to_image(current_full_grid, 50)]
            final_images.extend(training_images)
            
            description, final_grid = self.call_api(final_prompt, final_images, current_full_grid, None, None)
            
            if final_grid:
                print(f"  Generated final grid")
                all_step_results.append({
                    "step_num": step_num,
                    "description": description or "Final completion step (test mode)",
                    "grid": final_grid,
                    "object_num": None,
                    "is_crop_step": False,
                    "is_final_step": True,
                    "is_test_mode": True
                })
                current_full_grid = final_grid
            else:
                print(f"  ⚠️ Final step generation failed, using current state")
                all_step_results.append({
                    "step_num": step_num,
                    "description": "Final state (test mode)",
                    "grid": [row[:] for row in current_full_grid],
                    "object_num": None,
                    "is_crop_step": False,
                    "is_final_step": True,
                    "is_test_mode": True
                })
        
        # Save results
        self._save_results(puzzle_id, training_num, shared_analysis or "", 
                          all_step_results, input_grid, output_grid, is_test=is_test)
        
        print(f"\n{'='*80}")
        print(f"✅ COMPLETE: Generated {len(all_step_results)} steps")
        print(f"{'='*80}\n")
    
    def _load_or_generate_generalized_steps(self, puzzle_id: str, all_training_examples: List[Dict]) -> Dict:
        """Load existing generalized steps or generate them from training examples"""
        patterns_file = Path("visual_step_results") / puzzle_id / "generalized_steps" / "generalized_patterns.json"
        
        # Try to load existing
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                    print(f"✓ Loaded existing generalized steps from: {patterns_file}")
                    return existing
            except Exception as e:
                print(f"  ⚠️ Error loading existing generalized steps: {e}")
        
        # Generate new generalized steps
        print(f"\n{'='*80}")
        print("GENERATING GENERALIZED STEPS FROM ALL TRAINING EXAMPLES")
        print(f"{'='*80}\n")
        
        # Comprehensive analysis first
        comprehensive_analysis = self.comprehensive_analysis(all_training_examples)
        self._current_comprehensive_analysis = comprehensive_analysis
        
        # Generate transformation rule
        print("Analyzing transformation rule from training examples...")
        transformation_rule = self.analyze_transformation_rule(all_training_examples)
        
        # Generate generalized steps with full tool awareness
        print("\nGenerating generalized step sequence with full tool awareness...")
        generalized_step_sequence = self.generate_generalized_steps_v7(all_training_examples, comprehensive_analysis, transformation_rule)
        
        # Generate transition determinants (CONDITION, OBJECT) → transformations
        print("\nGenerating transition determinants...")
        transition_determinants = self.generate_transition_determinants_v7(all_training_examples, comprehensive_analysis)
        
        # Generate booklet pattern
        print("\nGenerating booklet pattern...")
        booklet_pattern = self.generate_booklet_pattern_v7(generalized_step_sequence, transition_determinants)
        
        # Save
        self._save_generalized_booklet(puzzle_id, transition_determinants, 
                                      generalized_step_sequence, booklet_pattern, 
                                      transformation_rule)
        
        return {
            "transition_determinants": transition_determinants,
            "generalized_step_sequence": generalized_step_sequence,
            "booklet_pattern": booklet_pattern,
            "transformation_rule": transformation_rule
        }
    
    def _save_generalized_booklet(self, puzzle_id: str, transition_determinants: Dict,
                                 generalized_step_sequence: List[Dict], booklet_pattern: Dict,
                                 transformation_rule: Dict = None):
        """Save generalized booklet/pattern to file"""
        output_dir = Path("visual_step_results") / puzzle_id / "generalized_steps"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generalized_data = {
            "puzzle_id": puzzle_id,
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "version": "v7",
            "transition_determinants": self._clean_for_json(transition_determinants) if transition_determinants else {},
            "generalized_step_sequence": self._clean_for_json(generalized_step_sequence) if generalized_step_sequence else [],
            "booklet_pattern": self._clean_for_json(booklet_pattern) if booklet_pattern else {},
            "transformation_rule": self._clean_for_json(transformation_rule) if transformation_rule else {}
        }
        
        try:
            with open(output_dir / "generalized_patterns.json", 'w', encoding='utf-8') as f:
                json.dump(generalized_data, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Saved generalized booklet/pattern to: {output_dir / 'generalized_patterns.json'}")
        except Exception as e:
            print(f"  ⚠️ Error saving generalized booklet: {e}")
    
    def _clean_for_json(self, obj):
        """Recursively clean object for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return int(obj) if isinstance(obj, np.integer) else float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            # Skip complex objects that can't be serialized
            return str(obj)
        elif isinstance(obj, (Image.Image, bytes)):
            # Skip PIL Images and bytes
            return None
        else:
            try:
                # Test if it's JSON serializable
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                # Convert to string if not serializable
                return str(obj)
    
    def _save_results(self, puzzle_id: str, training_num: int, analysis: str,
                     results: List[Dict], input_grid: List[List[int]], 
                     output_grid: List[List[int]], is_test: bool = False):
        """Save results in same format as v2/v3"""
        example_type = "testing" if is_test else "training"
        output_dir = Path("visual_step_results") / puzzle_id / f"{example_type}_{training_num:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean results for JSON serialization (remove non-serializable objects)
        cleaned_results = []
        for step in results:
            cleaned_step = {}
            for key, value in step.items():
                # Skip complex objects that are saved as images
                if key in ['reference_crop', 'cropped_input', 'cropped_output']:
                    continue  # These are saved as images, not in JSON
                cleaned_step[key] = self._clean_for_json(value)
            cleaned_results.append(cleaned_step)
        
        # Save JSON
        data = {
            "puzzle_id": puzzle_id,
            "training_num": training_num,
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "version": "v7",
            "phase1_analysis": analysis if isinstance(analysis, str) else str(analysis),
            "input_grid": input_grid,
            "expected_output_grid": output_grid,
            "steps": cleaned_results
        }
        
        # Clean the entire data structure
        data = self._clean_for_json(data)
        
        try:
            with open(output_dir / "results.json", 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  ⚠️ Error saving JSON: {e}")
            # Try saving without indentation (smaller file)
            try:
                with open(output_dir / "results.json", 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
            except Exception as e2:
                print(f"  ❌ Failed to save JSON: {e2}")
                # Save a minimal version
                minimal_data = {
                    "puzzle_id": puzzle_id,
                    "training_num": training_num,
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model,
                    "version": "v7",
                    "num_steps": len(results)
                }
                with open(output_dir / "results.json", 'w', encoding='utf-8') as f:
                    json.dump(minimal_data, f, indent=2, ensure_ascii=False)
        
        # Save images
        input_img = grid_to_image(input_grid, 30)
        input_img.save(output_dir / "input.png")
        
        if output_grid:
            output_img = grid_to_image(output_grid, 30)
            output_img.save(output_dir / "expected_output.png")
        
        for step_result in results:
            step_num = step_result['step_num']
            grid = step_result['grid']
            
            # Save main step image
            img = grid_to_image(grid, 30)
            img.save(output_dir / f"step_{step_num:02d}_final.png")
            
            # Save ground truth images (skip for test examples)
            if not is_test and output_grid:
                if step_result.get('is_cropped_view'):
                    cropped_gt = step_result.get('cropped_ground_truth')
                    if cropped_gt:
                        cropped_gt_img = grid_to_image(cropped_gt, 50)
                        cropped_gt_img.save(output_dir / f"step_{step_num:02d}_ground_truth.png")
                elif step_result.get('is_crop_step'):
                    cropped_input = step_result.get('cropped_input')
                    if cropped_input:
                        cropped_input_img = grid_to_image(cropped_input, 50)
                        cropped_input_img.save(output_dir / f"step_{step_num:02d}_ground_truth.png")
                elif step_result.get('is_reference_step'):
                    ref_crop = step_result.get('reference_crop')
                    if ref_crop:
                        ref_img = grid_to_image(ref_crop, 50)
                        ref_img.save(output_dir / f"step_{step_num:02d}_ground_truth.png")
            elif step_num == 1:
                input_img = grid_to_image(input_grid, 30)
                input_img.save(output_dir / f"step_{step_num:02d}_ground_truth.png")
            elif step_result.get('is_uncrop_step'):
                if output_grid:
                    uncrop_gt = step_result.get('uncrop_ground_truth')
                    if uncrop_gt:
                        uncrop_gt_img = grid_to_image(uncrop_gt, 30)
                        uncrop_gt_img.save(output_dir / f"step_{step_num:02d}_ground_truth.png")
                    else:
                        uncrop_gt_img = grid_to_image(output_grid, 30)
                        uncrop_gt_img.save(output_dir / f"step_{step_num:02d}_ground_truth.png")
            elif step_result.get('is_final_step'):
                if output_grid:
                    final_gt_img = grid_to_image(output_grid, 30)
                    final_gt_img.save(output_dir / f"step_{step_num:02d}_ground_truth.png")
            
            # Save reference visualization
            if step_result.get('is_reference_step'):
                ref_crop = step_result.get('reference_crop')
                if ref_crop:
                    ref_img = grid_to_image(ref_crop, 50)
                    ref_img.save(output_dir / f"step_{step_num:02d}_reference.png")
            
            # Save cropped images
            if step_result.get('is_crop_step') and step_result.get('cropped_input'):
                cropped_input = step_result['cropped_input']
                cropped_output = step_result.get('cropped_output')
                
                crop_img = grid_to_image(cropped_input, 50)
                crop_img.save(output_dir / f"step_{step_num:02d}_crop_input.png")
                
                if cropped_output:
                    crop_target_img = grid_to_image(cropped_output, 50)
                    crop_target_img.save(output_dir / f"step_{step_num:02d}_crop_target.png")
            
            # Save cropped transformation
            if step_result.get('is_cropped_view'):
                crop_transform_img = grid_to_image(grid, 50)
                crop_transform_img.save(output_dir / f"step_{step_num:02d}_crop_transform.png")
                
                cropped_gt = step_result.get('cropped_ground_truth')
                if cropped_gt:
                    cropped_gt_img = grid_to_image(cropped_gt, 50)
                    cropped_gt_img.save(output_dir / f"step_{step_num:02d}_crop_ground_truth.png")
            
            # Save crop metadata
            if step_result.get('is_crop_step') and step_result.get('crop_metadata'):
                metadata_file = output_dir / f"step_{step_num:02d}_crop_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(step_result['crop_metadata'], f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"SAVED TO: {output_dir}/")
        print(f"{'='*80}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--puzzle', required=True, help='Puzzle ID')
    parser.add_argument('--model', default='gpt-5-mini', help='Model to use')
    parser.add_argument('--training', type=int, default=1, help='Training example number')
    parser.add_argument('--all', action='store_true', help='Process all training examples')
    parser.add_argument('--test', type=int, help='Test example number (1-indexed)')
    parser.add_argument('--test-all', action='store_true', help='Process all test examples')
    args = parser.parse_args()
    
    gen = VisualStepGeneratorV7(args.model)
    
    if args.test_all:
        import importlib.util
        spec = importlib.util.spec_from_file_location("visual_step_generator", 
                                                      Path(__file__).parent / "visual_step_generator.py")
        v2_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v2_module)
        v2_gen = v2_module.VisualStepGenerator(args.model)
        arc = v2_gen.load_arc_puzzle(args.puzzle)
        
        all_training_examples = arc['train']
        test_examples = arc.get('test', [])
        puzzle_analysis = gen.analyze_puzzle_pattern(all_training_examples)
        
        for test_num in range(1, len(test_examples) + 1):
            try:
                gen.run(args.puzzle, test_num, shared_analysis=puzzle_analysis, is_test=True)
            except Exception as e:
                print(f"\n❌ Error processing testing_{test_num:02d}: {e}\n")
                import traceback
                traceback.print_exc()
                continue
    elif args.test:
        gen.run(args.puzzle, args.test, is_test=True)
    elif args.all:
        import importlib.util
        spec = importlib.util.spec_from_file_location("visual_step_generator", 
                                                      Path(__file__).parent / "visual_step_generator.py")
        v2_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v2_module)
        v2_gen = v2_module.VisualStepGenerator(args.model)
        arc = v2_gen.load_arc_puzzle(args.puzzle)
        
        training_nums = list(range(1, len(arc['train']) + 1))
        
        for training_num in training_nums:
            try:
                gen.run(args.puzzle, training_num)
            except Exception as e:
                print(f"\n❌ Error processing training_{training_num:02d}: {e}\n")
                import traceback
                traceback.print_exc()
                continue
    else:
        gen.run(args.puzzle, args.training)


if __name__ == "__main__":
    main()

