#!/usr/bin/env python3
"""
ARC Comprehensive Solver V10
Visual-first analysis with GPT-4o-mini, following v4/v6 patterns
"""

import json
import os
import sys
import base64
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from io import BytesIO

from openai import OpenAI
from PIL import Image
import numpy as np

# Import visualizer
sys.path.insert(0, str(Path(__file__).parent))
from arc_visualizer import grid_to_image, ARC_COLORS


class ARCComprehensiveSolverV10:
    """V10: Visual-first comprehensive solver with LLM analysis"""
    
    def __init__(self, model: str = "gpt-5-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.tools = self._define_tools()
        print(f"[OK] Initialized V10 with {model}")
    
    def _define_tools(self) -> List[Dict]:
        """Define MCP tools for transformations"""
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
                                "description": "1-2 sentences explaining what this step does based on visual and grid analysis"
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
                    "description": "Detect all distinct OBJECTS in a grid. CRITICAL: Objects are distinct connected regions. Each separate filled region is a DISTINCT object, even if same color. The background can be any color - objects are the distinct regions that get transformed. Lines follow different rules - sequential, can change color by section.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "objects": {
                                "type": "array",
                                "description": "List of detected distinct objects (connected regions). Each object is a separate entity that can be transformed independently.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "bbox": {
                                            "type": "array",
                                            "description": "Bounding box [min_row, min_col, max_row, max_col] - 0-indexed, should only include the object cells",
                                            "items": {"type": "integer"}
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
                                        "input_idx": {"type": "integer"},
                                        "output_idx": {"type": "integer", "nullable": True},
                                        "reason": {"type": "string"}
                                    },
                                    "required": ["input_idx", "output_idx", "reason"]
                                }
                            }
                        },
                        "required": ["matches"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "crop",
                    "description": "Crop grid to a specific region. Returns cropped grid and metadata.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "grid": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "integer"}}
                            },
                            "bbox": {
                                "type": "array",
                                "description": "Bounding box [min_row, min_col, max_row, max_col]",
                                "items": {"type": "integer"}
                            }
                        },
                        "required": ["grid", "bbox"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "transform",
                    "description": "Transform the DISTINCT OBJECT in a cropped grid. CRITICAL: Transform the distinct object region, NOT the background/surrounding cells. The background/surrounding cells should typically remain unchanged. Provide the transformed grid with the distinct object transformed. Transition name must match the analysis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "grid": {
                                "type": "array",
                                "description": "The transformed grid after applying the transformation to the DISTINCT OBJECT. Background/surrounding cells should typically remain unchanged unless the transformation specifically requires changing them.",
                                "items": {"type": "array", "items": {"type": "integer"}}
                            },
                            "transformation_type": {
                                "type": "string",
                                "description": "Type of transformation: color_mapping, rotate, flip_horizontal, flip_vertical, tile, etc. Must match transition name from analysis exactly."
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Transformation-specific parameters (optional)"
                            }
                        },
                        "required": ["grid", "transformation_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "uncrop",
                    "description": "Place transformed grid back into full-size grid at original position",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cropped_grid": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "integer"}}
                            },
                            "original_grid": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "integer"}}
                            },
                            "bbox": {
                                "type": "array",
                                "description": "Original bounding box where cropped grid came from",
                                "items": {"type": "integer"}
                            }
                        },
                        "required": ["cropped_grid", "original_grid", "bbox"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_objects",
                    "description": "Create new objects by copying or generating them. Similar to copy operation - creates objects at specified positions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "grid": {
                                "type": "array",
                                "description": "The grid after creating new objects",
                                "items": {"type": "array", "items": {"type": "integer"}}
                            },
                            "objects": {
                                "type": "array",
                                "description": "List of objects created",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "bbox": {"type": "array", "items": {"type": "integer"}},
                                        "source": {"type": "string", "description": "Where object came from (copy, generate, etc.)"}
                                    }
                                }
                            },
                            "visual_analysis": {
                                "type": "string",
                                "description": "1-2 sentences explaining what objects were created and why"
                            }
                        },
                        "required": ["grid", "objects", "visual_analysis"]
                    }
                }
            }
        ]
    
    def _image_to_base64(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def _create_visual_analysis_prompt(self, train_examples: List[Dict], test_input: Optional[List[List[int]]] = None) -> Tuple[str, List[Dict]]:
        """Create prompt for visual-first comprehensive analysis following the complete guide"""
        
        # Calculate size relationships
        size_info = []
        for i, ex in enumerate(train_examples):
            input_dims = (len(ex['input']), len(ex['input'][0]))
            output_dims = (len(ex['output']), len(ex['output'][0]))
            input_size = input_dims[0] * input_dims[1]
            output_size = output_dims[0] * output_dims[1]
            size_info.append(f"Example {i+1}: Input {input_dims} (size {input_size}) → Output {output_dims} (size {output_size})")
        
        content = [
            {
                "type": "text",
                "text": f"""You are analyzing ARC puzzles. Perform COMPREHENSIVE visual analysis first, then grid analysis for EACH STEP.

CRITICAL: Do visual analysis FIRST, then grid analysis. Use BOTH for each step.

=== STEP 1: PUZZLE TYPE IDENTIFICATION ===

Size Relationships:
{chr(10).join(size_info)}

1. Identify puzzle type based on size differences:
   - Input Size > Output Size: Pattern Extraction/Cropping
     * 73.7% pattern extraction, 71.4% cropping
     * May need to crop if output size smaller
     * Something in input indicates getting to output size
   - Input Size = Output Size: Same-Size Transformation
     * 46.5% color mapping, 0.4% spatial transforms, 0.5% identity
     * Reference objects will exist, stay similar location/color between inputs
   - Input Size < Output Size: Expansion/Tiling
     * 74.7% expansion, 63.3% tiling
     * May repeat input on output (copy/paste)
     * Look for input pattern repeated, reflected, or shown in output

2. Check if input/output ratio varies between training samples:
   - If ratio varies: something in puzzle corresponds to output size - determine that first
   - If ratio constant: standard transformation

3. Guess what kind of grid to start with:
   - Pretty much always start with input size
   - But need to crop if output size smaller
   - Or something in input is indicative of getting to size of output

=== STEP 2: COMPREHENSIVE COMPARISONS ===

1. Compare ALL INPUTS against each other (INCLUDING TEST INPUT if provided):
   - What do differences in inputs say about differences in outputs?
   - What is same among inputs? Are similar objects going through similar transitions?
   - What is different about inputs? How are they giving more information about transition steps?
   - Do they build upon each other?
   - Is there a correspondence (if-then) being built upon?

2. Compare ALL OUTPUTS against each other:
   - Anything consistent in all training outputs (exact same) should be in test output
   - What is common across all outputs?
   - What patterns emerge?

3. Compare INPUT TO OUTPUT for EACH TRAINING EXAMPLE INDIVIDUALLY (NOT test input):
   - What is similar between all inputs and outputs?
   - What is common in transition steps from input to output?
   - What is changed?
   - Analyze shape, color, size, location of objects
   - Compare training sample analyses

=== STEP 3: REFERENCE OBJECTS (CRITICAL) ===

Focus on Reference Objects:
- Objects that stay the same input-to-input OR input-to-output
- Object types, object transitions
- Reference objects can be used for shape or color
- Sometimes order in reference object matters (left to right, up to down)
- Reference objects can be a bar of different color that divides two parts
  * Either common quality on both sides
  * Or edit one side based on other side (based on conditions)
- If they move, why do they move? What does it say in transition?
- If no solid reference object, something about input/output tells location, color, or shape
- Generally what stays same will be same:
  * Input to output they stay same in training → input objects in output
  * Objects same output-to-output → usually in test output
- When reference object same color/shape/pattern/location/size as one or more objects in puzzle:
  * What changes about this/these object(s)?

=== STEP 4: WHOLE GRID ANALYSIS ===

Using the WHOLE GRID to determine why output is output (per training sample):

1. Patterns in color:
   - Color transitions
   - Color mappings
   - 1-to-1 correlations between shapes and colors, or colors and shapes

2. Patterns in shape:
   - Shape transformations
   - Awareness of sized up/down objects
   - When object is scaled version of another (same grid or input to output)
   - Reshaping: "rectangularizing" or making fuzzy edges into standard shape

3. Negative space:
   - SEE NEGATIVE SPACE! (color 0)
   - Cells common between parts of input
   - Cells common to only one part of input
   - Get rid of stray cells - difference between objects and stray cells

4. Common cells:
   - Cells that stay same position and color
   - What stays constant?

5. Parts divided by reference object bars:
   - Bars of different color that divide sections
   - Common quality on both sides?
   - Edit one side based on other?

6. Incremental steps:
   - Look for incremental steps building on each other
   - Training sample to training sample: what builds upon what?

=== STEP 5: OBJECT ANALYSIS ===

- Same colored cells are usually one object. Focus on object by object unless lines
- Objects inside objects (or distinct parts of objects can be objects too)
- Usually can find operation to do over and over, maybe more than one transition step when crop
- Lines are different type of object - follow different rules:
  * Sequential
  * Can have starting/ending point indicated by puzzle
  * Can change color by section and turn conditionally
  * Can be drawn in order if multiple lines
  * Lines can connect objects (see output similarities)
- Understanding of dimension: when objects overlap, which is frontmost?
- Something countable about objects: holes, cell length, height, whatever
- Multiple objects same shape or color count: most common (similarly shaped/colored object) is recurring conditional

=== STEP 6: GRID SIZE CHANGE SPECIFICS ===

If Input Size < Output Size (up):
- May repeat input on output (copy/paste)
- Look for cells filled in for patterns, also patterns in background, repeated
- Look for input pattern repeated, reflected, or shown in output
- Line objects section by section (different object when perpendicular even if same color)
- Cells can have 1-1 correlation with scaled up block (say 3x3, 9x9) in output

If Input Size > Output Size (down):
- There may be pattern to complete or some part of input is output repeated
- There may be "zoom", "crop", you may delete objects
- What is important about input? Keep in output? Color? Shape? Pattern?
- What does input tell about location, size, color, patterns about output?
- What does output say about input?

Generally if grid changes:
- Try to overlay input on output or output on input

=== STEP 7: TRANSFORMATION TOOLS ===

Available tools (use via MCP toolcalls):
- select, edit, fill tools
- Large features: shape first then color
- See cell shapes but also see negative space!
- Rotate, center
- Reflect on axis
- Repeat pattern - fill pattern
- CONDITIONS: what objects to include in transition based on condition
- Completing objects: recolor, reshape, or move conditionally
- "Fitting" objects together: finding edges that counter each other
- "Drawing" in object's color
- Lines vs objects: how lines change color or turn, can change in line based on conditional
- Zoom in/out on objects including or not including borders

=== STEP 8: GENERAL RULE ===

Finally: The Rule. 3-5 sentences of general common rule, based on WHOLE analysis.

=== STEP 9: GENERAL STEPS ===

Predict general steps looking at test input (if provided), based on preceding analysis.

Format: "Step x: for each (CONDITION) A, perform transition B"

Where:
- CONDITION: color, location, shape, etc. Something that indicates this A experiences common transition with all other A
- A: object, section of line, section of grid (each 3x3 section, 9x9, etc.) but usually object by object unless no objects
- B: transition name (MUST MATCH toolcall transformation_type EXACTLY)

Each step targets same kind of transition, compares against ground truth for each object.

CRITICAL: Transition names in analysis MUST EXACTLY MATCH toolcall transformation_type names!

=== COLOR REFERENCE (NEVER USE COLOR NAMES) ===

Always refer to colors by NUMBER ONLY: color 0, color 1, etc. NEVER mention color names explicitly.

Color mapping:
- color 0 = Black (background)
- color 1 = Blue  
- color 2 = Red
- color 3 = Green
- color 4 = Yellow
- color 5 = Orange
- color 6 = Magenta/Pink
- color 7 = Light Blue/Cyan
- color 8 = Dark Red/Maroon
- color 9 = Purple

Now analyze the following puzzle with COMPLETE visual-first then grid analysis:"""
            }
        ]
        
        # Add training example images
        images = []
        for i, ex in enumerate(train_examples):
            input_img = grid_to_image(ex['input'], cell_size=40)
            output_img = grid_to_image(ex['output'], cell_size=40)
            
            content.append({
                "type": "text",
                "text": f"\n=== TRAINING EXAMPLE {i+1} ===\nInput dimensions: {len(ex['input'])}x{len(ex['input'][0])}\nOutput dimensions: {len(ex['output'])}x{len(ex['output'][0])}"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(input_img)}"}
            })
            content.append({
                "type": "text",
                "text": "Input Grid (visual above, grid below):"
            })
            content.append({
                "type": "text",
                "text": self._format_grid(ex['input'])
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(output_img)}"}
            })
            content.append({
                "type": "text",
                "text": "Output Grid (visual above, grid below):"
            })
            content.append({
                "type": "text",
                "text": self._format_grid(ex['output'])
            })
        
        # Add test input if provided
        if test_input:
            test_img = grid_to_image(test_input, cell_size=40)
            content.append({
                "type": "text",
                "text": f"\n=== TEST INPUT ===\nDimensions: {len(test_input)}x{len(test_input[0])}"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(test_img)}"}
            })
            content.append({
                "type": "text",
                "text": "Test Input Grid:"
            })
            content.append({
                "type": "text",
                "text": self._format_grid(test_input)
            })
        
        content.append({
            "type": "text",
            "text": """
Now provide your COMPREHENSIVE analysis following ALL steps above. Structure your response EXACTLY as:

=== 1. PUZZLE TYPE IDENTIFICATION ===
- Size relationship: input > output, input = output, or input < output
- Puzzle type name (Pattern Extraction/Cropping, Same-Size Transformation, or Expansion/Tiling)
- Does input/output ratio vary between training samples? If yes, what corresponds to output size?
- Initial grid suggestion: what size to start with? Need to crop? Will expand?

=== 2. VISUAL ANALYSIS (FIRST) ===
For EACH training example, describe what you SEE:
- Shapes, patterns, spatial relationships
- Visual transformations observed
- Negative space patterns
- Object arrangements

=== 3. GRID ANALYSIS (SECOND) ===
For EACH training example, analyze grid data:
- Exact dimensions and size relationships
- Color patterns and counts (use color 0, color 1, etc. - NEVER color names)
- Grid structure analysis
- Precise transformations

=== 4. COMPREHENSIVE COMPARISONS ===

4a. ALL INPUTS vs EACH OTHER (including test input if provided):
- What differences in inputs say about differences in outputs?
- What is same among inputs? Similar objects going through similar transitions?
- What is different? How do they build upon each other?
- Is there correspondence (if-then) being built upon?

4b. ALL OUTPUTS vs EACH OTHER:
- What is consistent in all training outputs (exact same)?
- What patterns emerge?
- What should be in test output?

4c. INPUT TO OUTPUT for EACH TRAINING EXAMPLE INDIVIDUALLY:
- What is similar between inputs and outputs?
- What is common in transition steps?
- What is changed?
- Shape, color, size, location analysis

4d. TRAINING SAMPLE TO TRAINING SAMPLE:
- What builds upon what?
- Incremental steps between examples

=== 5. REFERENCE OBJECTS (FOCUS) ===
- Objects that stay same input-to-input OR input-to-output
- Object types, object transitions
- Reference bars that divide sections
- How reference objects guide transformations
- If reference object same as other objects, what changes about those objects?

=== 6. WHOLE GRID PATTERNS (per training sample) ===
Using WHOLE GRID to determine why output is output:

6a. COLOR PATTERNS:
- Color transitions and mappings
- 1-to-1 correlations between shapes and colors

6b. SHAPE PATTERNS:
- Shape transformations
- Sized up/down objects
- Scaled versions

6c. NEGATIVE SPACE:
- Cells common between parts
- Cells common to only one part
- Stray cells vs objects

6d. COMMON CELLS:
- What stays same position and color?

6e. DIVIDED SECTIONS:
- Parts divided by reference object bars
- Common quality on both sides?
- Edit one side based on other?

6f. INCREMENTAL STEPS:
- How examples build upon each other

=== 7. OBJECT ANALYSIS ===
- Objects identified (object by object unless lines)
- Lines vs objects (different rules)
- Countable aspects (holes, length, height)
- Most common similarly shaped/colored objects

=== 8. GENERAL RULE (3-5 sentences) ===
Based on WHOLE analysis above, provide comprehensive rule.

=== 9. GENERAL STEPS ===
Format: "Step x: for each (CONDITION) A, perform transition B"

List each step. CRITICAL: Transition name (B) MUST EXACTLY MATCH toolcall transformation_type!

Example:
- Step 1: for each (object with color 1) object, perform color_mapping transition
- Step 2: for each (3x3 section) grid_section, perform tile transition

(Transition names like "color_mapping", "tile", "rotate", "flip_horizontal" must match toolcalls exactly)

=== 10. INITIAL GRID DETERMINATION ===
- What size grid to start with?
- Need to crop? If so, how?
- Will expand? If so, how?

CRITICAL REMINDERS:
- Transition names MUST EXACTLY MATCH toolcall transformation_type
- Use color 0, color 1, etc. NEVER color names
- Visual analysis FIRST, then grid analysis
- Use WHOLE GRID to determine why output is output
- Focus on reference objects
- Compare everything comprehensively
"""
        })
        
        # Return both text version and structured content
        text_version = "\n".join([item["text"] for item in content if item["type"] == "text"])
        return text_version, content
    
    def _sanitize_text(self, text: str) -> str:
        """Remove Unicode characters that Windows can't represent in prompts"""
        if not text:
            return text
        # Replace problematic Unicode characters with ASCII equivalents
        replacements = {
            '→': '->', '←': '<-', '↑': '^', '↓': 'v',
            '×': 'x', '÷': '/', '±': '+/-', '≠': '!=', '≤': '<=', '≥': '>=',
            '\u2011': '-', '\u2013': '-', '\u2014': '--',  # Various dashes
            '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',  # Smart quotes
            '\u2026': '...',  # Ellipsis
            '°': 'deg', '²': '^2', '³': '^3',
        }
        result = text
        for unicode_char, ascii_replacement in replacements.items():
            result = result.replace(unicode_char, ascii_replacement)
        return result
    
    def _format_grid(self, grid: List[List[int]]) -> str:
        """Format grid as text"""
        return '\n'.join(['[' + ', '.join(str(cell) for cell in row) + ']' for row in grid])
    
    def _call_llm_analysis(self, content: List[Dict], use_tools: bool = False) -> Tuple[str, Optional[Dict]]:
        """Call LLM for analysis with visual inputs"""
        messages = [{"role": "user", "content": content}]
        
        try:
            # For analysis calls, pass tools but set tool_choice="none" to force text response
            # For tool calls, pass tools and tool_choice="auto"
            call_params = {
                "model": self.model,
                "messages": messages,
                "tools": self.tools  # Always pass tools (like v4/v6)
            }
            
            if use_tools:
                call_params["tool_choice"] = "auto"
            else:
                # Force text response for analysis
                call_params["tool_choice"] = "none"
            
            # Check if gpt-5 model (different parameters)
            if "gpt-5" in self.model:
                call_params["max_completion_tokens"] = 16000  # Very high limit for comprehensive analysis
            else:
                call_params["max_tokens"] = 4000
            
            # Debug: Print content structure
            print(f"[DEBUG] Sending {len(messages)} messages")
            if messages and messages[0].get('content'):
                content_items = messages[0]['content']
                if isinstance(content_items, list):
                    text_items = [item for item in content_items if item.get('type') == 'text']
                    image_items = [item for item in content_items if item.get('type') == 'image_url']
                    print(f"[DEBUG] Content: {len(text_items)} text items, {len(image_items)} image items")
                    if text_items:
                        total_text_len = sum(len(item.get('text', '')) for item in text_items)
                        print(f"[DEBUG] Total text length: {total_text_len} characters")
                    print(f"[DEBUG] Tool choice: {call_params.get('tool_choice', 'not set')}")
            
            print(f"[DEBUG] Calling API (this may take a while for comprehensive analysis)...")
            response = self.client.chat.completions.create(**call_params)
            print(f"[DEBUG] API call completed")
            
            message = response.choices[0].message
            text_response = message.content if message.content else ""
            finish_reason = response.choices[0].finish_reason
            
            # Debug output
            print(f"[DEBUG] Response finish_reason: {finish_reason}")
            print(f"[DEBUG] Response content length: {len(text_response)} characters")
            
            if finish_reason == "length":
                print(f"[WARNING] Response was truncated! Content length: {len(text_response)}")
                if text_response:
                    try:
                        preview = text_response[:500]
                        preview = preview.replace('→', '->').replace('←', '<-').replace('↑', '^').replace('↓', 'v')
                        preview = preview.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '--')
                        preview = preview.replace('\u2018', "'").replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
                        print(f"[DEBUG] First 500 chars of truncated response: {preview}")
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        print(f"[DEBUG] First 500 chars: [Unable to display due to encoding]")
                else:
                    print(f"[ERROR] Response truncated but content is empty - this shouldn't happen!")
            
            # Empty text response is normal when tool calls are made
            if not text_response:
                if message.tool_calls:
                    # This is expected - tool calls don't return text content
                    print(f"[DEBUG] Tool call made (empty text is normal)")
                else:
                    # This is a problem - no text and no tool calls
                    print(f"[WARNING] Empty response from LLM (no text, no tool calls)")
                    print(f"[DEBUG] Message content: {repr(message.content)}")
                    print(f"[DEBUG] Message refusal: {message.refusal}")
                    if hasattr(message, 'refusal') and message.refusal:
                        print(f"[ERROR] Model refused: {message.refusal}")
            
            # Extract tool calls if any
            tool_calls = None
            if message.tool_calls:
                tool_calls = message.tool_calls[0]  # Take first tool call
                print(f"[DEBUG] Tool call detected: {tool_calls.function.name if tool_calls else 'None'}")
            
            return text_response, tool_calls
        except Exception as e:
            print(f"[ERROR] API call failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return "", None
    
    def analyze_puzzle(self, train_examples: List[Dict], test_input: Optional[List[List[int]]] = None) -> Dict[str, Any]:
        """Perform comprehensive visual-first analysis in multiple stages"""
        print("[ANALYSIS] Performing visual-first comprehensive analysis (multi-stage)...")
        
        # Stage 1: Puzzle type, visual analysis, grid analysis
        print("[ANALYSIS STAGE 1/4] Puzzle type, visual & grid analysis...")
        analysis_part1 = self._analyze_stage1_puzzle_type_and_visual(train_examples, test_input)
        
        # Stage 2: Comprehensive comparisons
        print("[ANALYSIS STAGE 2/4] Comprehensive comparisons...")
        analysis_part2 = self._analyze_stage2_comparisons(train_examples, test_input, analysis_part1)
        
        # Stage 3: Reference objects, whole grid patterns, object analysis
        print("[ANALYSIS STAGE 3/4] Reference objects & whole grid patterns...")
        analysis_part3 = self._analyze_stage3_patterns_and_objects(train_examples, test_input, analysis_part1, analysis_part2)
        
        # Parse stage 2 into separate comparison sections (before generating rule)
        comparison_sections = self._parse_comparison_sections(analysis_part2)
        
        # Stage 4: Generate rule (needs all previous analyses)
        print("[ANALYSIS STAGE 4/4] Generating general rule...")
        rule = self._generate_rule(train_examples, test_input, analysis_part1, analysis_part2, analysis_part3, comparison_sections)
        
        # Combine all analyses
        full_analysis_text = f"""{analysis_part1}

{analysis_part2}

{analysis_part3}

=== GENERAL RULE ===
{rule}
"""
        
        print("[OK] Analysis complete")
        
        # comparison_sections already parsed above for rule generation
        
        # Print comparison sections debug info
        if comparison_sections.get('input_input') or comparison_sections.get('output_output') or comparison_sections.get('input_output'):
            print(f"[DEBUG] Parsed comparison sections:")
            if comparison_sections.get('input_input'):
                print(f"  - Input-Input: {len(comparison_sections['input_input'])} chars")
            if comparison_sections.get('output_output'):
                print(f"  - Output-Output: {len(comparison_sections['output_output'])} chars")
            if comparison_sections.get('input_output'):
                print(f"  - Input-Output: {len(comparison_sections['input_output'])} chars")
        else:
            print(f"[DEBUG] No comparison sections parsed from stage 2")
        
        # Parse and structure analysis for better display
        analysis = {
            'raw_analysis': full_analysis_text,
            'train_examples': train_examples,
            'test_input': test_input,
            'structured_analysis': self._parse_analysis_text(full_analysis_text),
            'rule': rule,
            'analysis_part1': analysis_part1,
            'analysis_part2': analysis_part2,
            'analysis_part3': analysis_part3,
            'comparison_sections': comparison_sections  # Add structured comparisons
        }
        
        # Print summary for debugging
        print(f"[DEBUG] Full analysis length: {len(full_analysis_text)} characters")
        print(f"[DEBUG] Part 1: {len(analysis_part1)} chars, Part 2: {len(analysis_part2)} chars, Part 3: {len(analysis_part3)} chars, Rule: {len(rule)} chars")
        
        if analysis['structured_analysis']:
            print(f"[DEBUG] Extracted sections: {list(analysis['structured_analysis'].keys())}")
        
        # Helper function to safely print Unicode strings
        def safe_print(text, max_length=None):
            """Safely print text that may contain Unicode characters"""
            try:
                if max_length:
                    text = text[:max_length]
                # Replace common problematic Unicode characters
                text = text.replace('→', '->').replace('←', '<-').replace('↑', '^').replace('↓', 'v')
                text = text.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '--')
                text = text.replace('\u2018', "'").replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
                print(text)
            except (UnicodeEncodeError, UnicodeDecodeError):
                # Fallback: encode with error handling
                try:
                    print(text.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
                except:
                    print("[Unable to display text due to encoding issues]")
        
        # Print FULL analysis (not just preview)
        if full_analysis_text:
            print(f"\n[ANALYSIS] Full analysis:")
            print("=" * 80)
            try:
                safe_print(full_analysis_text)
            except UnicodeEncodeError:
                # Fallback: encode with errors='replace' to avoid crashes
                safe_text = full_analysis_text.encode('ascii', errors='replace').decode('ascii')
                print(safe_text)
            print("=" * 80)
            print()
        
        return analysis
    
    def _analyze_stage1_puzzle_type_and_visual(self, train_examples: List[Dict], test_input: Optional[List[List[int]]] = None) -> str:
        """Stage 1: Puzzle type identification, visual analysis, and grid analysis"""
        content = []
        
        # Calculate size relationships
        size_info = []
        for i, ex in enumerate(train_examples):
            input_dims = (len(ex['input']), len(ex['input'][0]))
            output_dims = (len(ex['output']), len(ex['output'][0]))
            input_size = input_dims[0] * input_dims[1]
            output_size = output_dims[0] * output_dims[1]
            size_info.append(f"Example {i+1}: Input {input_dims} (size {input_size}) → Output {output_dims} (size {output_size})")
        
        content.append({
            "type": "text",
            "text": self._sanitize_text(f"""Analyze ARC puzzle. Response MUST be under 500 chars. Be CONCISE.

Size: {chr(10).join(size_info)}

ANSWER THESE KEY QUESTIONS:

1. TRANSFORMATION TYPE (1 sentence):
   - Input vs output size? Ratio consistent?

2. PATTERN THAT WORKS FOR ALL EXAMPLES (CRITICAL):
   - What transformation is CONSISTENT across ALL training examples?
   - If pattern only works for one example, it's WRONG

3. KEY CHANGES (per example, 1 sentence each):
   - What changes? What stays the same?
   - Use color numbers (0-9), not names

CRITICAL: The rule MUST work for ALL examples. Focus on consistency.

Provide concise analysis (under 500 chars):""")
        })
        
        # Add training example images
        for i, ex in enumerate(train_examples):
            input_img = grid_to_image(ex['input'], cell_size=40)
            output_img = grid_to_image(ex['output'], cell_size=40)
            
            content.append({
                "type": "text",
                "text": f"\n=== TRAINING EXAMPLE {i+1} ===\nInput dimensions: {len(ex['input'])}x{len(ex['input'][0])}\nOutput dimensions: {len(ex['output'])}x{len(ex['output'][0])}"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(input_img)}"}
            })
            content.append({
                "type": "text",
                "text": "Input Grid:\n" + self._format_grid(ex['input'])
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(output_img)}"}
            })
            content.append({
                "type": "text",
                "text": "Output Grid:\n" + self._format_grid(ex['output'])
            })
        
        # Add test input if provided
        if test_input:
            test_img = grid_to_image(test_input, cell_size=40)
            content.append({
                "type": "text",
                "text": f"\n=== TEST INPUT ===\nDimensions: {len(test_input)}x{len(test_input[0])}"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(test_img)}"}
            })
            content.append({
                "type": "text",
                "text": "Test Input Grid:\n" + self._format_grid(test_input)
            })
        
        content.append({
            "type": "text",
            "text": "\nProvide your analysis for Stage 1 covering puzzle type, visual analysis, and grid analysis for each training example."
        })
        
        analysis_text, _ = self._call_llm_analysis(content, use_tools=False)
        return analysis_text
    
    def _analyze_stage2_comparisons(self, train_examples: List[Dict], test_input: Optional[List[List[int]]], analysis_part1: str) -> str:
        """Stage 2: Comprehensive comparisons"""
        content = []
        
        safe_part1 = self._sanitize_text(analysis_part1[:200] if analysis_part1 else '')
        content.append({
            "type": "text",
            "text": self._sanitize_text(f"""Compare training examples. Response MUST be under 500 chars. TEXT ONLY - no tools.

Previous: {safe_part1}...

ANSWER:

=== INPUT-INPUT COMPARISONS ===
- Key differences? What stays same?

=== OUTPUT-OUTPUT COMPARISONS ===
- What's EXACTLY the same in all outputs? (CRITICAL for test)

=== INPUT-OUTPUT COMPARISONS ===
- What transformation works for ALL examples? (If only one, it's WRONG)

CRITICAL: Rule must work for ALL examples. Use color numbers (0-9).

Provide concise comparison (under 500 chars):""")
        })
        
        # Add all training example images
        for i, ex in enumerate(train_examples):
            input_img = grid_to_image(ex['input'], cell_size=40)
            output_img = grid_to_image(ex['output'], cell_size=40)
            
            content.append({
                "type": "text",
                "text": f"\n=== TRAINING EXAMPLE {i+1} ==="
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(input_img)}"}
            })
            content.append({
                "type": "text",
                "text": "Input:\n" + self._format_grid(ex['input'])
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(output_img)}"}
            })
            content.append({
                "type": "text",
                "text": "Output:\n" + self._format_grid(ex['output'])
            })
        
        # Add test input if provided
        if test_input:
            test_img = grid_to_image(test_input, cell_size=40)
            content.append({
                "type": "text",
                "text": f"\n=== TEST INPUT ==="
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(test_img)}"}
            })
            content.append({
                "type": "text",
                "text": "Test Input:\n" + self._format_grid(test_input)
            })
        
        content.append({
            "type": "text",
            "text": "\nProvide your comprehensive comparison analysis."
        })
        
        analysis_text, _ = self._call_llm_analysis(content, use_tools=False)
        return analysis_text
    
    def _analyze_stage3_patterns_and_objects(self, train_examples: List[Dict], test_input: Optional[List[List[int]]], analysis_part1: str, analysis_part2: str) -> str:
        """Stage 3: Reference objects, whole grid patterns, object analysis"""
        content = []
        
        safe_part1 = self._sanitize_text(analysis_part1[:150] if analysis_part1 else '')
        safe_part2 = self._sanitize_text(analysis_part2[:150] if analysis_part2 else '')
        content.append({
            "type": "text",
            "text": self._sanitize_text(f"""Identify key patterns. Response MUST be under 500 chars.

Previous: {safe_part1}... {safe_part2}...

KEY QUESTIONS:

1. REFERENCE OBJECTS:
   - What stays the same? How does it guide transformation?

2. GRID PATTERNS:
   - What transformation pattern works for ALL examples?

3. OBJECTS:
   - Most important objects? Recurring patterns?

CRITICAL: Pattern must work for ALL examples.

Provide concise analysis (under 500 chars):""")
        })
        
        # Add all training example images
        for i, ex in enumerate(train_examples):
            input_img = grid_to_image(ex['input'], cell_size=40)
            output_img = grid_to_image(ex['output'], cell_size=40)
            
            content.append({
                "type": "text",
                "text": f"\n=== TRAINING EXAMPLE {i+1} ==="
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(input_img)}"}
            })
            content.append({
                "type": "text",
                "text": "Input:\n" + self._format_grid(ex['input'])
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(output_img)}"}
            })
            content.append({
                "type": "text",
                "text": "Output:\n" + self._format_grid(ex['output'])
            })
        
        # Add test input if provided
        if test_input:
            test_img = grid_to_image(test_input, cell_size=40)
            content.append({
                "type": "text",
                "text": f"\n=== TEST INPUT ==="
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(test_img)}"}
            })
            content.append({
                "type": "text",
                "text": "Test Input:\n" + self._format_grid(test_input)
            })
        
        content.append({
            "type": "text",
            "text": "\nProvide your analysis for reference objects, whole grid patterns, and object analysis."
        })
        
        analysis_text, _ = self._call_llm_analysis(content, use_tools=False)
        return analysis_text
    
    def _generate_rule(self, train_examples: List[Dict], test_input: Optional[List[List[int]]], analysis_part1: str, analysis_part2: str, analysis_part3: str, comparison_sections: Optional[Dict[str, str]] = None) -> str:
        """Stage 4: Generate general rule from all analyses"""
        content = []
        
        # Include only key summaries from comparison sections (very concise)
        comparison_text = ""
        if comparison_sections:
            if comparison_sections.get('input_input') or comparison_sections.get('output_output') or comparison_sections.get('input_output'):
                comparison_text = "\n\n=== KEY COMPARISONS ===\n"
                if comparison_sections.get('input_input'):
                    comparison_text += f"Input-Input: {comparison_sections['input_input'][:300]}...\n"
                if comparison_sections.get('output_output'):
                    comparison_text += f"Output-Output: {comparison_sections['output_output'][:300]}...\n"
                if comparison_sections.get('input_output'):
                    comparison_text += f"Input-Output: {comparison_sections['input_output'][:300]}...\n"
        
        # Sanitize analysis text to remove Unicode
        safe_part1 = self._sanitize_text(analysis_part1[:500] if analysis_part1 else '')
        safe_part2 = self._sanitize_text(analysis_part2[:500] if analysis_part2 else '')
        safe_part3 = self._sanitize_text(analysis_part3[:500] if analysis_part3 else '')
        safe_comparison = self._sanitize_text(comparison_text[:300] if comparison_text else '')
        
        content.append({
            "type": "text",
            "text": self._sanitize_text(f"""Generate GENERAL RULE that works for ALL training examples.

Analysis Summary:
- Stage 1: {safe_part1}...
- Stage 2: {safe_part2}...
- Stage 3: {safe_part3}...
{safe_comparison}

CRITICAL REQUIREMENTS:
- Rule MUST work for ALL training examples
- If rule only works for one example, it's WRONG
- Be general enough for test input
- Use color numbers (0-9), not names
- Provide a COMPREHENSIVE rule (5-10 sentences) that fully describes the transformation pattern
- FORMAT THE RULE AS A NUMBERED LIST OF ATOMIC STEPS
- EACH STEP MUST BE IN THE FORMAT: "Step X: IF [condition] THEN [atomic action]"
- Use EXACT conditions (e.g., "color is 2", "is a 3x3 block", "is in corner")
- Use ATOMIC actions (e.g., "change color to 4", "move 1 unit right", "create object from output pattern")

Generate general rule (numbered list of atomic IF/THEN steps):""")
        })
        
        # Add training example images for visual reference
        for i, ex in enumerate(train_examples):
            input_img = grid_to_image(ex['input'], cell_size=40)
            output_img = grid_to_image(ex['output'], cell_size=40)
            
            content.append({
                "type": "text",
                "text": f"\n=== TRAINING EXAMPLE {i+1} ==="
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(input_img)}"}
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(output_img)}"}
            })
            content.append({
                "type": "text",
                "text": f"Input:\n{self._format_grid(ex['input'])}\n\nOutput:\n{self._format_grid(ex['output'])}"
            })
        
        # Add test input image if available
        if test_input:
            test_img = grid_to_image(test_input, cell_size=40)
            content.append({
                "type": "text",
                "text": f"\n=== TEST INPUT ==="
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(test_img)}"}
            })
            content.append({
                "type": "text",
                "text": f"Test Input:\n{self._format_grid(test_input)}"
            })
        
        rule_text, _ = self._call_llm_analysis(content, use_tools=False)
        return rule_text
    
    def _parse_comparison_sections(self, comparison_text: str) -> Dict[str, str]:
        """Parse stage 2 comparison text into input-input, output-output, and input-output sections"""
        sections = {
            'input_input': None,
            'output_output': None,
            'input_output': None
        }
        
        lines = comparison_text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Detect section headers
            if '=== INPUT-INPUT COMPARISONS ===' in line_stripped:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'input_input'
                current_content = []
            elif '=== OUTPUT-OUTPUT COMPARISONS ===' in line_stripped:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'output_output'
                current_content = []
            elif '=== INPUT-OUTPUT COMPARISONS ===' in line_stripped:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'input_output'
                current_content = []
            elif current_section and (line_stripped.startswith('===') or line_stripped.startswith('STAGE')):
                # End of current section, start of new major section
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = None
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _parse_analysis_text(self, text: str) -> Dict[str, Any]:
        """Parse analysis text into structured sections"""
        structured = {}
        
        # Try to extract key sections
        sections = {
            'puzzle_type': None,
            'visual_analysis': None,
            'grid_analysis': None,
            'comparisons': None,
            'reference_objects': None,
            'whole_grid_patterns': None,
            'general_rule': None,
            'general_steps': None
        }
        
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Detect section headers
            if '=== 1. PUZZLE TYPE' in line_stripped or 'PUZZLE TYPE IDENTIFICATION' in line_stripped:
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'puzzle_type'
                current_content = []
            elif '=== 2. VISUAL ANALYSIS' in line_stripped or 'VISUAL ANALYSIS' in line_stripped:
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'visual_analysis'
                current_content = []
            elif '=== 3. GRID ANALYSIS' in line_stripped or 'GRID ANALYSIS' in line_stripped:
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'grid_analysis'
                current_content = []
            elif '=== 4. COMPREHENSIVE COMPARISONS' in line_stripped or 'COMPREHENSIVE COMPARISONS' in line_stripped:
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'comparisons'
                current_content = []
            elif '=== 5. REFERENCE OBJECTS' in line_stripped or 'REFERENCE OBJECTS' in line_stripped:
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'reference_objects'
                current_content = []
            elif '=== 6. WHOLE GRID PATTERNS' in line_stripped or 'WHOLE GRID PATTERNS' in line_stripped:
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'whole_grid_patterns'
                current_content = []
            elif '=== 8. GENERAL RULE' in line_stripped or 'GENERAL RULE' in line_stripped:
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'general_rule'
                current_content = []
            elif '=== 9. GENERAL STEPS' in line_stripped or 'GENERAL STEPS' in line_stripped:
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'general_steps'
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content)
        
        # Remove None values
        structured = {k: v for k, v in sections.items() if v}
        
        return structured
    
    def _parse_rule_to_steps(self, rule_text: str) -> List[Dict]:
        """Parse general rule text and convert to step format"""
        if not rule_text:
            return []
        
        steps = []
        step_num = 1
        
        # Common transformation names to look for
        transformation_keywords = {
            'color_mapping': ['color', 'recolor', 'change color', 'map color', 'paint', 'fill'],
            'position_change': ['move', 'position', 'relocate', 'shift', 'translate', 'slide'],
            'size_change': ['resize', 'scale', 'size', 'expand', 'shrink', 'grow', 'extend'],
            'rotate': ['rotate', 'rotation', 'turn'],
            'flip_horizontal': ['flip horizontal', 'horizontal flip', 'mirror horizontal'],
            'flip_vertical': ['flip vertical', 'vertical flip', 'mirror vertical'],
            'create_objects': ['create', 'add', 'new object', 'generate', 'copy', 'duplicate', 'propagate', 'tile', 'repeat'],
            'transform': ['transform', 'modify', 'change', 'alter', 'apply']
        }
        
        # Split rule into sentences - handle numbered lists too
        import re
        # Split by newlines first, then by period
        raw_lines = rule_text.split('\n')
        sentences = []
        for line in raw_lines:
            # Remove numbering like "1.", "1)", "-", "*"
            clean_line = re.sub(r'^\s*(?:\d+[\.\)]|\-|\*)\s*', '', line).strip()
            if clean_line:
                # Split by period but keep it reasonable
                parts = clean_line.split('.')
                for p in parts:
                    if p.strip():
                        sentences.append(p.strip())
        
        current_step = None
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
            
            # Look for transformation keywords
            sentence_lower = sentence.lower()
            transition = None
            for trans_name, keywords in transformation_keywords.items():
                if any(kw in sentence_lower for kw in keywords):
                    transition = trans_name
                    break
            
            if not transition:
                # If no keyword, skip this sentence unless it looks like a step
                if not (sentence_lower.startswith('step') or sentence_lower.startswith('for each')):
                     continue
                transition = 'transform'  # Default
            
            # Extract condition (look for "for each", "if", "when", etc.)
            condition = 'all_objects'  # Default
            if 'for each' in sentence_lower:
                # Try to extract the condition after "for each"
                parts = sentence_lower.split('for each', 1)
                if len(parts) > 1:
                    condition_part = parts[1].split(',')[0].split('perform')[0].strip()
                    if condition_part:
                        condition = condition_part[:100]  # Limit length
            elif 'if' in sentence_lower:
                 parts = sentence_lower.split('if', 1)
                 if len(parts) > 1:
                    condition_part = parts[1].split(',')[0].split('then')[0].strip()
                    if condition_part:
                        condition = condition_part[:100]
            
            # Create step
            steps.append({
                'step_number': step_num,
                'instruction': f"Step {step_num}: {sentence[:200]}",
                'condition': condition,
                'transition': transition
            })
            step_num += 1
        
        return steps if steps else [{
            'step_number': 1,
            'instruction': f"Step 1: {rule_text[:200]}",
            'condition': 'all_objects',
            'transition': 'transform'
        }]
    
    def generate_general_steps(self, analysis: Dict[str, Any]) -> List[Dict]:
        """Generate general steps from analysis (with images)"""
        print("[STEPS] Generating general steps from analysis...")
        
        train_examples = analysis.get('train_examples', [])
        test_input = analysis.get('test_input')
        
        # Try to use the general rule first (it's already phrased as steps)
        rule = analysis.get('rule', '')
        if rule and len(rule) > 50:
            print("[STEPS] Attempting to parse general rule as steps...")
            rule_steps = self._parse_rule_to_steps(rule)
            if rule_steps:
                print(f"[STEPS] Parsed {len(rule_steps)} steps from general rule")
                # Validate and refine the steps with full analysis
                # But use rule as base
                pass  # Continue to full step generation but use rule as guidance
        
        # Analyze which objects actually change vs stay the same
        object_change_analysis = []
        has_new_objects = False
        new_objects_info = []
        
        # Track objects that remain unchanged across ALL examples
        all_unchanged_objects = []  # Objects that stay same in ALL examples
        example_unchanged_sets = []  # Unchanged objects per example
        
        for i, ex in enumerate(train_examples):
            if ex.get('input') and ex.get('output'):
                input_objs = self._detect_objects(ex['input'])
                output_objs = self._detect_objects(ex['output'])
                
                # Match objects between input and output
                object_matches = self._match_objects(input_objs, output_objs, ex['input'], ex['output'])
                
                # Analyze which objects change and how
                changed_objects = []
                unchanged_objects = []
                
                for input_idx, input_obj in enumerate(input_objs):
                    output_idx = object_matches.get(input_idx)
                    if output_idx is not None and output_idx < len(output_objs):
                        output_obj = output_objs[output_idx]
                        
                        # Check if object actually changed
                        input_colors = set(input_obj.get('colors', []))
                        output_colors = set(output_obj.get('colors', []))
                        input_bbox = input_obj.get('bbox', [])
                        output_bbox = output_obj.get('bbox', [])
                        
                        color_changed = input_colors != output_colors
                        position_changed = input_bbox != output_bbox
                        size_changed = input_obj.get('size', 0) != output_obj.get('size', 0)
                        
                        if color_changed or position_changed or size_changed:
                            changed_objects.append({
                                'input_colors': sorted(input_colors),
                                'output_colors': sorted(output_colors),
                                'input_desc': input_obj.get('description', 'object'),
                                'output_desc': output_obj.get('description', 'object'),
                                'color_changed': color_changed,
                                'position_changed': position_changed,
                                'size_changed': size_changed
                            })
                        else:
                            unchanged_obj_info = {
                                'colors': sorted(input_colors),
                                'description': input_obj.get('description', 'object'),
                                'size': input_obj.get('size', 0),
                                'bbox': input_bbox
                            }
                            unchanged_objects.append(unchanged_obj_info)
                            # Track for cross-example analysis
                            example_unchanged_sets.append({
                                'example': i + 1,
                                'object': unchanged_obj_info
                            })
                
                # Find output objects that don't match any input object (new objects)
                matched_output_indices = set(v for v in object_matches.values() if v is not None)
                unmatched_output_objs = [obj for obj in output_objs if output_objs.index(obj) not in matched_output_indices]
                if unmatched_output_objs:
                    has_new_objects = True
                    new_objects_info.append({
                        'example': i + 1,
                        'count': len(unmatched_output_objs),
                        'descriptions': [obj.get('description', 'object') for obj in unmatched_output_objs[:3]]
                    })
                
                object_change_analysis.append({
                    'example': i + 1,
                    'changed_count': len(changed_objects),
                    'unchanged_count': len(unchanged_objects),
                    'changed_objects': changed_objects[:5],  # Limit to first 5
                    'unchanged_objects': unchanged_objects[:5]  # Limit to first 5
                })
        
        if has_new_objects:
            print(f"[STEPS] Detected new objects in {len(new_objects_info)} training examples - will include create step")
        
        print(f"[STEPS] Object change analysis: {sum(a['changed_count'] for a in object_change_analysis)} changed, {sum(a['unchanged_count'] for a in object_change_analysis)} unchanged across examples")
        
        # Find objects that remain unchanged in ALL examples
        if len(example_unchanged_sets) > 0:
            # Group by object characteristics
            from collections import defaultdict
            unchanged_by_chars = defaultdict(list)
            for item in example_unchanged_sets:
                obj = item['object']
                key = (tuple(sorted(obj['colors'])), obj['size'], obj['description'])
                unchanged_by_chars[key].append(item['example'])
            
            # Objects that appear unchanged in all examples
            for (colors, size, desc), examples in unchanged_by_chars.items():
                if len(examples) == len(train_examples):
                    all_unchanged_objects.append({
                        'colors': list(colors),
                        'size': size,
                        'description': desc,
                        'examples': examples
                    })
        
        if all_unchanged_objects:
            print(f"[STEPS] Found {len(all_unchanged_objects)} object type(s) that remain UNCHANGED in ALL examples - will add protection step")
            # Store for use in booklet generation
            self._last_unchanged_objects = all_unchanged_objects
        
        # Build content with images
        content = []
        
        # Include ALL analysis and comparison sections
        rule = analysis.get('rule', '')
        analysis_part1 = analysis.get('analysis_part1', '')
        analysis_part2 = analysis.get('analysis_part2', '')
        analysis_part3 = analysis.get('analysis_part3', '')
        
        # Include structured comparison sections
        comparison_sections = analysis.get('comparison_sections', {})
        comparison_text = ""
        if comparison_sections.get('input_input') or comparison_sections.get('output_output') or comparison_sections.get('input_output'):
            comparison_text = "\n\n=== STRUCTURED COMPARISONS ===\n"
            if comparison_sections.get('input_input'):
                comparison_text += f"\nINPUT-INPUT COMPARISONS:\n{comparison_sections['input_input']}\n"
            if comparison_sections.get('output_output'):
                comparison_text += f"\nOUTPUT-OUTPUT COMPARISONS:\n{comparison_sections['output_output']}\n"
            if comparison_sections.get('input_output'):
                comparison_text += f"\nINPUT-OUTPUT COMPARISONS:\n{comparison_sections['input_output']}\n"
        
        # Sanitize analysis text
        safe_part1 = self._sanitize_text(analysis_part1[:400] if analysis_part1 else '')
        safe_part2 = self._sanitize_text(analysis_part2[:400] if analysis_part2 else '')
        safe_part3 = self._sanitize_text(analysis_part3[:400] if analysis_part3 else '')
        safe_rule = self._sanitize_text(rule[:500] if rule else '')
        
        # Try to use rule as steps first (primary method)
        rule_steps = []
        if rule and len(rule) > 50:
            rule_steps = self._parse_rule_to_steps(rule)
            if rule_steps:
                print(f"[STEPS] Parsed {len(rule_steps)} steps from general rule")
                # Use rule steps directly, but refine transition names to match tool format
                for step in rule_steps:
                    # Normalize transition names to match tool format
                    trans = step.get('transition', 'transform').lower()
                    if 'color' in trans or 'recolor' in trans:
                        step['transition'] = 'color_mapping'
                    elif 'move' in trans or 'position' in trans:
                        step['transition'] = 'position_change'
                    elif 'size' in trans or 'scale' in trans:
                        step['transition'] = 'size_change'
                    elif 'rotate' in trans:
                        step['transition'] = 'rotate'
                    elif 'flip' in trans and 'horizontal' in trans:
                        step['transition'] = 'flip_horizontal'
                    elif 'flip' in trans and 'vertical' in trans:
                        step['transition'] = 'flip_vertical'
                    elif 'create' in trans or 'add' in trans:
                        step['transition'] = 'create_objects'
                    else:
                        step['transition'] = 'transform'
        
        # Build the text content first
        step_text = self._sanitize_text(f"""Generate GENERAL STEPS that work for ALL training examples.

GENERAL RULE (use this as the PRIMARY source - parse it into steps with proper format):
{safe_rule}

CRITICAL: The rule above is already phrased as transformation steps. Parse it into the JSON format below, ensuring:
- Transition names match exactly: color_mapping, position_change, size_change, rotate, flip_horizontal, flip_vertical, create_objects, or transform
- Conditions are specific (e.g., "composed only of colors X and Y")
- Each sentence in the rule becomes a step

Analysis Summary:
- Stage 1: {safe_part1}...
- Stage 2: {safe_part2}...
- Stage 3: {safe_part3}...

CRITICAL OBJECT DETECTION RULES:
- An object with a same-colored object inside it is USUALLY ONE OBJECT (e.g., a square with a dot inside, both same color = one object)
- Only split into separate objects if they are DIFFERENT colors or clearly disconnected
- Pay attention to what objects DO NOT CHANGE - these should NOT be in transformation steps
- Identify which objects stay the same across all examples - these are reference objects

CRITICAL: UNCHANGED OBJECTS (MUST NOT BE TRANSFORMED):
{chr(10).join(f"- Objects with colors {obj['colors']}, size {obj['size']}, description '{obj['description']}' - these remain UNCHANGED in ALL {len(train_examples)} training examples" for obj in all_unchanged_objects) if all_unchanged_objects else "- No objects remain completely unchanged across all examples"}

CRITICAL STEP GENERATION:
1. FIRST STEP: Add a step that says "Under NO circumstances change objects matching [condition]" for objects that remain unchanged
2. Steps MUST work for ALL training examples (if only one, it's WRONG)
3. Conditions must be EXACT and SPECIFIC - identify EXACTLY which objects transform with precise conditions
   - Use exact colors: "composed only of colors X and Y" (not just "color X")
   - Use exact size if relevant: "size exactly N cells"
   - Use exact shape/structure: "connected object", "horizontal line", etc.
   - Be PRECISE: "connected two-cell object composed only of colors 1 and 6" is better than "object with colors 1 and 6"
4. Transition names must match toolcall transformation_type exactly
5. DO NOT create unnecessary substeps - only create substeps/sub-substeps when needed for crop-transform-uncrop operations
6. If an object doesn't need cropping (already isolated), skip crop/uncrop steps
7. Focus on what CHANGES - objects that stay the same should NOT be in transformation steps
8. For each transformation step, state the EXACT condition that identifies which objects transform - be extremely specific

Format:
{{
  "steps": [
    {{
      "step_number": 1,
      "instruction": "Step 1: for each (condition) A, perform transition B",
      "condition": "specific_condition",
      "transition": "transition_name"
    }}
  ]
}}

Examine training examples below. Focus on what changes consistently across ALL examples.""")
        
        # Add detailed object change analysis
        change_analysis_text = f"""

=== CRITICAL: OBJECT CHANGE ANALYSIS ===

PAY ATTENTION TO WHICH OBJECTS ACTUALLY CHANGE vs WHICH STAY THE SAME.

This analysis shows exactly which objects transform and how they change:

"""
        for analysis in object_change_analysis:
            change_analysis_text += f"\nTRAINING EXAMPLE {analysis['example']}:\n"
            change_analysis_text += f"- {analysis['changed_count']} objects CHANGED\n"
            change_analysis_text += f"- {analysis['unchanged_count']} objects UNCHANGED\n"
            
            if analysis['changed_objects']:
                change_analysis_text += "\nCHANGED OBJECTS (these are the ones that need transformation steps):\n"
                for obj in analysis['changed_objects']:
                    changes = []
                    if obj['color_changed']:
                        changes.append(f"color {sorted(obj['input_colors'])} -> {sorted(obj['output_colors'])}")
                    if obj['position_changed']:
                        changes.append("position")
                    if obj['size_changed']:
                        changes.append("size")
                    change_analysis_text += f"  - {obj['input_desc']}: {', '.join(changes)}\n"
            
            if analysis['unchanged_objects']:
                change_analysis_text += "\nUNCHANGED OBJECTS (these should NOT be in transformation steps):\n"
                for obj in analysis['unchanged_objects']:
                    change_analysis_text += f"  - {obj['description']} (colors: {obj['colors']})\n"
        
        change_analysis_text += """

CRITICAL INSTRUCTIONS FOR STEP GENERATION:
1. Focus ONLY on objects that ACTUALLY CHANGE - ignore objects that stay the same
2. Create conditionals that SPECIFICALLY identify the CHANGING objects
3. Use the change patterns (color changes, position changes, etc.) to create precise conditionals
4. Example: If objects with colors [1, 2] change to colors [3, 4], the condition should be "objects composed only of colors 1 and 2"
5. Do NOT create steps for objects that remain unchanged
6. The condition must be specific enough to ONLY match the objects that change

"""
        step_text += change_analysis_text
        
        # Add warning about new objects if detected
        if has_new_objects:
            new_objects_warning = f"""

CRITICAL: NEW OBJECTS DETECTED
Objects appear in the output that do NOT exist in the input. You MUST include a step to CREATE these new objects.

New objects found in training examples:
{chr(10).join(f"- Example {info['example']}: {info['count']} new object(s) - {', '.join(info['descriptions'])}" for info in new_objects_info)}

Add a final step like: 'Step N: Create new objects that appear in output but not in input' with transition='create_objects'"""
            step_text += new_objects_warning
        
        step_text += """

CRITICAL VALIDATION REQUIREMENT:
- The steps you generate MUST work for ALL training examples
- If a step only works for one example, it's WRONG
- Test your steps mentally: "Does this step work for example 1? Example 2? Example 3?"
- If the answer is NO for any example, revise the step

STEP GENERATION PROCESS:
1. Identify the transformation that works for ALL training examples
2. Create specific conditions that identify which objects/sections transform
3. Verify: Does this step work for example 1? ✓ Example 2? ✓ Example 3? ✓
4. If any example fails, the step is wrong - revise it

Examine the training examples below to determine the transformation steps.
REMEMBER: 
- Focus on objects that CHANGE, not objects that stay the same
- The steps MUST work for ALL training examples - if one fails, the steps are wrong
- Be specific about conditions - vague conditions lead to incorrect transformations"""
        
        content.append({
            "type": "text",
            "text": step_text
        })
        
        # Add all training example images
        for i, ex in enumerate(train_examples):
            input_img = grid_to_image(ex['input'], cell_size=40)
            output_img = grid_to_image(ex['output'], cell_size=40)
            
            content.append({
                "type": "text",
                "text": f"\n=== TRAINING EXAMPLE {i+1} ==="
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(input_img)}"}
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(output_img)}"}
            })
        
        # Add test input if provided
        if test_input:
            test_img = grid_to_image(test_input, cell_size=40)
            content.append({
                "type": "text",
                "text": f"\n=== TEST INPUT ==="
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(test_img)}"}
            })
        
        content.append({
            "type": "text",
            "text": """
CRITICAL: You MUST respond with valid JSON in this exact format:
{
  "steps": [
    {
      "step_number": 1,
      "instruction": "Step 1: [description of what to do]",
      "condition": "[condition for which objects this applies to]",
      "transition": "[transformation type: color_mapping, position_change, size_change, create_objects, etc.]"
    },
    {
      "step_number": 2,
      "instruction": "Step 2: [description]",
      "condition": "[condition]",
      "transition": "[transformation type]"
    }
  ]
}

IMPORTANT:
- You MUST return valid JSON, not plain text
- The JSON must have a "steps" array
- Each step must have step_number, instruction, condition, and transition
- If new objects need to be created, include a step with transition="create_objects"
- Do NOT use tool calls - return the JSON directly in your response

Generate the general steps in JSON format now:"""
        })
        
        try:
            # For step generation, also use tool_choice="none" to force text/JSON response
            call_params = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_object"},
                "tools": self.tools,  # Always pass tools
                "tool_choice": "none"  # Force JSON response, no tool calls
            }
            
            if "gpt-5" in self.model:
                call_params["max_completion_tokens"] = 2000
            else:
                call_params["max_tokens"] = 2000
            
            text_items = [item for item in content if item.get('type') == 'text']
            image_items = [item for item in content if item.get('type') == 'image_url']
            total_text_len = sum(len(item.get('text', '')) for item in text_items)
            print(f"[DEBUG] Generating steps with {len(text_items)} text items, {len(image_items)} images, {total_text_len} chars")
            response = self.client.chat.completions.create(**call_params)
            
            # Check for tool calls first (even though tool_choice is "none", sometimes models still try)
            if response.choices[0].message.tool_calls:
                print("[WARNING] LLM made tool calls instead of returning JSON - this should not happen with tool_choice='none'")
                # Force a retry with a simpler prompt
                print("[RETRY] Retrying with explicit JSON-only instruction...")
                simple_prompt = [{
                    "type": "text",
                    "text": f"""Based on this analysis, generate steps in JSON format only:

{analysis.get('rule', 'No rule available')}

Return ONLY valid JSON in this format:
{{
  "steps": [
    {{"step_number": 1, "instruction": "...", "condition": "...", "transition": "..."}}
  ]
}}"""
                }]
                retry_params = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": simple_prompt}],
                    "response_format": {"type": "json_object"},
                    "tool_choice": "none"
                }
                if "gpt-5" in self.model:
                    retry_params["max_completion_tokens"] = 2000
                else:
                    retry_params["max_tokens"] = 2000
                response = self.client.chat.completions.create(**retry_params)
            
            content = response.choices[0].message.content
            print(f"[DEBUG] LLM response length: {len(content) if content else 0} characters")
            if content:
                # Safely print first 200 chars (handle Unicode)
                try:
                    preview = content[:200]
                    preview = preview.replace('→', '->').replace('←', '<-').replace('↑', '^').replace('↓', 'v')
                    preview = preview.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '--')
                    preview = preview.replace('\u2018', "'").replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
                    print(f"[DEBUG] First 200 chars: {preview}")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    print(f"[DEBUG] First 200 chars: [Unable to display due to encoding]")
            else:
                print("[WARNING] Empty response content from LLM - using fallback step extraction")
                # Fallback to using parsed rule steps FIRST
                rule = analysis.get('rule', '')
                if rule:
                    print("[FALLBACK] Trying to parse steps from rule text directly...")
                    steps = self._parse_rule_to_steps(rule)
                    if steps:
                        print(f"[FALLBACK] Successfully parsed {len(steps)} steps from rule")
                        return steps

                # Try to extract steps from raw text as fallback
                raw_analysis_text = analysis.get('raw_analysis', '') or analysis.get('analysis_part1', '') + '\n' + analysis.get('analysis_part2', '') + '\n' + analysis.get('analysis_part3', '')
                if raw_analysis_text:
                    print("[FALLBACK] Extracting steps from analysis text...")
                    steps = self._extract_steps_from_text(raw_analysis_text)
                    if steps:
                        print(f"[FALLBACK] Extracted {len(steps)} steps from analysis text")
                        return steps
                print("[ERROR] No analysis text available for step extraction")
                # Create a default step as last resort
                return [{
                    'step_number': 1,
                    'instruction': 'Step 1: Apply transformation based on analysis',
                    'condition': 'all_objects',
                    'transition': 'unknown'
                }]
            
            if not content or not content.strip():
                print("[ERROR] Empty or whitespace-only response from LLM")
                raw_analysis_text = analysis.get('raw_analysis', '') or analysis.get('analysis_part1', '') + '\n' + analysis.get('analysis_part2', '') + '\n' + analysis.get('analysis_part3', '')
                if raw_analysis_text:
                    steps = self._extract_steps_from_text(raw_analysis_text)
                    return steps
                return []
            
            result = json.loads(content)
            steps = result.get('steps', [])
            
            # If no steps or empty response, use parsed rule steps as primary fallback
            if not steps:
                print("[WARNING] No steps found in response, using parsed rule steps...")
                # Use parsed rule steps (already normalized)
                rule = analysis.get('rule', '')
                if rule:
                    steps = self._parse_rule_to_steps(rule)
                    if steps:
                        print(f"[STEPS] Using {len(steps)} steps parsed from general rule")
                        # Normalize transition names
                        for step in steps:
                            trans = step.get('transition', 'transform').lower()
                            if 'color' in trans or 'recolor' in trans:
                                step['transition'] = 'color_mapping'
                            elif 'move' in trans or 'position' in trans:
                                step['transition'] = 'position_change'
                            elif 'size' in trans or 'scale' in trans:
                                step['transition'] = 'size_change'
                            elif 'rotate' in trans:
                                step['transition'] = 'rotate'
                            elif 'flip' in trans and 'horizontal' in trans:
                                step['transition'] = 'flip_horizontal'
                            elif 'flip' in trans and 'vertical' in trans:
                                step['transition'] = 'flip_vertical'
                            elif 'create' in trans or 'add' in trans:
                                step['transition'] = 'create_objects'
                            else:
                                step['transition'] = 'transform'
                
                # Fallback to extracting from analysis text
                if not steps:
                    raw_analysis_text = analysis.get('raw_analysis', '') or analysis.get('analysis_part1', '') + '\n' + analysis.get('analysis_part2', '') + '\n' + analysis.get('analysis_part3', '')
                    if raw_analysis_text:
                        steps = self._extract_steps_from_text(raw_analysis_text)
                    else:
                        print("[WARNING] No analysis text available for step extraction")
            
            # Add protection step for unchanged objects FIRST (step 0 or before step 1)
            if all_unchanged_objects:
                print("[STEPS] Adding protection step for unchanged objects...")
                # Build condition string for unchanged objects
                unchanged_conditions = []
                for obj in all_unchanged_objects:
                    color_str = f"colors {obj['colors']}" if len(obj['colors']) == 1 else f"composed only of colors {obj['colors']}"
                    size_str = f"size exactly {obj['size']}" if obj['size'] > 0 else ""
                    desc_str = obj['description']
                    condition_parts = [color_str]
                    if size_str:
                        condition_parts.append(size_str)
                    condition_parts.append(desc_str)
                    unchanged_conditions.append(" and ".join(condition_parts))
                
                protection_condition = " or ".join(f"({cond})" for cond in unchanged_conditions)
                
                # Insert as first step (step 0 or renumber)
                protection_step = {
                    'step_number': 0,
                    'instruction': f"Step 0: Under NO circumstances change objects matching: {protection_condition}",
                    'condition': protection_condition,
                    'transition': 'no_change'
                }
                # Renumber existing steps
                for step in steps:
                    step['step_number'] = step.get('step_number', 1) + 1
                steps.insert(0, protection_step)
                print(f"[STEPS] Added protection step: {protection_step['instruction']}")
            
            # If new objects detected but no create step found, add it
            if has_new_objects:
                has_create_step = any('create' in step.get('transition', '').lower() or 
                                    'create' in step.get('instruction', '').lower() or
                                    'new object' in step.get('instruction', '').lower()
                                    for step in steps)
                if not has_create_step:
                    print("[STEPS] Adding create_objects step for new objects...")
                    create_step_num = max([s.get('step_number', 0) for s in steps] + [0]) + 1
                    steps.append({
                        'step_number': create_step_num,
                        'instruction': f"Step {create_step_num}: Create new objects that appear in output but not in input",
                        'condition': 'new_objects_in_output',
                        'transition': 'create_objects'
                    })
            
            print(f"[OK] Generated {len(steps)} general steps")
            for i, step in enumerate(steps):
                print(f"  Step {step.get('step_number', i+1)}: {step.get('instruction', 'N/A')[:80]}...")
            
            return steps
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse JSON: {e}")
            try:
                if response and response.choices and response.choices[0].message.content:
                    preview = response.choices[0].message.content[:500]
                    preview = preview.replace('→', '->').replace('←', '<-').replace('↑', '^').replace('↓', 'v')
                    preview = preview.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '--')
                    preview = preview.replace('\u2018', "'").replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
                    print(f"[DEBUG] Response content: {preview}")
            except (UnicodeEncodeError, UnicodeDecodeError):
                print(f"[DEBUG] Response content: [Unable to display due to encoding]")
            except:
                pass
            # Try to extract steps from raw text
            raw_analysis_text = analysis.get('raw_analysis', '') or analysis.get('analysis_part1', '') + '\n' + analysis.get('analysis_part2', '') + '\n' + analysis.get('analysis_part3', '')
            if raw_analysis_text:
                steps = self._extract_steps_from_text(raw_analysis_text)
                return steps
            else:
                print("[ERROR] No analysis text available for step extraction")
                return []
        except Exception as e:
            print(f"[ERROR] Error generating steps: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _extract_steps_from_text(self, text: str) -> List[Dict]:
        """Fallback: Try to extract steps from analysis text"""
        steps = []
        lines = text.split('\n')
        step_num = 1
        
        # If text is short/one line, treat as single step
        if len(lines) <= 2 and len(text) < 300:
             return [{
                'step_number': 1,
                'instruction': f"Step 1: {text}",
                'condition': 'all_objects',
                'transition': 'transform'
             }]
             
        # Try to parse rule-like lists (1., 2., - )
        import re
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Check for numbered list or "Step X"
            is_list_item = bool(re.match(r'^\d+[\.\)]|\-|\*|Step\s+\d+', line))
            
            if is_list_item or 'perform' in line.lower() or 'transform' in line.lower():
                # Clean up numbering
                instruction = re.sub(r'^\d+[\.\)]\s*|\-\s*|\*\s*', '', line).strip()
                if not instruction: continue
                
                # Try to extract condition and transition
                condition = "all_objects"
                transition = "transform"
                
                # Condition
                if 'for each' in instruction.lower():
                     parts = instruction.lower().split('for each', 1)
                     if len(parts) > 1:
                         condition = parts[1].split(',')[0].split('perform')[0].strip()
                elif 'if' in instruction.lower():
                     parts = instruction.lower().split('if', 1)
                     if len(parts) > 1:
                         condition = parts[1].split(',')[0].split('then')[0].strip()
                
                # Transition detection
                if 'color' in instruction.lower() or 'paint' in instruction.lower(): transition = 'color_mapping'
                elif 'move' in instruction.lower(): transition = 'position_change'
                elif 'size' in instruction.lower(): transition = 'size_change'
                elif 'create' in instruction.lower() or 'add' in instruction.lower(): transition = 'create_objects'
                elif 'rotate' in instruction.lower(): transition = 'rotate'
                elif 'flip' in instruction.lower(): transition = 'flip_horizontal' # simplified
                
                steps.append({
                    'step_number': step_num,
                    'instruction': f"Step {step_num}: {instruction}",
                    'condition': condition,
                    'transition': transition
                })
                step_num += 1
        
        return steps if steps else [{
            'step_number': 1,
            'instruction': 'Step 1: Analyze and transform based on puzzle pattern',
            'condition': 'all_objects',
            'transition': 'transform'
        }]
    
    def generate_booklet_for_example(self, example: Dict, general_steps: List[Dict], 
                                    is_test: bool = False, training_booklets: List[Dict] = None,
                                    analysis: Dict = None, all_unchanged_objects: List[Dict] = None) -> Dict[str, Any]:
        """Generate step-by-step booklet for a single example - object by object like v4/v6"""
        print(f"  [{'TEST' if is_test else 'TRAIN'}] Generating booklet for {'test' if is_test else 'training'} example...")
        
        if all_unchanged_objects is None:
            all_unchanged_objects = []
        
        # Deep copy input and output to avoid reference issues
        input_copy = [row[:] for row in example['input']]
        output_copy = [row[:] for row in example['output']] if example.get('output') else None
        
        # Debug: Verify output is different from input (for training examples)
        if not is_test and output_copy:
            if input_copy == output_copy:
                print(f"    [WARNING] Input and output are identical - this may indicate a data issue")
        
        # Determine initial grid size: if output is larger, start with output size
        input_rows = len(input_copy)
        input_cols = len(input_copy[0]) if input_copy else 0
        output_rows = len(output_copy) if output_copy else input_rows
        output_cols = len(output_copy[0]) if output_copy and output_copy[0] else input_cols
        
        # Initialize current_grid: if output is larger, use output size; otherwise use input size
        if output_rows > input_rows or output_cols > input_cols:
            # Start with output size (filled with zeros/background)
            print(f"    [INFO] Input size ({input_rows}x{input_cols}) < Output size ({output_rows}x{output_cols}) - initializing with output size")
            current_grid = [[0] * output_cols for _ in range(output_rows)]
            # Place input grid at top-left of the larger grid
            for i in range(min(input_rows, output_rows)):
                for j in range(min(input_cols, output_cols)):
                    current_grid[i][j] = input_copy[i][j]
            
            # For input < output puzzles, detect objects from OUTPUT to create
            print(f"    [INFO] Detecting objects in OUTPUT to create (input < output puzzle)")
            if output_copy:
                output_objects_for_creation = self._detect_objects(output_copy)
                print(f"    [INFO] Found {len(output_objects_for_creation)} objects in output that may need to be created")
        else:
            # Use input size (normal case)
            current_grid = [row[:] for row in input_copy]
        
        booklet = {
            'steps': [],
            'input': input_copy,
            'output': output_copy,
            'current_grid': current_grid  # Initialize with appropriate size
        }
        
        # Step 1: Detect objects in input
        print(f"    Detecting objects in input...")
        input_objects = self._detect_objects(booklet['current_grid'])
        print(f"    Found {len(input_objects)} objects")
        
        # Step 2: Match objects between input and output (if training)
        object_matches = {}
        output_objects = []
        new_objects = []  # Objects in output but not in input
        if not is_test and example.get('output'):
            print(f"    Matching objects to output...")
            output_objects = self._detect_objects(example['output'])
            object_matches = self._match_objects(input_objects, output_objects, example['input'], example['output'])
            print(f"    Matched {len([v for v in object_matches.values() if v is not None])} objects")
            
            # Find output objects that don't have a match (new objects to create)
            matched_output_indices = set(v for v in object_matches.values() if v is not None)
            for i, output_obj in enumerate(output_objects):
                if i not in matched_output_indices:
                    new_objects.append(output_obj)
            print(f"    Found {len(new_objects)} new objects to create in output")
        
        # Step 3: For each general step, process each matching object
        for general_step in general_steps:
            print(f"\n    Processing general step {general_step['step_number']}: {general_step['instruction']}")
            
            # Find objects that match this step's condition
            objects_to_process = self._filter_objects_by_condition(
                input_objects, 
                general_step['condition'],
                full_grid=booklet['current_grid']
            )
            print(f"      Found {len(objects_to_process)}/{len(input_objects)} objects matching condition: {general_step['condition']}")
            
            # Reset substep counter for this general step (each object is a substep)
            substep_counter = 1
            
            # Skip step 0 (protection step) - it doesn't process objects
            if general_step.get('step_number') == 0 or general_step.get('transition') == 'no_change':
                print(f"      [SKIP] Step 0 is protection step - no objects to process")
                continue
            
            # Process each object with crop-transform-uncrop
            for obj_idx, obj in enumerate(objects_to_process):
                obj_num = input_objects.index(obj) + 1
                print(f"      Processing object {obj_num}/{len(input_objects)}: {obj.get('description', 'object')}")
                
                # Check if this object matches the protection condition (should not be transformed)
                obj_colors = set(obj.get('colors', []))
                obj_size = obj.get('size', 0)
                obj_desc = obj.get('description', '')
                
                # Check against all unchanged objects
                is_protected = False
                if all_unchanged_objects:
                    for unchanged_obj in all_unchanged_objects:
                        unchanged_colors = set(unchanged_obj.get('colors', []))
                        unchanged_size = unchanged_obj.get('size', 0)
                        unchanged_desc = unchanged_obj.get('description', '')
                        
                        if (obj_colors == unchanged_colors and
                            obj_size == unchanged_size and
                            obj_desc == unchanged_desc):
                            is_protected = True
                            print(f"      [PROTECTED] Object {obj_num} matches unchanged condition - skipping transformation")
                            break
                
                if is_protected:
                    continue  # Skip this object - it should not be transformed
                
                # Get corresponding output object if available
                output_obj = None
                if obj_num - 1 in object_matches and object_matches[obj_num - 1] is not None:
                    output_obj_idx = object_matches[obj_num - 1]
                    if output_obj_idx < len(output_objects):
                        output_obj = output_objects[output_obj_idx]
                
                # Object substep number (e.g., 1.1, 1.2)
                object_substep_num = f"{general_step['step_number']}.{substep_counter}"
                substep_counter += 1
                
                # SUB-SUBSTEP 1: Crop to object (1.1.1, 1.1.2, 1.1.3)
                subsubstep_num = f"{object_substep_num}.1"
                
                # Store full grid state (shows bbox region on full grid)
                full_grid_with_bbox = [row[:] for row in booklet['current_grid']]
                cropped_input, bbox = self._crop_to_object(booklet['current_grid'], obj['bbox'])
                # bbox will be updated to union_bbox if object moves
                
                booklet['steps'].append({
                    'step_number': subsubstep_num,
                    'object_substep': object_substep_num,
                    'general_step': general_step['step_number'],
                    'instruction': f"CROP: Object {obj_num} ({obj.get('description', 'object')})",
                    'substep_reasoning': f"Cropping to object {obj_num} to apply {general_step['transition']} transformation. Bbox: {obj['bbox']}",
                    'grid_before': full_grid_with_bbox,  # Full grid showing bbox region
                    'grid_after': full_grid_with_bbox,  # Crop doesn't change full grid, but shows bbox
                    'tool_used': 'crop',
                    'tool_params': {'bbox': obj['bbox']},
                    'bbox': obj['bbox'],  # Bbox to highlight on full grid
                    'object_num': obj_num,
                    'is_crop_step': True,
                    'cropped_grid': cropped_input,  # The actual cropped region
                    'is_subsubstep': True,
                    'shows_bbox_on_full_grid': True  # Flag to indicate bbox should be shown
                })
                
                # SUB-SUBSTEP 2: Transform the cropped object
                subsubstep_num = f"{object_substep_num}.2"
                
                # Get cropped output for reference
                cropped_output = None
                if output_obj and not is_test:
                    # For moving objects, bbox must be union of before and after positions
                    input_bbox = obj['bbox']
                    output_bbox = output_obj.get('bbox', input_bbox)
                    # Calculate union bbox: [min(min_r), min(min_c), max(max_r), max(max_c)]
                    union_bbox = [
                        min(input_bbox[0], output_bbox[0]),  # min_r
                        min(input_bbox[1], output_bbox[1]),  # min_c
                        max(input_bbox[2], output_bbox[2]),  # max_r
                        max(input_bbox[3], output_bbox[3])   # max_c
                    ]
                    # Use union bbox for cropping output (ensures we capture the full movement)
                    cropped_output, _ = self._crop_to_object(example['output'], union_bbox)
                    # Update bbox to union for uncrop step (ensures full movement is captured)
                    bbox = union_bbox
                else:
                    # No output object - keep original bbox
                    bbox = obj['bbox']
                
                # Call LLM to transform the cropped object
                transformed_crop = self._transform_object(
                    cropped_input, cropped_output, obj, output_obj,
                    general_step, is_test, training_booklets, analysis, example.get('input', [])
                )
                
                # Store full grid state before transform (for visualization)
                full_grid_before = [row[:] for row in booklet['current_grid']]
                
                booklet['steps'].append({
                    'step_number': subsubstep_num,
                    'object_substep': object_substep_num,
                    'general_step': general_step['step_number'],
                    'instruction': f"TRANSFORM: Object {obj_num} ({general_step['transition']})",
                    'substep_reasoning': f"Applying {general_step['transition']} to object {obj_num} based on condition: {general_step['condition']}",
                    'grid_before': cropped_input,  # Cropped input region
                    'grid_after': transformed_crop,  # Transformed cropped view (predicted)
                    'full_grid_before': full_grid_before,  # Full grid state before
                    'tool_used': 'transform',
                    'tool_params': {'transformation_type': general_step['transition']},
                    'bbox': bbox,  # Use union bbox if object moved
                    'object_num': obj_num,
                    'is_cropped_view': True,
                    'parent_crop_step': f"{object_substep_num}.1",
                    'is_subsubstep': True,
                    'cropped_grid_before': cropped_input,  # Explicitly store cropped input
                    'cropped_grid_after': transformed_crop,  # Transformed result (predicted)
                    'cropped_grid_target': cropped_output,  # Ground truth cropped output (target)
                    'input_bbox': obj['bbox'],  # Original input bbox
                    'output_bbox': output_obj.get('bbox', obj['bbox']) if output_obj else None,  # Output bbox
                    'union_bbox': bbox  # Union bbox (for moving objects)
                })
                
                # SUB-SUBSTEP 3: Uncrop back to full grid
                subsubstep_num = f"{object_substep_num}.3"
                
                # Store full grid state before uncrop
                full_grid_before_uncrop = [row[:] for row in booklet['current_grid']]
                
                # Uncrop the transformed object back to full grid
                booklet['current_grid'] = self._uncrop_to_full_grid(
                    transformed_crop, booklet['current_grid'], bbox
                )
                
                # Store full grid state after uncrop
                full_grid_after_uncrop = [row[:] for row in booklet['current_grid']]
                
                booklet['steps'].append({
                    'step_number': subsubstep_num,
                    'object_substep': object_substep_num,
                    'general_step': general_step['step_number'],
                    'instruction': f"UNCROP: Object {obj_num} back to full grid",
                    'substep_reasoning': f"Placing transformed object {obj_num} back into full grid at bbox {bbox}",
                    'grid_before': full_grid_before_uncrop,  # Current whole grid before uncrop
                    'grid_after': full_grid_after_uncrop,  # Current whole grid after uncrop (updated state)
                    'tool_used': 'uncrop',
                    'tool_params': {'bbox': bbox},
                    'bbox': bbox,
                    'object_num': obj_num,
                    'is_uncrop_step': True,
                    'parent_crop_step': f"{object_substep_num}.1",
                    'is_subsubstep': True,
                    'transformed_cropped_grid': transformed_crop,  # The cropped transformed grid that was placed back
                    'shows_current_whole_grid': True  # Flag to indicate this shows current whole grid state
                })
        
        # Step 4: Process create_objects step if it exists in general_steps
        create_steps = [s for s in general_steps if s.get('transition', '').lower() == 'create_objects']
        if create_steps:
            create_step = create_steps[0]  # Use first create step
            create_step_num = create_step['step_number']
            print(f"\n    Processing create_objects step {create_step_num}: {create_step['instruction']}")
            
            # For training: use detected new_objects if available
            if not is_test:
                if new_objects:
                    print(f"    Creating {len(new_objects)} new objects that appear in output but not in input...")
                    
                    for new_obj_idx, new_obj in enumerate(new_objects):
                        obj_num = len(input_objects) + new_obj_idx + 1
                        print(f"      Creating new object {obj_num}: {new_obj.get('description', 'object')}")
                        
                        # Create object substep
                        object_substep_num = f"{create_step_num}.{new_obj_idx + 1}"
                        
                        # Get the object from output grid
                        new_obj_grid = [row[:] for row in example['output']]
                        new_obj_bbox = new_obj.get('bbox', [0, 0, len(new_obj_grid)-1, len(new_obj_grid[0])-1])
                        
                        # Crop to see the new object
                        cropped_new_obj, _ = self._crop_to_object(example['output'], new_obj_bbox)
                        
                        # Create the object using LLM
                        created_grid = self._create_object(
                            booklet['current_grid'], 
                            cropped_new_obj, 
                            new_obj, 
                            new_obj_bbox,
                            example.get('output', []),
                            training_booklets,
                            analysis
                        )
                        
                        # Store full grid before creation
                        full_grid_before_create = [row[:] for row in booklet['current_grid']]
                        
                        # Update current grid with created object
                        booklet['current_grid'] = created_grid
                        
                        # Store full grid after creation
                        full_grid_after_create = [row[:] for row in booklet['current_grid']]
                        
                        # Add create step
                        booklet['steps'].append({
                            'step_number': object_substep_num,
                            'object_substep': object_substep_num,
                            'general_step': create_step_num,
                            'instruction': f"CREATE: New object {obj_num} ({new_obj.get('description', 'object')})",
                            'substep_reasoning': f"Creating new object {obj_num} that appears in output but not in input. This object must be added to complete the transformation.",
                            'grid_before': full_grid_before_create,
                            'grid_after': full_grid_after_create,
                            'tool_used': 'create_objects',
                            'tool_params': {'object': new_obj},
                            'bbox': new_obj_bbox,
                            'object_num': obj_num,
                            'is_create_step': True,
                            'new_object_grid': cropped_new_obj,  # The new object to create
                            'target_output': example.get('output', [])  # Full target output for reference
                        })
                else:
                    # If no new objects detected but create_objects step exists, try to use LLM to create objects anyway
                    # This happens when matching failed but the step says "create"
                    print(f"    No explicit new objects detected, but create_objects step exists. Trying LLM creation...")
                    print(f"    Determining new objects to create for training input...")
                    created_grid = self._create_object_for_test(
                        booklet['current_grid'],
                        create_step,
                        training_booklets,
                        analysis,
                        example.get('input', [])
                    )
                    
                    if created_grid and created_grid != booklet['current_grid']:
                        full_grid_before = [row[:] for row in booklet['current_grid']]
                        booklet['current_grid'] = created_grid
                        full_grid_after = [row[:] for row in booklet['current_grid']]
                        
                        booklet['steps'].append({
                            'step_number': f"{create_step_num}.1",
                            'object_substep': f"{create_step_num}.1",
                            'general_step': create_step_num,
                            'instruction': f"CREATE: New objects based on training patterns",
                            'substep_reasoning': f"Creating new objects that should appear in output based on training examples.",
                            'grid_before': full_grid_before,
                            'grid_after': full_grid_after,
                            'tool_used': 'create_objects',
                            'tool_params': {'step': create_step},
                            'is_create_step': True
                        })

            elif is_test:
                # For test mode: use LLM to determine what objects to create based on training patterns
                print(f"    Determining new objects to create for test input...")
                created_grid = self._create_object_for_test(
                    booklet['current_grid'],
                    create_step,
                    training_booklets,
                    analysis,
                    example.get('input', [])
                )
                
                if created_grid and created_grid != booklet['current_grid']:
                    full_grid_before = [row[:] for row in booklet['current_grid']]
                    booklet['current_grid'] = created_grid
                    full_grid_after = [row[:] for row in booklet['current_grid']]
                    
                    booklet['steps'].append({
                        'step_number': f"{create_step_num}.1",
                        'object_substep': f"{create_step_num}.1",
                        'general_step': create_step_num,
                        'instruction': f"CREATE: New objects based on training patterns",
                        'substep_reasoning': f"Creating new objects that should appear in output based on training examples.",
                        'grid_before': full_grid_before,
                        'grid_after': full_grid_after,
                        'tool_used': 'create_objects',
                        'tool_params': {'step': create_step},
                        'is_create_step': True
                    })
        
        return booklet
    
    def _create_step_prompt(self, general_step: Dict, current_grid: List[List[int]],
                           example: Dict, is_test: bool) -> List[Dict]:
        """Create prompt for performing a specific step"""
        content = [
            {
                "type": "text",
                "text": f"""Perform this step on the current grid:

GENERAL STEP: {general_step['instruction']}
CONDITION: {general_step['condition']}
TRANSITION: {general_step['transition']}

{'TEST MODE: Adapt this step to the test input using training patterns.' if is_test else 'TRAINING MODE: Follow the pattern from training examples.'}

CURRENT GRID STATE:"""
            }
        ]
        
        # Add current grid visual and text
        current_img = grid_to_image(current_grid, cell_size=40)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(current_img)}"}
        })
        content.append({
            "type": "text",
            "text": self._format_grid(current_grid)
        })
        
        if not is_test and example.get('output'):
            # Add target output for training
            target_img = grid_to_image(example['output'], cell_size=40)
            content.append({
                "type": "text",
                "text": "\nTARGET OUTPUT:"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(target_img)}"}
            })
            content.append({
                "type": "text",
                "text": self._format_grid(example['output'])
            })
        
        content.append({
            "type": "text",
            "text": f"""
Perform this step using the available tools.

CRITICAL TOOL USAGE:
1. This step requires the transition: {general_step['transition']}
2. You MUST call the corresponding tool:
   - If transition is 'color_mapping', use 'transform' with transformation_type='color_mapping'
   - If transition is 'position_change', use 'transform' with transformation_type='position_change'
   - If transition is 'size_change', use 'transform' with transformation_type='size_change'
   - If transition is 'rotate', use 'transform' with transformation_type='rotate'
   - If transition is 'flip_horizontal', use 'transform' with transformation_type='flip_horizontal'
   - If transition is 'flip_vertical', use 'transform' with transformation_type='flip_vertical'
   - If transition is 'create_objects', use 'create_objects'
   - If transition is 'transform' or generic, use 'transform' with transformation_type='transform'

3. First, identify objects that match condition: {general_step['condition']}
4. Then apply the transformation to those objects.

Provide 1-2 sentences explaining why this transformation is applied to these objects.
"""
        })
        
        return content
    
    def _perform_step_with_llm(self, prompt: List[Dict], general_step: Dict,
                               current_grid: List[List[int]]) -> Dict[str, Any]:
        """Perform a step using LLM with tool calls"""
        messages = [{"role": "user", "content": prompt}]
        
        max_iterations = 10
        iteration = 0
        substeps = []
        current_state = [row[:] for row in current_grid]
        
        while iteration < max_iterations:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                max_completion_tokens=2000
            )
            
            message = response.choices[0].message
            messages.append(message)
            
            if not message.tool_calls:
                break
            
            # Process tool calls
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Execute tool
                result = self._execute_tool(function_name, function_args, current_state)
                
                # Add to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
                
                # Track substep for grid-modifying operations
                if function_name == "generate_grid":
                    grid_result = result.get('grid', function_args.get('grid', current_state))
                    substeps.append({
                        'grid': grid_result,
                        'tool': function_name,
                        'params': function_args,
                        'reasoning': function_args.get('visual_analysis', ''),
                        'bbox': None
                    })
                    current_state = grid_result
                elif function_name == "crop":
                    # Store cropped grid for later transform
                    cropped_grid = result.get('cropped_grid', current_state)
                    # Don't add substep yet - wait for transform/uncrop
                    current_state = cropped_grid
                elif function_name == "transform":
                    # Transform the current (cropped) grid
                    # The LLM should provide the transformed grid in the function args
                    transformed_grid = function_args.get('grid', result.get('transformed_grid', current_state))
                    substeps.append({
                        'grid': transformed_grid,
                        'tool': function_name,
                        'params': function_args,
                        'reasoning': f"Applied {function_args.get('transformation_type', 'transformation')} transformation",
                        'bbox': None
                    })
                    current_state = transformed_grid
                elif function_name == "uncrop":
                    # Track uncrop result
                    uncropped_grid = result.get('uncropped_grid', current_state)
                    substeps.append({
                        'grid': uncropped_grid,
                        'tool': function_name,
                        'params': function_args,
                        'reasoning': f"Uncropped transformed region back to full grid",
                        'bbox': function_args.get('bbox')
                    })
                    current_state = uncropped_grid
            
            iteration += 1
        
        return {'substeps': substeps if substeps else [{'grid': current_state, 'tool': 'generate_grid', 'params': {}, 'reasoning': 'No transformation needed'}]}
    
    def _execute_tool(self, tool_name: str, args: Dict, current_grid: List[List[int]]) -> Dict[str, Any]:
        """Execute a tool and return result"""
        if tool_name == "generate_grid":
            return {"status": "success", "grid": args['grid'], "message": "Grid generated"}
        
        elif tool_name == "detect_objects":
            return {"status": "success", "objects": args['objects'], "count": len(args['objects'])}
        
        elif tool_name == "match_objects":
            return {"status": "success", "matches": args['matches'], "count": len(args['matches'])}
        
        elif tool_name == "crop":
            bbox = args.get('bbox', [0, 0, len(current_grid)-1, len(current_grid[0])-1])
            grid = args.get('grid', current_grid)
            min_r, min_c, max_r, max_c = bbox
            # Ensure valid indices
            min_r = max(0, min(min_r, len(grid)-1))
            min_c = max(0, min(min_c, len(grid[0])-1))
            max_r = max(min_r, min(max_r, len(grid)-1))
            max_c = max(min_c, min(max_c, len(grid[0])-1))
            cropped = [row[min_c:max_c+1] for row in grid[min_r:max_r+1]]
            return {"status": "success", "cropped_grid": cropped, "bbox": [min_r, min_c, max_r, max_c]}
        
        elif tool_name == "transform":
            # Transform the grid - LLM should provide the transformed grid
            transformed_grid = args.get('grid', current_grid)
            transformation_type = args.get('transformation_type', 'unknown')
            return {"status": "success", "transformed_grid": transformed_grid, "transformation_type": transformation_type}
        
        elif tool_name == "uncrop":
            cropped = args.get('cropped_grid', [])
            original = args.get('original_grid', current_grid)
            bbox = args.get('bbox', [0, 0, len(original)-1, len(original[0])-1])
            min_r, min_c, max_r, max_c = bbox
            
            # Create new grid with cropped region replaced
            result = [row[:] for row in original]
            for i, row in enumerate(cropped):
                if min_r + i < len(result) and i < len(cropped):
                    for j, val in enumerate(row):
                        if min_c + j < len(result[min_r + i]) and j < len(row):
                            result[min_r + i][min_c + j] = val
            
            return {"status": "success", "uncropped_grid": result}
        
        elif tool_name == "create_objects":
            # Create new objects (like copy operation)
            new_grid = args.get('grid', current_grid)
            created_objects = args.get('objects', [])
            return {"status": "success", "grid": new_grid, "created_objects": created_objects}
        
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    
    def _detect_objects(self, grid: List[List[int]]) -> List[Dict]:
        """Detect objects in grid using LLM"""
        content = [
            {
                "type": "text",
                "text": f"""Detect all distinct OBJECTS in this grid.

CRITICAL OBJECT DEFINITION:
- An OBJECT is a distinct connected region or meaningful group of pixels.
- CONTAINMENT: If an object FULLY CONTAINS another object (e.g., a square with a dot inside), and they appear to move/transform together, treat them as ONE COMPOSITE OBJECT.
- SEPARATION: If objects are spatially distinct and transform independently, they are separate.
- GRID STRUCTURE: If the grid lines define cells, each cell (and its content) can be an object.
- MULTICOLOR: Objects can be multicolor if the colors form a single connected shape or pattern.
- BACKGROUND: The background is usually the dominant color (often 0/black) - objects are the distinct regions on top of it.

Your Task:
1. Identify all distinct objects.
2. Check for nested objects (containment). If found, decide if they should be merged into one composite object (e.g. "hollow square with dot").
3. Return the list of objects.

Grid:
{self._format_grid(grid)}

Use the detect_objects tool to return all distinct OBJECTS with their bounding boxes, colors, descriptions, and sizes.
Each object should be a separate, distinct region."""
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(grid_to_image(grid, cell_size=40))}"}
            }
        ]
        
        _, tool_calls = self._call_llm_analysis(content, use_tools=True)
        if tool_calls and tool_calls.function.name == "detect_objects":
            args = json.loads(tool_calls.function.arguments)
            return args.get('objects', [])
        return []
    
    def _match_objects(self, input_objects: List[Dict], output_objects: List[Dict],
                      input_grid: List[List[int]], output_grid: List[List[int]]) -> Dict[int, Optional[int]]:
        """Match objects between input and output using LLM - ONLY match objects that exist in both input and output"""
        content = [
            {
                "type": "text",
                "text": f"""CRITICAL: Match objects between input and output grids.

IMPORTANT RULES:
1. ONLY match output objects that correspond to input objects (objects that exist in BOTH input and output)
2. DO NOT match output objects that are CREATED/NEW (objects that appear in output but NOT in input)
3. New objects in output will be handled separately via create_objects - exclude them from matching
4. Match based on: position, shape, size, and transformation pattern
5. An output object is "new/created" if it has no corresponding input object (different position, completely new shape, etc.)

Input Objects ({len(input_objects)}):
{json.dumps([{k: v for k, v in obj.items() if k != 'cells'} for obj in input_objects], indent=2)}

Output Objects ({len(output_objects)}):
{json.dumps([{k: v for k, v in obj.items() if k != 'cells'} for obj in output_objects], indent=2)}

ANALYSIS:
- Look at the input and output grids carefully
- Identify which output objects correspond to which input objects (even if transformed)
- Identify which output objects are NEW/CREATED (no corresponding input object)
- ONLY match output objects that have a corresponding input object
- For new/created output objects, do NOT include them in matches (they will be created separately)

Use the match_objects tool to return matches. Only include matches for objects that exist in BOTH input and output."""
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(grid_to_image(input_grid, cell_size=40))}"}
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(grid_to_image(output_grid, cell_size=40))}"}
            }
        ]
        
        _, tool_calls = self._call_llm_analysis(content, use_tools=True)
        if tool_calls and tool_calls.function.name == "match_objects":
            args = json.loads(tool_calls.function.arguments)
            matches = args.get('matches', [])
            result = {}
            for match in matches:
                input_idx = match.get('input_idx')
                output_idx = match.get('output_idx')
                result[input_idx] = output_idx
            return result
        return {}
    
    def _filter_objects_by_condition(self, objects: List[Dict], condition: str, full_grid: List[List[int]] = None) -> List[Dict]:
        """Filter objects that match the condition - be VERY SPECIFIC about what matches"""
        if not objects:
            return []
        
        # Use LLM to carefully analyze which objects match the condition
        objects_info = []
        for i, obj in enumerate(objects):
            obj_info = {
                'index': i,
                'description': obj.get('description', ''),
                'colors': obj.get('colors', []),
                'size': obj.get('size', 0),
                'bbox': obj.get('bbox', [])
            }
            objects_info.append(obj_info)
        
        content = [
            {
                "type": "text",
                "text": f"""CRITICAL: You must be VERY SPECIFIC about which objects match this condition.

CONDITION: {condition}

This condition describes EXACTLY which objects should be transformed. Do NOT apply the transformation to objects that look similar but don't match the EXACT condition.

ALL OBJECTS IN THE GRID:
{json.dumps(objects_info, indent=2)}

ANALYZE EACH OBJECT CAREFULLY:
1. Read the condition word-by-word
2. Check if the object's colors EXACTLY match what the condition specifies
3. Check if the object's size EXACTLY matches (if size is specified)
4. Check if the object's shape/structure EXACTLY matches (if shape is specified)
5. Check if the object's position/location EXACTLY matches (if position is specified)

IMPORTANT: 
- "composed only of colors X and Y" means the object contains ONLY those colors, no other colors
- "connected object" means the object's cells are adjacent/connected
- "size exactly N" means the object has exactly N cells
- Pay attention to words like "only", "exactly", "specifically", "each" - these are restrictive

Return a JSON list of object indices (0-based) that EXACTLY match the condition. Be conservative - only include objects that clearly match."""
            }
        ]
        
        # Add full grid image for context
        if full_grid:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(grid_to_image(full_grid, cell_size=40))}"}
            })
        
        # Use LLM to filter
        try:
            response_text, _ = self._call_llm_analysis(content, use_tools=False)
            
            # Try to parse JSON list of indices from response
            import re
            # Look for JSON array of numbers
            json_match = re.search(r'\[[\s\d,]*\]', response_text)
            if json_match:
                try:
                    indices = json.loads(json_match.group())
                    if isinstance(indices, list):
                        filtered = [objects[i] for i in indices if 0 <= i < len(objects)]
                        print(f"      [FILTER] LLM selected {len(filtered)}/{len(objects)} objects matching condition")
                        return filtered
                except:
                    pass
            
            # Fallback: look for "indices:" or numbered list
            indices_match = re.findall(r'\b(\d+)\b', response_text)
            if indices_match:
                try:
                    indices = [int(i) for i in indices_match if i.isdigit()]
                    filtered = [objects[i] for i in indices if 0 <= i < len(objects)]
                    if filtered:
                        print(f"      [FILTER] LLM selected {len(filtered)}/{len(objects)} objects matching condition")
                        return filtered
                except:
                    pass
        except Exception as e:
            print(f"      [WARNING] LLM filtering failed: {e}, using fallback")
        
        # Fallback to enhanced rule-based filtering
        condition_lower = condition.lower()
        filtered = []
        
        for obj in objects:
            obj_desc = obj.get('description', '').lower()
            obj_colors = set(obj.get('colors', []))
            obj_size = obj.get('size', 0)
            
            import re
            
            # Extract key requirements from condition
            color_matches = re.findall(r'color\s*(\d+)', condition_lower)
            required_colors = set(int(c) for c in color_matches) if color_matches else None
            
            # Check for "composed only of" or "only" - very strict
            only_colors = 'composed only' in condition_lower or ('only' in condition_lower and 'color' in condition_lower)
            
            # Check for "exactly" - strict matching
            exactly_size = 'exactly' in condition_lower or 'size' in condition_lower
            
            # Apply strict filters
            matches = True
            
            if required_colors:
                if only_colors:
                    # Object must be composed EXACTLY of these colors (subset and no extras)
                    matches = matches and obj_colors == required_colors
                else:
                    # Object must contain at least one of these colors
                    matches = matches and bool(obj_colors & required_colors)
            
            if exactly_size and 'size' in condition_lower:
                size_match = re.search(r'size\s*(?:exactly|==|=)?\s*(\d+)', condition_lower)
                if size_match:
                    required_size = int(size_match.group(1))
                    matches = matches and obj_size == required_size
            
            # Check for "connected" requirement
            if 'connected' in condition_lower:
                matches = matches and obj_size > 1  # Connected objects have multiple cells
            
            if matches:
                filtered.append(obj)
        
        print(f"      [FILTER] Rule-based selected {len(filtered)}/{len(objects)} objects matching condition")
        return filtered
    
    def _crop_to_object(self, grid: List[List[int]], bbox: List[int]) -> Tuple[List[List[int]], List[int]]:
        """Crop grid to object bounding box"""
        min_r, min_c, max_r, max_c = bbox
        min_r = max(0, min(min_r, len(grid)-1))
        min_c = max(0, min(min_c, len(grid[0])-1))
        max_r = max(min_r, min(max_r, len(grid)-1))
        max_c = max(min_c, min(max_c, len(grid[0])-1))
        
        cropped = [row[min_c:max_c+1] for row in grid[min_r:max_r+1]]
        return cropped, [min_r, min_c, max_r, max_c]
    
    def _transform_object(self, cropped_input: List[List[int]], cropped_output: Optional[List[List[int]]],
                         obj: Dict, output_obj: Optional[Dict], general_step: Dict, is_test: bool,
                         training_booklets: List[Dict] = None, analysis: Dict = None, 
                         test_input_grid: List[List[int]] = None) -> List[List[int]]:
        """Transform a cropped object using LLM"""
        # Identify which cells are likely the object vs background
        # The object is the distinct region we're operating on
        obj_colors = set(obj.get('colors', []))
        object_cells = []
        other_cells = []
        for r in range(len(cropped_input)):
            for c in range(len(cropped_input[r])):
                if cropped_input[r][c] in obj_colors:
                    object_cells.append((r, c))
                else:
                    other_cells.append((r, c))
        
        # Analyze what transformation is needed by comparing input to output
        transformation_description = ""
        if cropped_output:
            # Compare input and output to describe the transformation
            input_obj_colors = set()
            output_obj_colors = set()
            for r in range(len(cropped_input)):
                for c in range(len(cropped_input[r])):
                    if cropped_input[r][c] in obj.get('colors', []):
                        input_obj_colors.add(cropped_input[r][c])
                    if r < len(cropped_output) and c < len(cropped_output[r]):
                        if cropped_output[r][c] in obj.get('colors', []):
                            output_obj_colors.add(cropped_output[r][c])
            
            # Describe the transformation
            if input_obj_colors != output_obj_colors:
                transformation_description = f"\nTRANSFORMATION NEEDED: The object colors change from {sorted(input_obj_colors)} to {sorted(output_obj_colors)}. "
            
            # Check if positions swap
            input_pattern = []
            output_pattern = []
            for r in range(min(len(cropped_input), len(cropped_output))):
                for c in range(min(len(cropped_input[r]), len(cropped_output[r]))):
                    if cropped_input[r][c] in obj.get('colors', []):
                        input_pattern.append((r, c, cropped_input[r][c]))
                    if cropped_output[r][c] in obj.get('colors', []):
                        output_pattern.append((r, c, cropped_output[r][c]))
            
            if input_pattern != output_pattern:
                transformation_description += f"Object cell positions/colors change. "
        
        # Build training booklet examples text for test mode
        training_examples_text = ""
        if is_test and training_booklets:
            training_examples_text = "\n\n=== TRAINING BOOKLET EXAMPLES (for reference - see how similar objects were transformed) ===\n"
            for i, train_booklet in enumerate(training_booklets[:3]):  # Limit to first 3 to avoid token bloat
                # Find steps for the same general step that transformed similar objects
                relevant_steps = [
                    step for step in train_booklet.get('steps', [])
                    if step.get('general_step') == general_step['step_number']
                    and step.get('tool_used') == 'transform'
                ]
                if relevant_steps:
                    training_examples_text += f"\nTraining Example {i+1} - Step {general_step['step_number']}:\n"
                    for step in relevant_steps[:2]:  # Limit to 2 steps per training example
                        if step.get('grid_before') and step.get('grid_after'):
                            training_examples_text += f"  Input: {self._format_grid(step['grid_before'])}\n"
                            training_examples_text += f"  Output: {self._format_grid(step['grid_after'])}\n"
                            if step.get('substep_reasoning'):
                                training_examples_text += f"  Reasoning: {step['substep_reasoning']}\n"
        
        # Build analysis summary text for test mode
        analysis_text = ""
        if is_test and analysis:
            # Use training booklet analysis (short, focused on steps)
            if analysis.get('type') == 'training_booklet_analysis':
                analysis_text = f"""
=== TRAINING BOOKLET ANALYSIS ===
{analysis.get('analysis_text', '')}

GENERAL STEPS:
"""
                for step in analysis.get('rule', []):
                    analysis_text += f"Step {step.get('step_number')}: {step.get('instruction')}\n"
                
                # Add step-specific context for current step
                current_step_summary = next(
                    (s for s in analysis.get('step_summaries', []) 
                     if str(s.get('step_number')) == str(general_step['step_number'])), 
                    None
                )
                if current_step_summary:
                    analysis_text += f"""
CURRENT STEP CONTEXT (Step {general_step['step_number']}):
- Condition: {current_step_summary['condition']}
- Transition: {current_step_summary['transition']}
- Applied {current_step_summary['num_transforms']} times in training examples
- This step transforms objects matching: {current_step_summary['condition']}
"""
            else:
                # Fallback to old analysis format
                if analysis.get('puzzle_type'):
                    analysis_text += f"\nPuzzle Type: {analysis['puzzle_type']}\n"
                if analysis.get('rule'):
                    analysis_text += f"General Rule: {analysis['rule']}\n"
        
        # Build the transform prompt text
        transform_prompt = f"""Transform THIS SPECIFIC OBJECT according to the general step.

CRITICAL: You are transforming a DISTINCT OBJECT.

GENERAL STEP: {general_step['instruction']}
CONDITION: {general_step['condition']}
TRANSFORMATION TYPE: {general_step['transition']}
{transformation_description}

CRITICAL: VERIFY THIS OBJECT EXACTLY MATCHES THE CONDITION

THE EXACT CONDITION FOR THIS STEP:
"{general_step['condition']}"

This condition specifies EXACTLY which objects should be transformed. Read it word-by-word and verify this object matches EVERY part of the condition.

THIS OBJECT'S PROPERTIES:
- Description: {obj.get('description', 'object')}
- Object Colors: {obj.get('colors', [])}
- Object Size: {obj.get('size', 'N/A')} cells
- Object Bounding Box: {obj.get('bbox', [])}

EXACT MATCHING CHECKLIST:
- If condition says "composed only of colors X and Y", verify this object contains EXACTLY those colors and NO OTHER colors
- If condition says "colors X and Y", verify this object contains BOTH colors
- If condition specifies size (e.g., "size exactly N"), verify this object's size matches EXACTLY
- If condition specifies shape/structure (e.g., "connected object", "horizontal line"), verify this object matches EXACTLY
- If condition specifies position (e.g., "at top", "in corner"), verify this object's position matches
- Do NOT transform objects that look similar but don't match the EXACT condition word-by-word

ONLY transform if this object matches the condition EXACTLY. If there's any doubt, or if the object does NOT match, RETURN THE INPUT GRID UNCHANGED.

CROPPED INPUT (before transformation):
{self._format_grid(cropped_input)}

OBJECT CELLS (the distinct object you are transforming): {len(object_cells)} cells with colors {obj.get('colors', [])}
SURROUNDING CELLS (background/other regions, should typically remain unchanged): {len(other_cells)} cells

{'CROPPED OUTPUT (target - this is what the transformed object should look like - USE THIS AS YOUR PRIMARY REFERENCE):' if cropped_output else 'Test mode - apply transformation pattern to the DISTINCT OBJECT:'}
{self._format_grid(cropped_output) if cropped_output else 'N/A'}

CRITICAL: If you have CROPPED OUTPUT above, that is the EXACT target. Compare CROPPED INPUT to CROPPED OUTPUT.
- If CROPPED OUTPUT is IDENTICAL to CROPPED INPUT, you MUST return the input grid UNCHANGED.
- If they are different, apply the exact transformation shown.

{training_examples_text}
{analysis_text}

CRITICAL:
1. Verify object matches condition: "{general_step['condition']}"
2. Compare CROPPED INPUT vs CROPPED OUTPUT above.
3. If target is same as input, OR object doesn't match condition -> Return INPUT grid.
4. If target is different AND object matches -> Apply transformation.
5. Use transform tool with:
   - transformation_type="{general_step['transition']}"
   - grid: The result grid (changed or unchanged)

Transform now."""
        
        content = [
            {
                "type": "text",
                "text": self._sanitize_text(transform_prompt)
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(grid_to_image(cropped_input, cell_size=50))}"}
            }
        ]
        
        if cropped_output:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(grid_to_image(cropped_output, cell_size=50))}"}
            })
        
        # Add test input full grid image for test mode
        if is_test and test_input_grid:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(grid_to_image(test_input_grid, cell_size=40))}"}
            })
        
        response_text, tool_calls = self._call_llm_analysis(content, use_tools=True)
        
        # Empty text response is expected when tool calls are made - that's normal
        if not response_text and tool_calls:
            # This is normal - tool calls don't return text content
            pass
        
        if tool_calls:
            if hasattr(tool_calls, 'function') and tool_calls.function.name == "transform":
                try:
                    args = json.loads(tool_calls.function.arguments)
                    transformed_grid = args.get('grid')
                    
                    if not transformed_grid:
                        print(f"[ERROR] Transform tool call missing 'grid' argument")
                        print(f"[DEBUG] Tool arguments: {tool_calls.function.arguments[:500] if hasattr(tool_calls, 'function') else 'N/A'}")
                        return cropped_input
                    
                    # Validate grid structure
                    if not isinstance(transformed_grid, list) or not transformed_grid:
                        print(f"[ERROR] Transform tool returned invalid grid structure")
                        return cropped_input
                    
                    # Validate that grid actually changed
                    if transformed_grid != cropped_input:
                        print(f"[OK] Transform successful: {len(transformed_grid)}x{len(transformed_grid[0]) if transformed_grid else 0}")
                        return transformed_grid
                    else:
                        print(f"[INFO] Transform tool returned UNCHANGED grid - accepting as correct (no change needed)")
                        return transformed_grid
                        
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"[ERROR] Failed to parse transform tool arguments: {e}")
                    print(f"[DEBUG] Raw arguments: {tool_calls.function.arguments[:500] if hasattr(tool_calls, 'function') and hasattr(tool_calls.function, 'arguments') else 'N/A'}")
                    return cropped_input
            else:
                tool_name = tool_calls.function.name if hasattr(tool_calls, 'function') else 'unknown'
                print(f"[WARNING] Unexpected tool call: {tool_name}, expected 'transform'")
                return cropped_input
        else:
            print(f"[ERROR] No tool calls returned for transform - model should use transform tool")
            return cropped_input
    
    def _uncrop_to_full_grid(self, cropped_grid: List[List[int]], full_grid: List[List[int]], 
                            bbox: List[int]) -> List[List[int]]:
        """Uncrop transformed grid back into full grid"""
        result = [row[:] for row in full_grid]
        min_r, min_c, max_r, max_c = bbox
        
        for i, row in enumerate(cropped_grid):
            if min_r + i < len(result) and i < len(cropped_grid):
                for j, val in enumerate(row):
                    if min_c + j < len(result[min_r + i]) and j < len(row):
                        result[min_r + i][min_c + j] = val
        
        return result
    
    def _create_object(self, current_grid: List[List[int]], new_object_grid: List[List[int]],
                      new_obj: Dict, bbox: List[int], target_output: List[List[int]],
                      training_booklets: List[Dict] = None, analysis: Dict = None) -> List[List[int]]:
        """Create a new object that appears in output but not in input"""
        content = [
            {
                "type": "text",
                "text": f"""CRITICAL: Create a NEW OBJECT that appears in the output but NOT in the input.

This is a NEW OBJECT that must be CREATED/ADDED to the grid.

NEW OBJECT INFORMATION:
- Description: {new_obj.get('description', 'object')}
- Object Colors: {new_obj.get('colors', [])}
- Object Size: {new_obj.get('size', 'N/A')} cells
- Object Bounding Box: {bbox}
- Object Location: This object should be placed at bbox {bbox} in the output

CURRENT GRID STATE (before creating new object):
{self._format_grid(current_grid)}

NEW OBJECT TO CREATE (cropped view from output):
{self._format_grid(new_object_grid)}

TARGET OUTPUT (shows where the new object should be):
{self._format_grid(target_output)}

CRITICAL INSTRUCTIONS:
1. This object does NOT exist in the input - it must be CREATED
2. Look at the TARGET OUTPUT to see exactly where and how this object should appear
3. Use the create_objects tool to add this new object to the current grid
4. The new object should be placed at bbox {bbox}
5. The object colors are: {new_obj.get('colors', [])}
6. The final grid MUST include this new object

CRITICAL: You MUST use the create_objects tool. The tool call MUST include:
1. grid: The COMPLETE grid with the new object added
2. objects: List describing the created object(s)

The grid parameter is REQUIRED and MUST show the current grid WITH the new object added."""
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(grid_to_image(current_grid, cell_size=40))}"}
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(grid_to_image(new_object_grid, cell_size=50))}"}
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(grid_to_image(target_output, cell_size=40))}"}
            }
        ]
        
        # Add training booklet examples if available
        if training_booklets:
            training_examples_text = "\n\nTRAINING BOOKLET EXAMPLES (for reference - see how new objects were created):\n"
            for i, train_booklet in enumerate(training_booklets[:2]):
                create_steps = [s for s in train_booklet.get('steps', []) if s.get('tool_used') == 'create_objects']
                if create_steps:
                    training_examples_text += f"\nTraining Example {i+1} - Create Object Steps:\n"
                    for step in create_steps[:2]:
                        if step.get('grid_before') and step.get('grid_after'):
                            training_examples_text += f"  Before: {self._format_grid(step['grid_before'])}\n"
                            training_examples_text += f"  After: {self._format_grid(step['grid_after'])}\n"
                            if step.get('substep_reasoning'):
                                training_examples_text += f"  Reasoning: {step['substep_reasoning']}\n"
            content[0]["text"] += training_examples_text
        
        response_text, tool_calls = self._call_llm_analysis(content, use_tools=True)
        
        if tool_calls:
            if hasattr(tool_calls, 'function') and tool_calls.function.name == "create_objects":
                try:
                    args = json.loads(tool_calls.function.arguments)
                    created_grid = args.get('grid')
                    
                    if not created_grid:
                        print(f"[ERROR] Create_objects tool call missing 'grid' argument")
                        return current_grid
                    
                    # Validate grid structure
                    if not isinstance(created_grid, list) or not created_grid:
                        print(f"[ERROR] Create_objects tool returned invalid grid structure")
                        return current_grid
                    
                    print(f"[OK] Created new object: {len(created_grid)}x{len(created_grid[0]) if created_grid else 0}")
                    return created_grid
                    
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"[ERROR] Failed to parse create_objects tool arguments: {e}")
                    return current_grid
            else:
                tool_name = tool_calls.function.name if hasattr(tool_calls, 'function') else 'unknown'
                print(f"[WARNING] Unexpected tool call: {tool_name}, expected 'create_objects'")
                return current_grid
        else:
            print(f"[ERROR] No tool calls returned for create_objects - model should use create_objects tool")
            return current_grid
    
    def _create_object_for_test(self, current_grid: List[List[int]], create_step: Dict,
                                training_booklets: List[Dict] = None, analysis: Dict = None,
                                test_input: List[List[int]] = None) -> List[List[int]]:
        """Create objects for test input based on training patterns"""
        content = [
            {
                "type": "text",
                "text": f"""CRITICAL: Create NEW OBJECTS for the test input based on training patterns.

GENERAL STEP: {create_step['instruction']}
CONDITION: {create_step['condition']}

CURRENT TEST GRID STATE:
{self._format_grid(current_grid)}

TEST INPUT (original):
{self._format_grid(test_input) if test_input else 'N/A'}

TRAINING PATTERNS:
Look at the training booklets to understand what new objects were created in training examples.
Apply the same pattern to create new objects in the test input.

CRITICAL INSTRUCTIONS:
1. Analyze the training booklets to see what new objects were created
2. Determine what new objects should be created for the test input
3. Use the create_objects tool to add new objects to the current grid
4. The new objects should follow the same pattern as in training examples

CRITICAL: You MUST use the create_objects tool. The tool call MUST include:
1. grid: The COMPLETE grid with the new object(s) added
2. objects: List describing the created object(s)

The grid parameter is REQUIRED and MUST show the current grid WITH the new object(s) added."""
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(grid_to_image(current_grid, cell_size=40))}"}
            }
        ]
        
        if test_input:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(grid_to_image(test_input, cell_size=40))}"}
            })
        
        # Add training booklet examples
        if training_booklets:
            training_examples_text = "\n\nTRAINING BOOKLET EXAMPLES (showing how new objects were created):\n"
            for i, train_booklet in enumerate(training_booklets[:2]):
                create_steps = [s for s in train_booklet.get('steps', []) if s.get('tool_used') == 'create_objects']
                if create_steps:
                    training_examples_text += f"\nTraining Example {i+1} - Create Object Steps:\n"
                    for step in create_steps[:2]:
                        if step.get('grid_before') and step.get('grid_after'):
                            training_examples_text += f"  Before: {self._format_grid(step['grid_before'])}\n"
                            training_examples_text += f"  After: {self._format_grid(step['grid_after'])}\n"
                            if step.get('substep_reasoning'):
                                training_examples_text += f"  Reasoning: {step['substep_reasoning']}\n"
            content[0]["text"] += training_examples_text
        
        response_text, tool_calls = self._call_llm_analysis(content, use_tools=True)
        
        if tool_calls:
            if hasattr(tool_calls, 'function') and tool_calls.function.name == "create_objects":
                try:
                    args = json.loads(tool_calls.function.arguments)
                    created_grid = args.get('grid')
                    
                    if not created_grid:
                        print(f"[ERROR] Create_objects tool call missing 'grid' argument")
                        return current_grid
                    
                    if not isinstance(created_grid, list) or not created_grid:
                        print(f"[ERROR] Create_objects tool returned invalid grid structure")
                        return current_grid
                    
                    print(f"[OK] Created new object(s) for test: {len(created_grid)}x{len(created_grid[0]) if created_grid else 0}")
                    return created_grid
                    
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"[ERROR] Failed to parse create_objects tool arguments: {e}")
                    return current_grid
        
        return current_grid
    
    def _analyze_training_booklets(self, training_booklets: List[Dict], general_steps: List[Dict]) -> Dict[str, Any]:
        """Short analysis of training booklets by step and overall - for test generation"""
        print("    Analyzing training booklets by step...")
        
        # Build summary of what happened in each step across all training examples
        step_summaries = []
        for step in general_steps:
            step_num = step['step_number']
            step_condition = step['condition']
            step_transition = step['transition']
            
            # Collect all transform steps for this general step across all training booklets
            step_transforms = []
            for i, booklet in enumerate(training_booklets):
                booklet_steps = [s for s in booklet.get('steps', []) 
                               if s.get('general_step') == step_num and s.get('tool_used') == 'transform']
                for transform_step in booklet_steps:
                    step_transforms.append({
                        'training_example': i + 1,
                        'object_num': transform_step.get('object_num'),
                        'object_desc': transform_step.get('instruction', '').replace('TRANSFORM: ', ''),
                        'grid_before': transform_step.get('grid_before'),
                        'grid_after': transform_step.get('grid_after'),
                        'cropped_target': transform_step.get('cropped_grid_target')
                    })
            
            step_summaries.append({
                'step_number': step_num,
                'condition': step_condition,
                'transition': step_transition,
                'num_transforms': len(step_transforms),
                'transforms': step_transforms[:5]  # Limit to first 5 for brevity
            })
        
        # Overall summary
        overall_summary = {
            'total_training_examples': len(training_booklets),
            'total_steps': len(general_steps),
            'accuracy': sum(1 for b in training_booklets if b.get('accuracy') == 1.0) / len(training_booklets) if training_booklets else 0.0,
            'common_patterns': []
        }
        
        # Analyze common patterns across steps
        for step_summary in step_summaries:
            if step_summary['num_transforms'] > 0:
                overall_summary['common_patterns'].append(
                    f"Step {step_summary['step_number']}: {step_summary['condition']} -> {step_summary['transition']} "
                    f"({step_summary['num_transforms']} transformations across {len(training_booklets)} examples)"
                )
        
        # Create concise analysis text
        analysis_text = f"""=== TRAINING BOOKLET ANALYSIS ===

OVERALL SUMMARY:
- {overall_summary['total_training_examples']} training examples processed
- {overall_summary['total_steps']} general steps applied
- Overall accuracy: {overall_summary['accuracy']:.1%}

STEP-BY-STEP SUMMARY:
"""
        for step_summary in step_summaries:
            analysis_text += f"""
Step {step_summary['step_number']}: {step_summary['condition']}
- Transition: {step_summary['transition']}
- Applied {step_summary['num_transforms']} times across {len(training_booklets)} training examples
- This step transforms objects matching: {step_summary['condition']}
"""
        
        analysis_text += "\n=== KEY INSIGHTS ===\n"
        analysis_text += "\n".join(overall_summary['common_patterns'])
        
        return {
            'type': 'training_booklet_analysis',
            'summary': overall_summary,
            'step_summaries': step_summaries,
            'analysis_text': analysis_text,
            'rule': general_steps  # Include general steps as the "rule"
        }
    
    def solve(self, puzzle_file: str, output_dir: str = None) -> Dict[str, Any]:
        """Main solve function"""
        # Default to booklets_ARCAGI/traces if not specified
        if output_dir is None:
            # Get path relative to saturn-arc directory
            base_dir = Path(__file__).parent.parent
            output_dir = str(base_dir / "booklets_ARCAGI" / "traces")
        
        # Load puzzle
        with open(puzzle_file, 'r') as f:
            puzzle_data = json.load(f)
        
        train_examples = puzzle_data.get('train', [])
        test_examples = puzzle_data.get('test', [])
        
        print(f"[LOAD] Loaded puzzle with {len(train_examples)} training examples and {len(test_examples)} test examples")
        
        # Step 1: Comprehensive analysis (visual-first)
        analysis = self.analyze_puzzle(train_examples)
        
        # Step 2: Generate general steps
        general_steps = self.generate_general_steps(analysis)
        
        # Step 3: Generate training booklets
        print("[BOOKLETS] Generating training booklets...")
        training_booklets = []
        training_accuracies = []
        
        # Get unchanged objects from step generation (if available)
        all_unchanged_objs = getattr(self, '_last_unchanged_objects', [])
        
        for i, ex in enumerate(train_examples):
            print(f"  Processing training example {i+1}/{len(train_examples)}...")
            booklet = self.generate_booklet_for_example(ex, general_steps, is_test=False, all_unchanged_objects=all_unchanged_objs)
            # Calculate accuracy for training example
            predicted_grid = booklet.get('current_grid', [])
            expected_grid = ex.get('output', [])
            is_correct = predicted_grid == expected_grid if predicted_grid and expected_grid else False
            booklet['accuracy'] = 1.0 if is_correct else 0.0
            booklet['is_correct'] = is_correct
            booklet['predicted_grid'] = predicted_grid
            booklet['expected_grid'] = expected_grid
            training_accuracies.append(1.0 if is_correct else 0.0)
            if is_correct:
                print(f"    [OK] Training example {i+1}: CORRECT")
            else:
                print(f"    [ERROR] Training example {i+1}: INCORRECT")
                # If this is not the first example and previous ones were correct, this indicates a step problem
                if i > 0 and all(training_accuracies[:-1]):
                    print(f"    [WARNING] Previous examples were correct but this one failed!")
                    print(f"    [WARNING] Steps are not general enough - they work for some examples but not all")
            training_booklets.append(booklet)
        
        # Validation: If steps are correct, ALL training examples should be correct
        if len(training_accuracies) > 1:
            all_correct = all(training_accuracies)
            some_correct = any(training_accuracies)
            if some_correct and not all_correct:
                print(f"\n[VALIDATION ERROR] Steps work for some examples but not all!")
                print(f"  - This indicates the steps are not general enough")
                print(f"  - The transformation rule must work for ALL training examples")
                print(f"  - Review the conditions in the steps - they may be too specific or missing edge cases")
            elif all_correct:
                print(f"\n[VALIDATION OK] All training examples correct - steps are consistent!")
        
        # Step 4: Generate short analysis of training booklets
        print("[TEST] Analyzing training booklets for test generation...")
        training_booklet_analysis = self._analyze_training_booklets(training_booklets, general_steps)
        
        # Step 5: Generate test predictions
        print("[TEST] Generating test predictions...")
        test_booklets = []
        test_accuracies = []
        for i, ex in enumerate(test_examples):
            print(f"  Processing test example {i+1}/{len(test_examples)}...")
            test_booklet = self.generate_booklet_for_example(
                ex, general_steps, is_test=True, 
                training_booklets=training_booklets, 
                analysis=training_booklet_analysis,
                all_unchanged_objects=all_unchanged_objs
            )
            # Calculate accuracy for test example (if expected output available)
            predicted_grid = test_booklet.get('current_grid', [])
            expected_grid = ex.get('output', [])
            test_booklet['predicted_grid'] = predicted_grid
            test_booklet['expected_grid'] = expected_grid
            if expected_grid:
                is_correct = predicted_grid == expected_grid if predicted_grid else False
                test_booklet['accuracy'] = 1.0 if is_correct else 0.0
                test_booklet['is_correct'] = is_correct
                test_accuracies.append(1.0 if is_correct else 0.0)
                if is_correct:
                    print(f"    [OK] Test example {i+1}: CORRECT")
                else:
                    print(f"    [ERROR] Test example {i+1}: INCORRECT")
            else:
                test_booklet['accuracy'] = None
                test_booklet['is_correct'] = None
                print(f"    [INFO] Test example {i+1}: No expected output for comparison")
            test_booklets.append(test_booklet)
        
        # Ensure output_dir exists and is absolute (do this early)
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        puzzle_id = os.path.splitext(os.path.basename(puzzle_file))[0]
        
        # Check if file is starred (protected from overwriting)
        def is_file_starred(file_path):
            """Check if a file is in the starred files list"""
            try:
                # Look for .starred_files.json in traces directory
                traces_dir = Path(output_dir)
                if "traces" not in str(traces_dir):
                    # Try to find traces directory
                    booklets_dir = traces_dir.parent.parent / "booklets_ARCAGI" / "traces"
                    if booklets_dir.exists():
                        traces_dir = booklets_dir
                
                starred_file = traces_dir / ".starred_files.json"
                if starred_file.exists():
                    with open(starred_file, 'r') as f:
                        data = json.load(f)
                        starred_paths = data.get('starred', [])
                        
                        # Normalize file_path for comparison
                        file_path_str = str(file_path).replace('\\', '/')
                        file_name = Path(file_path).name
                        
                        # Try multiple matching strategies
                        for starred_path in starred_paths:
                            starred_str = str(starred_path).replace('\\', '/')
                            starred_name = Path(starred_path).name
                            
                            # Exact match
                            if file_path_str == starred_str or file_path == starred_path:
                                return True
                            
                            # Filename match (handles subdirectories)
                            if file_name == starred_name:
                                return True
                            
                            # Check if paths end the same way (handles relative vs absolute)
                            if file_path_str.endswith(starred_str) or starred_str.endswith(file_path_str):
                                return True
            except Exception as e:
                print(f"[WARNING] Could not check starred status: {e}")
            return False
        
        # Calculate overall accuracies
        training_accuracy = sum(training_accuracies) / len(training_accuracies) if training_accuracies else 0.0
        test_accuracy = sum(test_accuracies) / len(test_accuracies) if test_accuracies else None
        
        result = {
            'puzzle_id': puzzle_id,
            'analysis': analysis,
            'general_steps': general_steps,
            'training_booklets': training_booklets,
            'test_booklets': test_booklets,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'num_training_examples': len(train_examples),
                'num_test_examples': len(test_examples),
                'num_general_steps': len(general_steps),
                'num_training_booklets': len(training_booklets),
                'num_test_booklets': len(test_booklets),
                'analysis_length': len(analysis.get('raw_analysis', '')),
                'has_structured_analysis': bool(analysis.get('structured_analysis')),
                'training_accuracy': training_accuracy,
                'test_accuracy': test_accuracy,
                'training_correct': sum(1 for acc in training_accuracies if acc == 1.0),
                'training_total': len(training_accuracies),
                'test_correct': sum(1 for acc in test_accuracies if acc == 1.0) if test_accuracies else None,
                'test_total': len(test_accuracies) if test_accuracies else None
            }
        }
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Puzzle ID: {puzzle_id}")
        print(f"Training Examples: {len(train_examples)}")
        print(f"Test Examples: {len(test_examples)}")
        print(f"General Steps Generated: {len(general_steps)}")
        print(f"Training Booklets: {len(training_booklets)}")
        print(f"Test Booklets: {len(test_booklets)}")
        print(f"Analysis Length: {len(analysis.get('raw_analysis', ''))} characters")
        if analysis.get('structured_analysis'):
            print(f"Structured Sections: {len(analysis['structured_analysis'])}")
        print("\n" + "-"*60)
        print("ACCURACY RESULTS")
        print("-"*60)
        print(f"Training Accuracy: {training_accuracy:.2%} ({sum(1 for acc in training_accuracies if acc == 1.0)}/{len(training_accuracies)} correct)")
        if test_accuracy is not None:
            print(f"Test Accuracy: {test_accuracy:.2%} ({sum(1 for acc in test_accuracies if acc == 1.0)}/{len(test_accuracies)} correct)")
        else:
            print(f"Test Accuracy: N/A (no expected outputs provided)")
        print("="*60)
        
        # Check if output_dir contains "trial" to add to filename
        import re
        trial_suffix = ""
        if "trial" in output_dir.lower():
            trial_match = re.search(r'trial[_\s]*(\d+)', output_dir.lower())
            if trial_match:
                trial_suffix = f"_trial_{trial_match.group(1)}"
        
        output_file = os.path.join(output_dir, f"{puzzle_id}_v10_analysis{trial_suffix}.json")
        
        # Check if file exists and is starred (protected)
        if os.path.exists(output_file):
            # Get relative path for checking starred status
            try:
                traces_dir = Path(output_dir)
                if "traces" not in str(traces_dir):
                    booklets_dir = traces_dir.parent.parent / "booklets_ARCAGI" / "traces"
                    if booklets_dir.exists():
                        traces_dir = booklets_dir
                
                rel_path = Path(output_file).relative_to(traces_dir) if traces_dir.exists() else Path(output_file)
                rel_path_str = str(rel_path).replace('\\', '/')
                
                if is_file_starred(rel_path_str) or is_file_starred(output_file):
                    # File is starred - save with timestamp to avoid overwriting
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = os.path.join(output_dir, f"{puzzle_id}_v10_analysis{trial_suffix}_{timestamp}.json")
                    print(f"[PROTECTED] File is starred - saving as: {os.path.basename(output_file)}")
            except Exception as e:
                print(f"[WARNING] Could not check if file is starred: {e}")
        
        # Ensure we're saving to booklets_ARCAGI/traces (or subdirectory)
        # This ensures Flask app can always find it
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[OK] Saved results to {output_file}")
            print(f"[DEBUG] File exists: {os.path.exists(output_file)}")
            print(f"[DEBUG] File size: {os.path.getsize(output_file)} bytes")
        except Exception as e:
            print(f"[ERROR] Failed to save file: {e}")
            # Try saving to default location as fallback
            fallback_dir = str(Path(__file__).parent.parent / "booklets_ARCAGI" / "traces")
            os.makedirs(fallback_dir, exist_ok=True)
            fallback_file = os.path.join(fallback_dir, f"{puzzle_id}_v10_analysis{trial_suffix}.json")
            try:
                with open(fallback_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"[OK] Saved results to fallback location: {fallback_file}")
                output_file = fallback_file
            except Exception as e2:
                print(f"[ERROR] Failed to save to fallback location: {e2}")
        
        return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ARC Comprehensive Solver V10')
    parser.add_argument('puzzle_file', help='Path to puzzle JSON file')
    parser.add_argument('--output_dir', '-o', default=None, 
                       help='Output directory for results (default: booklets_ARCAGI/traces)')
    
    args = parser.parse_args()
    
    puzzle_file = args.puzzle_file
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default to booklets_ARCAGI/traces
        base_dir = Path(__file__).parent.parent
        output_dir = str(base_dir / "booklets_ARCAGI" / "traces")
    
    # Ensure output_dir is absolute and exists
    output_dir = os.path.abspath(output_dir)
    print(f"[INFO] Output directory: {output_dir}")
    
    solver = ARCComprehensiveSolverV10()
    result = solver.solve(puzzle_file, output_dir)

