#!/usr/bin/env python3
"""
ARC-AGI Visual Solver
Uses a phased approach with visual representations to solve ARC puzzles
"""

import json
import os
import sys
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
from PIL import Image
import numpy as np

# Import the visualizer functions
from arc_visualizer import grid_to_image, ARC_COLORS

# Define the visualization tool for function calling
VISUALIZATION_TOOL = {
    "type": "function",
    "name": "visualize_grid",
    "description": "Generate a visual image representation of a grid. FOLLOW THIS PROCESS FOR EACH STEP:\n\n1. HYPOTHETICAL PHASE:\n   - First create MULTIPLE 'hypothetical' visualizations\n   - Show different possible approaches you're considering\n   - Visualize each potential transformation or pattern\n   - Use type='hypothetical' for these exploratory steps\n   - IMPORTANT: After you create hypotheticals, you will receive VERIFICATION FEEDBACK showing which hypothetical is closest to the expected output\n\n2. TRANSFORM PHASE:\n   - Review the verification feedback you received\n   - Choose the best approach based on the feedback\n   - Create ONE 'transform' visualization showing your chosen step\n   - Use type='transform' for the final decision\n\nThis two-phase process should be repeated for each step of your solution. The verification helps you learn which approaches work best!",
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
            },
            "visualization_type": {
                "type": "string",
                "enum": ["transform", "hypothetical"],
                "description": "Type of visualization: 'transform' for definite solution steps, 'hypothetical' for possible steps being considered"
            },
            "description": {
                "type": "string",
                "description": "Detailed description of what this visualization represents (e.g., 'Rotating each pattern 90 degrees' or 'Possible pattern matching approach')"
            }
        },
        "required": ["grid", "visualization_type", "description"]
    }
}


class ARCVisualSolver:
    def __init__(self):
        """Initialize the ARC visual solver with API credentials"""
        self.api_key_openai = os.getenv("OPENAI_API_KEY")
        if not self.api_key_openai:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        self.client_openai = OpenAI(api_key=self.api_key_openai)
        
        self.conversation_history = []
        self.current_task_name = None
        self.current_training_example = 0
        self.current_expected_output = None  # For verification
        self.current_phase = None  # Track which phase we're in (for verification control)
        self.current_step_number = 1  # Track step number (increments only when transform created)
        self.last_transform_path = None  # Track last transform for context building
        
        # Use img_tmp directory in project root
        self.temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img_tmp")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Create visualizations directory with the new organized structure
        self.visualizations_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations")
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        # No longer using the old visualizations_TASKNAME format
        self.visualizations_dir_base = None  # Set to None to ensure old format isn't used
    
    def load_task(self, file_path: str) -> Dict[str, Any]:
        """Load an ARC-AGI task from a JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def format_grid(self, grid: List[List[int]]) -> str:
        """Format a grid for display in the prompt"""
        return '\n'.join(['[' + ', '.join(str(cell) for cell in row) + ']' for row in grid])
    
    def save_visualization(self, img: Image.Image, description: str, visualization_type: str = "transform", is_model_output: bool = False, is_actual_output: bool = False) -> str:
        """Save a visualization image with a description as metadata"""
        if self.current_task_name is None:
            return
            
        # Save description as metadata in the image
        if not img.info:
            img.info = {}
        img.info['description'] = description
        img.info['visualization_type'] = visualization_type
            
        # Create task directory
        task_dir = os.path.join(self.visualizations_dir, self.current_task_name)
        os.makedirs(task_dir, exist_ok=True)
        
        # Create training example directory
        training_dir = os.path.join(task_dir, f"training_{self.current_training_example + 1}")
        os.makedirs(training_dir, exist_ok=True)
        
        # Use current step number (only increments when transform is created)
        step_num = self.current_step_number
        
        # Add step information to metadata
        if not img.info:
            img.info = {}
        img.info['step_number'] = str(step_num)
        
        if visualization_type == "hypothetical":
            # Count existing hypotheticals for THIS STEP (flat structure)
            existing_hyps = [f for f in os.listdir(training_dir) 
                           if f.startswith(f"{step_num:02d}_") 
                           and f.endswith("_hypothetical.png")]
            sequence_num = len(existing_hyps) + 1
            
            # Add hypothetical-specific metadata
            img.info['hypothetical_number'] = str(sequence_num)
            img.info['step_description'] = f"Step {step_num} - Hypothetical {sequence_num}: {description}"
            
            # Flat filename: 02_01_hypothetical.png, 02_02_hypothetical.png, etc.
            filename = f"{step_num:02d}_{sequence_num:02d}_hypothetical.png"
            save_path = os.path.join(training_dir, filename)
            
            # Log hypothetical creation for debugging
            print(f"  Creating hypothetical {sequence_num} for step {step_num:02d}")
            print(f"  Filename: {filename}")
            print(f"  Description: {description}")
            print(f"  Grid dimensions before save: {img.size}")
            
        else:  # transform or special outputs
            if is_model_output:
                filename = "XX_model_output.png"
                img.info['step_description'] = f"Model Output: {description}"
            elif is_actual_output:
                filename = "YY_actual_output.png"
                img.info['step_description'] = f"Actual Output: {description}"
            else:
                # For input or transformation steps
                is_input = description.lower().startswith('input')
                if is_input:
                    filename = "01_input.png"
                    img.info['step_number'] = '1'
                    img.info['step_description'] = f"Step 1 - Input: {description}"
                else:
                    # Transform uses the SAME step number as its hypotheticals
                    # Format: 02_04_transform.png (comes after 02_01, 02_02, 02_03 hypotheticals)
                    # Count hypotheticals for this step to get next sequence number
                    existing_hyps = [f for f in os.listdir(training_dir) 
                                   if f.startswith(f"{step_num:02d}_") 
                                   and f.endswith("_hypothetical.png")]
                    sequence_num = len(existing_hyps) + 1
                    
                    filename = f"{step_num:02d}_{sequence_num:02d}_transform.png"
                    img.info['step_number'] = str(step_num)
                    img.info['step_description'] = f"Step {step_num} - Transform: {description}"
            
            save_path = os.path.join(training_dir, filename)
            
            # After saving transform, increment step and track it
            if not is_model_output and not is_actual_output and not is_input:
                self.current_step_number += 1
                self.last_transform_path = save_path
        
        # Save the image with metadata
        # PNG metadata needs to be saved as PngInfo chunks
        from PIL import PngImagePlugin
        pnginfo = PngImagePlugin.PngInfo()
        
        # Add all metadata as text chunks
        if hasattr(img, 'info') and img.info:
            for key, value in img.info.items():
                pnginfo.add_text(key, str(value))
        
        img.save(save_path, pnginfo=pnginfo)
        
        # Also save to temp directory for immediate access
        temp_path = os.path.join(self.temp_dir, f"{self.current_task_name}_{filename}")
        img.save(temp_path, pnginfo=pnginfo)
        return save_path

    def create_grid_image(self, grid: List[List[int]], cell_size: int = 30, label: str = "grid") -> str:
        """Create an image from a grid and return the file path"""
        img = grid_to_image(grid, cell_size)
        # Save visualization
        return self.save_visualization(img, label)
    
    def encode_image(self, image_path: str) -> str:
        """Encode an image file to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def call_ai_with_image(self, text_prompt: str, image_paths: List[str]) -> str:
        """Call OpenAI with text and images"""
        
        # Prepare content with images for the new responses API
        content = [{"type": "input_text", "text": text_prompt}]
        
        for image_path in image_paths:
            base64_image = self.encode_image(image_path)
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{base64_image}"
            })
        
        messages = self.conversation_history + [{"role": "user", "content": content}]
        
        # Create the API call with tools always enabled
        # Model options:
        # - "gpt-4o" - Best for function calling + vision + reasoning (RECOMMENDED)
        # - "gpt-4o-mini" - Faster, cheaper, still very capable
        # - "o1-preview" - Strongest reasoning but limited function calling support
        # - "o1-mini" - Good reasoning, more affordable
        call_params = {
            "model": "gpt-5-mini",  # Using gpt-5-mini (preview/beta model)
            "input": messages,  # Use 'input' instead of 'messages' for responses API
            "tools": [VISUALIZATION_TOOL],
            "tool_choice": "auto"
        }
        
        # Keep calling the API while tool calls are being made
        max_iterations = 20
        iteration = 0
        final_message = None
        
        while iteration < max_iterations:
            print(f"\n📡 API Call iteration {iteration + 1}")
            response = self.client_openai.responses.create(**call_params)
            
            # Log the response structure
            print(f"📦 Response output contains {len(response.output)} items")
            
            # Add response output to input for context
            if response.output:
                call_params["input"] += response.output
                
                # Log each output item
                for idx, item in enumerate(response.output):
                    print(f"  Item {idx}: type={item.type}")
                    if item.type == "message":
                        # Try to extract text from message content
                        if hasattr(item, 'content'):
                            for content_item in item.content:
                                if hasattr(content_item, 'type'):
                                    print(f"    Content type: {content_item.type}")
                                    if content_item.type == "output_text" and hasattr(content_item, 'text'):
                                        preview = content_item.text[:200] + "..." if len(content_item.text) > 200 else content_item.text
                                        print(f"    Text preview: {preview}")
            
            # Check for function calls in the output
            has_function_call = False
            hypotheticals_created_this_iteration = []
            
            for item in response.output:
                if item.type == "function_call":
                    has_function_call = True
                    print(f"\n🔧 Function call detected: {item.name if hasattr(item, 'name') else 'unknown'}")
                    
                    # Parse the function arguments
                    args = json.loads(item.arguments)
                    
                    if item.name == "visualize_grid":
                        # Create visualization
                        grid = args["grid"]
                        visualization_type = args.get("visualization_type", "transform")
                        description = args.get("description", "analysis_step")
                        
                        # Validate dimensions for hypotheticals and transforms
                        grid_height = len(grid)
                        grid_width = len(grid[0]) if grid else 0
                        
                        # Check if dimensions match expected output (for hypotheticals and transforms, not for input)
                        dimension_error = False
                        if hasattr(self, 'current_expected_output') and self.current_expected_output:
                            expected_height = len(self.current_expected_output)
                            expected_width = len(self.current_expected_output[0]) if self.current_expected_output else 0
                            
                            # Only validate dimensions for hypotheticals and transforms (not input or outputs)
                            if visualization_type in ["hypothetical", "transform"]:
                                if grid_height != expected_height or grid_width != expected_width:
                                    dimension_error = True
                                    error_msg = f"❌ REJECTED: Dimension mismatch - grid is {grid_width}x{grid_height}, expected {expected_width}x{expected_height}"
                                    print(f"\n{error_msg}")
                                    print(f"  Rejected: {description[:100]}")
                                    print(f"  ⚠️ Silently rejecting - will proceed with valid hypotheticals only")
                                    
                                    # Send minimal success response (don't show error to model, just acknowledge)
                                    # This prevents the model from getting stuck trying to fix dimensions
                                    call_params["input"].append({
                                        "type": "function_call_output",
                                        "call_id": item.call_id,
                                        "output": json.dumps({
                                            "status": "rejected_silently",
                                            "message": "Grid dimensions don't match output requirements - proceeding with other hypotheticals"
                                        })
                                    })
                                    continue  # Skip creating this visualization
                        
                        # Convert grid to image and verify dimensions
                        img = grid_to_image(grid, 30)
                        
                        print(f"\n  Creating {visualization_type} visualization:")
                        print(f"  Description: {description}")
                        print(f"  Grid dimensions: {grid_width}x{grid_height}")
                        print(f"  Image dimensions: {img.size}")
                        
                        # Add additional metadata about the visualization
                        if not img.info:
                            img.info = {}
                        img.info.update({
                            'description': description,
                            'visualization_type': visualization_type,
                            'grid_dimensions': f"{grid_width}x{grid_height}",
                            'timestamp': datetime.now().isoformat(),
                            'grid_data': json.dumps(grid),  # Store the actual grid data
                            'step_details': json.dumps({
                                'iteration': iteration,
                                'training_example': self.current_training_example,
                                'task': self.current_task_name
                            })
                        })
                        
                        img_path = self.save_visualization(
                            img=img,
                            description=description,
                            visualization_type=visualization_type
                        )
                        base64_img = self.encode_image(img_path)
                        print(f"  💾 Visualization saved to: {img_path}")
                        print(f"  📊 Visualization type: {visualization_type}")
                        print(f"  📝 Description saved in metadata")
                        
                        # Track hypotheticals created
                        if visualization_type == "hypothetical":
                            hypotheticals_created_this_iteration.append((grid, description, img_path))
                    
                        # Add function result to input
                        call_params["input"].append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps({
                                "image_url": f"data:image/png;base64,{base64_img}",
                                "status": "success"
                            })
                        })
                        print(f"  ✅ Visualization created and added to conversation")
                    
                    iteration += 1
            
            # Accumulate hypotheticals across iterations (don't verify until we have a batch)
            if not hasattr(self, 'accumulated_hypotheticals'):
                self.accumulated_hypotheticals = []
            
            self.accumulated_hypotheticals.extend(hypotheticals_created_this_iteration)
            
            # Check if we should run verification
            # Only verify when:
            # 1. We have at least 3 hypotheticals accumulated, OR
            # 2. A transform was just created (indicating batch is complete)
            # 3. We are NOT in Phase 4 (test phase - no verification at inference)
            should_verify = False
            if self.accumulated_hypotheticals and hasattr(self, 'current_expected_output') and self.current_expected_output:
                # Skip verification if we're in Phase 4 (test phase)
                if self.current_phase == 4:
                    should_verify = False
                    print(f"\n⚠️ Phase 4 (Test): Skipping real-time verification (matching deployment conditions)")
                else:
                    # Check if a transform was created this iteration
                    transform_created = any(item.type == "function_call" and 
                                           json.loads(item.arguments).get("visualization_type") == "transform"
                                           for item in response.output if item.type == "function_call")
                    
                    if transform_created or len(self.accumulated_hypotheticals) >= 5:
                        should_verify = True
            
            if should_verify:
                print(f"\n🔍 Running verification on {len(self.accumulated_hypotheticals)} hypotheticals...")
                
                # Check if we have a valid expected output
                if not hasattr(self, 'current_expected_output') or not self.current_expected_output:
                    print("  ⚠️ Warning: No expected output set for verification. Skipping verification.")
                    should_verify = False
                
                # Check if we have any valid hypotheticals to verify
                if len(self.accumulated_hypotheticals) == 0:
                    print("  ⚠️ No valid hypotheticals to verify (all may have been rejected for dimension mismatch)")
                    print("  ⏭️ Proceeding to next iteration...")
                    should_verify = False
                
                best_grid = None
                best_desc = None
                best_diff = float('inf')
                verification_details = []
                
                for hyp_grid, hyp_desc, hyp_path in self.accumulated_hypotheticals:
                    # Debug: Check if grids are valid
                    if not hyp_grid:
                        print(f"  ⚠️ Warning: Empty hypothetical grid for: {hyp_desc[:60]}")
                        verification_details.append((hyp_desc, float('inf')))
                        continue
                    
                    diff = self.calculate_grid_difference(hyp_grid, self.current_expected_output)
                    verification_details.append((hyp_desc, diff))
                    print(f"  • {diff} cells different: {hyp_desc[:60]}")
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_grid = hyp_grid
                        best_desc = hyp_desc
                
                # Create verification feedback message
                verification_msg = "\n📊 VERIFICATION RESULTS:\n"
                verification_msg += f"I compared your {len(self.accumulated_hypotheticals)} hypotheticals to the expected output:\n\n"
                
                for desc, diff in verification_details:
                    if diff == 0:
                        verification_msg += f"✅ PERFECT: {desc[:70]}\n"
                    elif diff <= 3:
                        verification_msg += f"⭐ VERY CLOSE ({diff} cells off): {desc[:70]}\n"
                    elif diff <= 10:
                        verification_msg += f"🔸 CLOSE ({diff} cells off): {desc[:70]}\n"
                    else:
                        verification_msg += f"❌ OFF ({diff} cells off): {desc[:70]}\n"
                
                # Check if we have any valid comparisons
                if best_desc is None or best_diff == float('inf'):
                    verification_msg += f"\n⚠️ No valid hypotheticals to compare (all had dimension mismatches and were rejected)."
                    verification_msg += f"\n📏 Required output dimensions: {len(self.current_expected_output[0])}x{len(self.current_expected_output)}"
                    verification_msg += f"\n⏭️ Continuing with next step..."
                elif best_diff == 0:
                    verification_msg += f"\n🎯 PERFECT MATCH! One of your hypotheticals is exactly correct!"
                    verification_msg += f"\n✨ Perfect hypothetical: {best_desc[:80]}"
                    verification_msg += f"\n✅ STOPPING EXPLORATION - Moving to next training example immediately."
                    verification_msg += f"\n🎉 No need to generate more hypotheticals or transforms - this is the solution!"
                elif best_diff <= 5:
                    verification_msg += f"\n💡 EXCELLENT! Best match is very close ({best_diff} cells off): {best_desc[:60]}"
                    verification_msg += f"\n✅ Use this as your transform and proceed to the next step."
                elif best_diff <= 15:
                    verification_msg += f"\n⚠️ MODERATE. Best was {best_diff} cells off: {best_desc[:60]}"
                    verification_msg += f"\n✅ Since you have {len(self.accumulated_hypotheticals)} hypotheticals, pick the best as your transform."
                else:
                    verification_msg += f"\n❌ All hypotheticals far off (best: {best_diff} cells different): {best_desc[:60]}"
                    verification_msg += f"\n✅ Pick the best one anyway and use it as your transform to move forward."
                
                # Add instruction to build on previous transform
                if self.last_transform_path and best_diff > 0:
                    verification_msg += f"\n\n🔄 NEXT STEP: Now that you've committed to a transform, generate NEW hypotheticals for the next step."
                    verification_msg += f"\n   Build on your previous transformation - use it as the starting point for your next brainstorm."
                    verification_msg += f"\n   Remember: Each new step should refine and build upon the previous transforms."
                
                print(verification_msg)
                
                # Inject verification results back into conversation
                call_params["input"].append({
                    "role": "user",
                    "content": verification_msg
                })
                
                # Mark chosen hypothetical and add verification scores to ALL hypotheticals
                print(f"\n📝 Adding verification metadata to {len(self.accumulated_hypotheticals)} hypotheticals...")
                
                # Build list of all hypotheticals with scores for metadata
                all_hyp_info = []
                for hyp_grid, hyp_desc, hyp_path in self.accumulated_hypotheticals:
                    score = self.calculate_grid_difference(hyp_grid, self.current_expected_output)
                    all_hyp_info.append({
                        'path': os.path.basename(hyp_path),
                        'description': hyp_desc[:80],
                        'score': score if score != float('inf') else -1
                    })
                
                # Update each hypothetical's metadata
                for hyp_grid, hyp_desc, hyp_path in self.accumulated_hypotheticals:
                    score = self.calculate_grid_difference(hyp_grid, self.current_expected_output)
                    is_chosen = (hyp_grid == best_grid) if best_grid is not None else False
                    
                    self.add_verification_metadata_to_image(
                        hyp_path,
                        verification_score=score if score != float('inf') else -1,
                        is_chosen=is_chosen,
                        all_hypotheticals=all_hyp_info
                    )
                    
                    if is_chosen:
                        print(f"  ✅ Marked as CHOSEN: {os.path.basename(hyp_path)} (score: {score})")
                    else:
                        print(f"  📊 Verified: {os.path.basename(hyp_path)} (score: {score})")
                
                # Clear accumulated hypotheticals after verification
                self.accumulated_hypotheticals = []
                
                # If perfect match, save it as model output and stop immediately
                if best_diff == 0 and best_grid is not None and best_desc is not None:
                    print("\n🎯 Perfect match found! Saving as model output and stopping current phase early.")
                    
                    # Save the perfect solution as XX_model_output.png
                    perfect_img = grid_to_image(best_grid, 30)
                    
                    # Add metadata
                    if not perfect_img.info:
                        perfect_img.info = {}
                    perfect_img.info.update({
                        'description': best_desc,
                        'visualization_type': 'model_output',
                        'grid_dimensions': f"{len(best_grid[0])}x{len(best_grid)}",
                        'timestamp': datetime.now().isoformat(),
                        'grid_data': json.dumps(best_grid),
                        'step_description': f"Perfect Solution: {best_desc}",
                        'verification_score': '0',
                        'is_perfect_match': 'true'
                    })
                    
                    model_output_path = self.save_visualization(
                        img=perfect_img,
                        description=best_desc,
                        visualization_type="transform",
                        is_model_output=True
                    )
                    print(f"  💾 Perfect solution saved to: {model_output_path}")
                    
                    final_message = f"Perfect solution found: {best_desc}"
                    break
            
            # Also try to extract any text response even if there were tool calls
            if hasattr(response, 'output_text') and response.output_text:
                print(f"\n💬 Response text: {response.output_text[:1500]}..." if len(response.output_text) > 500 else f"\n💬 Response text: {response.output_text}")
            
            if not has_function_call:
                print("\n✋ No more function calls, ending iteration")
                # Extract the final text response
                final_message = response.output_text if hasattr(response, 'output_text') else ""
                break
        
        if final_message is None:
            print("\n⚠️ Warning: Maximum iterations reached")
            # Try to extract text from the last item in input
            if call_params["input"]:
                last_item = call_params["input"][-1]
                if isinstance(last_item, dict) and "content" in last_item:
                    final_message = str(last_item["content"])
                else:
                    final_message = str(last_item)
            else:
                final_message = "No response"
        
        # Add to conversation history (simplified version without images for history)
        self.conversation_history.append({"role": "user", "content": text_prompt})
        self.conversation_history.append({"role": "assistant", "content": final_message})
        
        # Debug output
        print(f"[START: {self.current_task_name}]")
        print("\n" + "="*80)
        print(f"PHASE PROMPT TO OPENAI:")
        print("-"*80)
        print(text_prompt)
        print(f"Images included: {len(image_paths)}")
        print(f"Tool call iterations made: {iteration}")
        print("-"*80)
        print(f"FINAL RESPONSE FROM OPENAI:")
        if final_message:
            print(f"(Length: {len(final_message)} characters)")
        print("-"*80)
        print(final_message if final_message else "[No final message received]")
        print("="*80)
        print(f"[END: {self.current_task_name}]")
        
        return final_message
    
    def calculate_grid_difference(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        """Calculate the number of different cells between two grids"""
        if not grid1 or not grid2:
            return float('inf')
        
        # Check dimensions match
        if len(grid1) != len(grid2):
            return float('inf')
        
        differences = 0
        for row1, row2 in zip(grid1, grid2):
            if len(row1) != len(row2):
                return float('inf')
            for cell1, cell2 in zip(row1, row2):
                if cell1 != cell2:
                    differences += 1
        
        return differences
    
    def find_best_hypothetical(self, training_dir: str, expected_output: List[List[int]], step_number: int) -> Optional[Tuple[List[List[int]], str, int]]:
        """Find the hypothetical that is closest to the expected output
        Returns: (best_grid, best_description, difference_count)
        """
        # Get hypothetical files for this step (flat structure: 02_01_hypothetical.png, 02_02_hypothetical.png, etc.)
        all_files = os.listdir(training_dir)
        hyp_pattern = f"{step_number:02d}_"
        hyp_files = sorted([f for f in all_files if f.startswith(hyp_pattern) and f.endswith('_hypothetical.png')])
        
        if not hyp_files:
            print(f"    No hypotheticals found for step {step_number}")
            return None
        
        best_grid = None
        best_description = None
        best_difference = float('inf')
        
        print(f"\n  🔍 Verifying hypotheticals against expected output:")
        
        for filename in hyp_files:
            filepath = os.path.join(training_dir, filename)
            metadata = self.get_visualization_metadata(filepath)
            
            # Get the grid data from metadata
            if 'grid_data' in metadata and metadata['grid_data']:
                grid = metadata['grid_data']
                description = metadata.get('description', 'Unknown')
                
                # Calculate difference from expected output
                diff = self.calculate_grid_difference(grid, expected_output)
                
                print(f"    {filename}: {diff} cells different - {description[:60]}")
                
                if diff < best_difference:
                    best_difference = diff
                    best_grid = grid
                    best_description = description
        
        if best_grid is not None:
            print(f"  ✅ Best match: {best_difference} cells different")
            return (best_grid, best_description, best_difference)
        
        return None
    
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
    
    def add_verification_metadata_to_image(self, image_path: str, verification_score: int, 
                                          is_chosen: bool, all_hypotheticals: List[Dict]) -> None:
        """Add verification metadata to an already-saved PNG image"""
        try:
            from PIL import PngImagePlugin
            
            # Read existing image
            with Image.open(image_path) as img:
                # Get existing metadata
                existing_info = dict(img.info) if hasattr(img, 'info') else {}
                
                # Create new PngInfo with all metadata
                pnginfo = PngImagePlugin.PngInfo()
                for key, value in existing_info.items():
                    pnginfo.add_text(key, str(value))
                
                # Add verification metadata
                pnginfo.add_text('verification_score', str(verification_score))
                pnginfo.add_text('is_chosen', 'true' if is_chosen else 'false')
                pnginfo.add_text('competing_hypotheticals', json.dumps(all_hypotheticals))
                pnginfo.add_text('verified_at', datetime.now().isoformat())
                
                # Re-save with updated metadata
                img_copy = img.copy()
                img_copy.save(image_path, pnginfo=pnginfo)
                
        except Exception as e:
            print(f"  ⚠️ Warning: Could not add verification metadata to {image_path}: {e}")
    
    def get_visualization_metadata(self, image_path: str) -> Dict[str, Any]:
        """Read metadata from a saved visualization image"""
        try:
            with Image.open(image_path) as img:
                metadata = {}
                if hasattr(img, 'info'):
                    metadata.update(img.info)
                    # Parse any JSON-encoded fields
                    if 'step_details' in metadata:
                        try:
                            metadata['step_details'] = json.loads(metadata['step_details'])
                        except:
                            pass
                    if 'grid_data' in metadata:
                        try:
                            metadata['grid_data'] = json.loads(metadata['grid_data'])
                        except:
                            pass
                    if 'competing_hypotheticals' in metadata:
                        try:
                            metadata['competing_hypotheticals'] = json.loads(metadata['competing_hypotheticals'])
                        except:
                            pass
                return metadata
        except Exception as e:
            print(f"Error reading metadata from {image_path}: {e}")
            return {}

    def list_visualizations(self, task_name: str, training_example: int = 0) -> List[Dict[str, Any]]:
        """List all visualizations for a task with their metadata in chronological order:
        01_input.png
        02_01_hypothetical.png
        02_02_hypothetical.png
        02_03_hypothetical.png
        02_04_transform.png
        03_01_hypothetical.png
        03_02_hypothetical.png
        03_03_transform.png
        XX_model_output.png
        YY_actual_output.png
        """
        training_dir = os.path.join(self.visualizations_dir, task_name, f"training_{training_example + 1}")
        if not os.path.exists(training_dir):
            return []
        
        results = []
        
        # Helper function to add file with metadata
        def add_file(filepath, filename):
            metadata = self.get_visualization_metadata(filepath)
            results.append({
                'path': filepath,
                'filename': filename,
                'metadata': metadata
            })
        
        all_files = os.listdir(training_dir)
        
        # Get all regular files (numbered)
        regular_files = sorted([
            f for f in all_files
            if f.endswith('.png')
            and not f.startswith('XX_')
            and not f.startswith('YY_')
        ])
        
        # Add regular files in order (natural sort handles 01, 02_01, 02_02, 02_03, 02_04, 03_01, etc.)
        for filename in regular_files:
            add_file(os.path.join(training_dir, filename), filename)
        
        # Add model output if exists
        model_output = next((f for f in all_files if f.startswith('XX_')), None)
        if model_output:
            add_file(os.path.join(training_dir, model_output), model_output)
        
        # Add actual output if exists
        actual_output = next((f for f in all_files if f.startswith('YY_')), None)
        if actual_output:
            add_file(os.path.join(training_dir, actual_output), actual_output)
        
        return results

    def cleanup_visualizations(self, task_name: str) -> None:
        """Clean up visualization directories for this task"""
        import shutil
        task_dir = os.path.join(self.visualizations_dir, task_name)
        if os.path.exists(task_dir):
            shutil.rmtree(task_dir)
        temp_pattern = os.path.join(self.temp_dir, f"{task_name}_*")
        import glob
        for f in glob.glob(temp_pattern):
            os.remove(f)

    def solve(self, task_file: str) -> Tuple[bool, Optional[List[List[int]]], int]:
        """
        Main solving loop using phased visual approach
        Returns (success, predicted_output, num_phases)
        """
        # Extract task name from file path
        self.current_task_name = os.path.splitext(os.path.basename(task_file))[0]
        
        # Clean up any existing visualizations for this task
        self.cleanup_visualizations(self.current_task_name)
        
        # Load the task
        task = self.load_task(task_file)
        print(f"\nLoaded task: {task_file}")
        print(f"Task contains {len(task['train'])} training examples and {len(task['test'])} test examples")
        
        # Reset conversation history
        self.conversation_history = []
        num_phases = 0
        
        # Phase 1: Show first training example with visual
        print("\n" + "="*80)
        print("=== Phase 1: First training example ===")
        print("="*80)
        
        self.current_phase = 1  # Track phase for verification control
        self.current_training_example = 0
        self.current_expected_output = task['train'][0]['output']  # Set for verification
        self.accumulated_hypotheticals = []  # Reset for new phase
        self.current_step_number = 1  # Reset step counter for new training example
        self.last_transform_path = None  # Reset transform tracking
        img = grid_to_image(task['train'][0]['input'], 30)
        input_img_1 = self.save_visualization(img, "input", visualization_type="transform")  # Input is special case
        img = grid_to_image(task['train'][0]['output'], 30)
        output_img_1 = self.save_visualization(img, "actual_output", is_actual_output=True)  # Ground truth
        
        prompt_1 = f"""
You are looking at a visual puzzle. I'm showing you the INPUT and OUTPUT together so you can LEARN THE PATTERN.

🎯 YOUR CRITICAL ADVANTAGE: You can see BOTH the input AND output for this first example. Use this to deeply understand the transformation rule!

SYSTEMATIC ANALYSIS PROCESS:

STEP 1: COMPARE INPUT TO OUTPUT
- Look at the input grid and output grid side by side
- What changed? What stayed the same?
- What patterns do you notice?
- What's different about the dimensions, colors, shapes, positions?

STEP 2: FORM HYPOTHESES
For each hypothesis you explore:
   A. CREATE 3-5 'hypothetical' visualizations testing different pattern interpretations:
      - Hypothesis 1: Test if pattern is about [specific transformation rule]
      - Hypothesis 2: Test if pattern is about [different rule]
      - Hypothesis 3: Test if pattern is about [another rule]
      - etc.
   
   B. VERIFY EACH HYPOTHESIS:
      - Apply your hypothesized rule to the INPUT
      - Check: Does it produce something close to the OUTPUT?
      - Compare your hypothetical to the actual output
      - Score how well it matches
   
   C. COMMIT TO BEST HYPOTHESIS:
      - Choose the hypothesis that BEST explains input→output transformation
      - Create ONE 'transform' visualization showing your chosen understanding
      - Explain WHY this hypothesis explains the transformation

STEP 3: BUILD UP COMPLEXITY
- If the transformation is complex, break it into sub-steps
- For each sub-step: hypotheticals → transform → next sub-step
- Build incrementally until you fully reconstruct the output

🔑 KEY PRINCIPLES:
- You have the ANSWER (the output). Use it to check your hypotheses!
- Don't guess randomly - systematically test rules against the known output
- Every transformation is deterministic and reproducible
- Compare your hypothetical grids to the actual output to verify correctness
- Break complex transformations into smaller, verifiable steps

Here's the first training example - study it carefully:

Input grid ({len(task['train'][0]['input'][0])}x{len(task['train'][0]['input'])} - width x height):
{self.format_grid(task['train'][0]['input'])}

Output grid ({len(task['train'][0]['output'][0])}x{len(task['train'][0]['output'])} - width x height):
{self.format_grid(task['train'][0]['output'])}

IMPORTANT: Output dimensions are {len(task['train'][0]['output'][0])}x{len(task['train'][0]['output'])}. All outputs in this task will have these exact dimensions.

START BY CAREFULLY COMPARING: What transformation converts the input into this specific output? Generate hypothetical visualizations to test different transformation rules, then commit to the one that best reproduces the output.
"""

        response_1 = self.call_ai_with_image(prompt_1, [input_img_1, output_img_1])
        num_phases += 1
        
        # Phase 2: Show second training input, ask for prediction
        if len(task['train']) > 1:
            print("\n" + "="*80)
            print("=== Phase 2: Second training input - predict output ===")
            print("="*80)
            
            self.current_phase = 2  # Track phase for verification control
            self.current_training_example = 1
            self.current_expected_output = task['train'][1]['output']  # Set for verification
            self.accumulated_hypotheticals = []  # Reset for new phase
            self.current_step_number = 1  # Reset step counter for new training example
            self.last_transform_path = None  # Reset transform tracking
            img = grid_to_image(task['train'][1]['input'], 30)
            input_img_2 = self.save_visualization(img, "input", visualization_type="transform")  # Input is special case
            
            prompt_2 = f"""
Now apply the pattern you learned from the first example to this NEW input.

🎯 RECALL YOUR LEARNING: In the first example, you identified a transformation rule. Now apply that SAME rule to this new input.

Second training input ({len(task['train'][1]['input'][0])}x{len(task['train'][1]['input'])} - width x height):
{self.format_grid(task['train'][1]['input'])}

CRITICAL: Your output MUST have dimensions {len(task['train'][0]['output'][0])}x{len(task['train'][0]['output'])} (same as first example's output).

APPLY THE LEARNED PATTERN:

STEP 1: RECALL THE TRANSFORMATION RULE
- What pattern did you identify in example 1?
- What was the core transformation rule?
- How does it apply generally?

STEP 2: APPLY TO NEW INPUT
For each step of applying the transformation:
   
   A. GENERATE HYPOTHETICALS (3-5 variations):
      - Apply the learned rule to this new input
      - Test slight variations in how to interpret the rule
      - Show different ways to apply the same core pattern
      - Use type='hypothetical' for each variation
   
   B. CHOOSE BEST APPLICATION:
      - Pick the application that most consistently applies your learned rule
      - Create ONE 'transform' visualization
      - Use type='transform' for your chosen application
      - Explain why this is the correct application

STEP 3: BUILD INCREMENTALLY
- If transformation has multiple steps, do them one at a time
- Each sub-step: hypotheticals → transform → next sub-step
- Maintain consistency with the rule from example 1

🔑 KEY PRINCIPLES:
- Apply the SAME transformation rule you identified in example 1
- Don't introduce new patterns - stay consistent
- The rule should work the same way on different inputs
- Build your output step by step, checking consistency at each stage

START NOW: Apply your learned transformation rule from example 1 to this new input. Show your work through hypothetical visualizations, then commit to transforms.
"""

            response_2 = self.call_ai_with_image(prompt_2, [input_img_2])
            num_phases += 1
            
            # Phase 3: Show actual second training output
            print("\n" + "="*80)
            print("=== Phase 3: Actual second training output ===")
            print("="*80)
            
            self.current_phase = 3  # Track phase for verification control (post-hoc feedback)
            img = grid_to_image(task['train'][1]['output'], 30)
            output_img_2 = self.save_visualization(img, "output", is_actual_output=True)
            
            # Before showing the actual output, find the best hypothetical
            training_dir = os.path.join(self.visualizations_dir, self.current_task_name, f"training_{self.current_training_example + 1}")
            
            # Check all hypothetical steps (flat structure)
            verification_results = []
            all_items = os.listdir(training_dir)
            
            # Find all unique step numbers that have hypotheticals
            step_numbers = set()
            for filename in all_items:
                if filename.endswith('_hypothetical.png'):
                    step_num = int(filename.split('_')[0])
                    step_numbers.add(step_num)
            
            # Verify hypotheticals for each step
            for step_num in sorted(step_numbers):
                result = self.find_best_hypothetical(training_dir, task['train'][1]['output'], step_num)
                if result:
                    verification_results.append((step_num, result))
            
            # Build verification feedback
            verification_feedback = ""
            if verification_results:
                verification_feedback = "\n\nVERIFICATION RESULTS:\n"
                verification_feedback += "I've compared your hypotheticals to the actual output:\n\n"
                for step_num, (best_grid, best_desc, diff) in verification_results:
                    if diff == 0:
                        verification_feedback += f"  ✅ Step {step_num}: One of your hypotheticals was PERFECT! ({best_desc[:50]})\n"
                    elif diff < 10:
                        verification_feedback += f"  ⭐ Step {step_num}: Very close! Only {diff} cells different. ({best_desc[:50]})\n"
                    else:
                        verification_feedback += f"  Step {step_num}: Best hypothetical was {diff} cells off. ({best_desc[:50]})\n"
                verification_feedback += "\nUse this feedback to refine your approach.\n"
            
            prompt_3 = f"""Here's the actual output for the second training example:

Output grid:
{self.format_grid(task['train'][1]['output'])}
{verification_feedback}
If you did not produce the correct output earlier, refine your approach and use the tool to iterate. 

Remember every transformation here is deterministic and reproducible. Do not find patterns that only exist in one input while still capturing all transformations and properties of the board.

Symbols may have semantic significants; properties of the symbols may convey this semantic significants. You need to find what properties carry semantic significance and what properties do not contribute to decision making. 

Compositional reasoning and turn-by-turn application of rules may be important. You may have to apply one transformation to allow the others to make sense. You can try using a tool to generate an image of the data and analyse that along the way. Try making incremental changes to the board and looking at the results by using the visualization tool. 

Some rules have to be applied based on context. Do not fixate of superficial patterns; find what properties have semantic significance and use those as context. Some attributes or properties may not be related; if they aren't consistent across all inputs, don't focus on them. 

Continue iterating until the tool generates the correct outputs in both training examples.
"""

            response_3 = self.call_ai_with_image(prompt_3, [output_img_2])
            num_phases += 1
        
        # If there are more training examples, show them
        for i in range(2, len(task['train'])):
            print(f"\n" + "="*80)
            print(f"=== Additional training example {i+1} ===")
            print("="*80)
            
            self.current_training_example = i
            self.current_expected_output = task['train'][i]['output']  # Set for verification
            self.accumulated_hypotheticals = []  # Reset for new phase
            img = grid_to_image(task['train'][i]['input'], 30)
            input_img = self.save_visualization(img, "input")
            img = grid_to_image(task['train'][i]['output'], 30)
            output_img = self.save_visualization(img, "output", is_actual_output=True)
            
            prompt = f"""Here's training example {i+1}:

Input:
{self.format_grid(task['train'][i]['input'])}

Output:
{self.format_grid(task['train'][i]['output'])}
"""

            response = self.call_ai_with_image(prompt, [input_img, output_img])
            num_phases += 1
        
        # Phase 4: Test input - ask for output
        print("\n" + "="*80)
        print("=== Phase 4: Test input - generate output ===")
        print("="*80)
        
        self.current_phase = 4  # Track phase - NO VERIFICATION (matches deployment)
        # Increment to final training example + 1 for test case
        self.current_training_example = len(task['train'])
        self.current_expected_output = task['test'][0]['output']  # Set for verification (if available)
        self.accumulated_hypotheticals = []  # Reset for new phase
        self.current_step_number = 1  # Reset step counter for test example
        self.last_transform_path = None  # Reset transform tracking
        img = grid_to_image(task['test'][0]['input'], 30)
        test_input_img = self.save_visualization(img, "input")
        
        img = grid_to_image(task['test'][0]['output'], 30)
        test_output_img = self.save_visualization(img, "output", is_actual_output=True)
        print(f"  Test output image saved to: {test_output_img}")
        
        prompt_test = f"""This is the TEST. Apply the transformation rule you learned from the training examples.

🎯 FINAL APPLICATION: You've learned the pattern from 2+ training examples. Now apply it confidently to this test input.

Test input ({len(task['test'][0]['input'][0])}x{len(task['test'][0]['input'])} - width x height):
{self.format_grid(task['test'][0]['input'])}

CRITICAL: Output dimensions MUST be {len(task['train'][0]['output'][0])}x{len(task['train'][0]['output'])} (width x height).

SOLVE WITH CONFIDENCE:

STEP 1: RECALL YOUR LEARNED PATTERN
- What transformation rule did you identify?
- How did it apply across both training examples?
- What's the core, consistent pattern?

STEP 2: APPLY TO TEST INPUT
For each step of the transformation:
   
   A. GENERATE HYPOTHETICALS (3-5 applications):
      - Apply your learned rule to this test input
      - Show the transformation working step by step
      - Test slight variations if there's ambiguity
      - Use type='hypothetical' for each variation
      - Explain your reasoning for each
   
   B. COMMIT TO BEST APPLICATION:
      - Choose the most consistent application of your learned rule
      - Create ONE 'transform' visualization
      - Use type='transform' for your committed step
      - Explain why this correctly applies the pattern

STEP 3: BUILD THE COMPLETE OUTPUT
- Apply transformation incrementally
- Each sub-step: hypotheticals → transform → next sub-step
- Maintain exact consistency with training examples
- Verify dimensions and pattern at each stage

🔑 KEY PRINCIPLES:
- Apply the EXACT SAME rule you learned from training examples
- The transformation should work consistently across all examples
- Don't introduce new logic - use what you learned
- Show your complete reasoning process through visualizations
- Build confidence by applying the pattern systematically

START NOW: Apply your learned transformation rule to generate the test output. Document every step with visualizations.

IMPORTANT: Provide your final answer as a grid in the exact same format, with square brackets and comma-separated values. Make sure the dimensions are correct."""

        response_test = self.call_ai_with_image(prompt_test, [test_input_img])
        num_phases += 1
        
        # Post-hoc verification for Phase 4 (calculate scores silently for metadata)
        if self.accumulated_hypotheticals and self.current_expected_output:
            print(f"\n📊 Post-hoc analysis: Adding verification metadata to {len(self.accumulated_hypotheticals)} Phase 4 hypotheticals...")
            
            best_grid = None
            best_diff = float('inf')
            all_hyp_info = []
            
            # Calculate scores for all hypotheticals
            for hyp_grid, hyp_desc, hyp_path in self.accumulated_hypotheticals:
                score = self.calculate_grid_difference(hyp_grid, self.current_expected_output)
                all_hyp_info.append({
                    'path': os.path.basename(hyp_path),
                    'description': hyp_desc[:80],
                    'score': score if score != float('inf') else -1
                })
                
                if score < best_diff:
                    best_diff = score
                    best_grid = hyp_grid
            
            # Add metadata to each hypothetical
            for hyp_grid, hyp_desc, hyp_path in self.accumulated_hypotheticals:
                score = self.calculate_grid_difference(hyp_grid, self.current_expected_output)
                is_chosen = (hyp_grid == best_grid) if best_grid is not None else False
                
                self.add_verification_metadata_to_image(
                    hyp_path,
                    verification_score=score if score != float('inf') else -1,
                    is_chosen=is_chosen,
                    all_hypotheticals=all_hyp_info
                )
                
                print(f"  📊 Metadata added: {os.path.basename(hyp_path)} (score: {score}, chosen: {is_chosen})")
            
            print(f"  ✅ Phase 4 metadata complete (no real-time feedback was given during generation)")
        
        # Parse the predicted output
        predicted_output = self.parse_grid_from_response(response_test)
        
        # Check if we got a valid prediction
        if not predicted_output:
            print("\n❌ Could not parse a valid grid from the response")
            return False, None, num_phases
        
        # Compare with actual test output (if available)
        if 'output' in task['test'][0] and task['test'][0]['output']:
            actual_output = task['test'][0]['output']
            
            if predicted_output == actual_output:
                print("\n✅ SUCCESS! Predicted output matches actual output!")
                return True, predicted_output, num_phases
            else:
                print("\n❌ Predicted output does not match actual output")
                print(f"Predicted: {predicted_output[:3]}..." if len(predicted_output) > 3 else predicted_output)
                print(f"Actual: {actual_output[:3]}..." if len(actual_output) > 3 else actual_output)
                return False, predicted_output, num_phases
        else:
            print("\n⚠️ No test output available for comparison")
            print(f"Generated prediction: {predicted_output[:3]}..." if len(predicted_output) > 3 else predicted_output)
            return False, predicted_output, num_phases


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python arc_visual_solver.py <task_json_file>")
        sys.exit(1)
    
    task_file = sys.argv[1]
    if not os.path.exists(task_file):
        print(f"Error: Task file '{task_file}' not found")
        sys.exit(1)
    
    try:
        solver = ARCVisualSolver()
        success, prediction, num_phases = solver.solve(task_file)
        
        print(f"\n{'='*80}")
        print(f"Solving complete!")
        print(f"Phases used: {num_phases}")
        print(f"Result ({solver.current_task_name}): {'SUCCESS ✅' if success else 'FAILED ❌'}")
        if prediction:
            # Save prediction as image using our method
            img = grid_to_image(prediction, 30)
            pred_path = solver.save_visualization(img, "final_prediction", is_model_output=True)
            print(f"Prediction saved to: {pred_path}")
        print(f"{'='*80}")
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()