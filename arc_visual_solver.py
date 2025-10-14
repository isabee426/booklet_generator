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
        self.dimension_rejection_count = 0  # Track how many times all hypotheticals rejected for dimensions
        self.perfect_rule_found = False  # Track if we found a perfect rule to validate on remaining examples
        self.perfect_rule_description = None  # Store the perfect rule description
        self.perfect_rule_validation_failures = 0  # Track failed validation attempts (max 2 retries)
        self.validation_attempted_examples = []  # Track which training examples were attempted in validation mode
        
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
            # Verify when we have 3+ hypotheticals (encourages proper exploration)
            # Exception: Skip verification in Phase 4 (test phase - no real-time feedback)
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
                    
                    # Verify when: transform created OR 3+ hypotheticals accumulated
                    if transform_created or len(self.accumulated_hypotheticals) >= 3:
                        should_verify = True
            
            if should_verify:
                print(f"\n🔍 Running verification on {len(self.accumulated_hypotheticals)} hypotheticals...")
                
                # Reset dimension rejection counter since we have valid hypotheticals
                if len(self.accumulated_hypotheticals) > 0:
                    self.dimension_rejection_count = 0
                
                # Check if we have a valid expected output
                if not hasattr(self, 'current_expected_output') or not self.current_expected_output:
                    print("  ⚠️ Warning: No expected output set for verification. Skipping verification.")
                    should_verify = False
                
                # Check if we have any valid hypotheticals to verify
                if len(self.accumulated_hypotheticals) == 0:
                    self.dimension_rejection_count += 1
                    print(f"  ⚠️ No valid hypotheticals to verify (all were rejected for dimension mismatch)")
                    print(f"  🔄 Rejection #{self.dimension_rejection_count} - Asking model to retry...")
                    
                    # Give up after 2 dimension rejection cycles
                    if self.dimension_rejection_count >= 2:
                        print("  ❌ Too many dimension rejections (2+). Moving on to avoid infinite loop.")
                        print("  ⏭️ Proceeding to next phase...")
                        self.dimension_rejection_count = 0  # Reset for next phase
                        should_verify = False
                    else:
                        # Give feedback about rejection without revealing the solution
                        retry_msg = "\n⚠️ ALL HYPOTHETICALS REJECTED: Dimension mismatch\n\n"
                        retry_msg += "All of your hypothetical grids were rejected because they had incorrect output dimensions.\n\n"
                        retry_msg += "🔍 DIMENSION INFERENCE REMINDER:\n"
                        retry_msg += "- Look at the training examples you've seen\n"
                        retry_msg += "- What was the relationship between input dimensions and output dimensions?\n"
                        retry_msg += "- Does the output size stay constant?\n"
                        retry_msg += "- Does it scale with the input?\n"
                        retry_msg += "- Does it depend on the content/pattern?\n\n"
                        retry_msg += "⚠️ CRITICAL RULE: Pick ONE dimension size for ALL hypotheticals in your next batch.\n"
                        retry_msg += "Don't test multiple dimensions simultaneously - commit to one size, test content variations.\n\n"
                        retry_msg += f"Try again with different output dimensions (attempt {self.dimension_rejection_count}/2)."
                        
                        # Inject retry message back into conversation
                        call_params["input"].append({
                            "role": "user",
                            "content": retry_msg
                        })
                        
                        print(retry_msg)
                        should_verify = False  # Don't verify yet, let model retry
                
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
                    verification_msg += f"\n\n✅ ACTION REQUIRED: Call the visualization tool NOW with:"
                    verification_msg += f"\n   - visualization_type='transform'"
                    verification_msg += f"\n   - Use the EXACT SAME grid as your perfect hypothetical"
                    verification_msg += f"\n   - Description: explain why this is the solution"
                    verification_msg += f"\n\nDo NOT ask which to choose - the perfect one is obvious. Generate the transform visualization immediately."
                elif best_diff <= 5:
                    verification_msg += f"\n💡 EXCELLENT! Best match is very close ({best_diff} cells off): {best_desc[:60]}"
                    verification_msg += f"\n\n✅ ACTION: Call visualization tool NOW with visualization_type='transform'"
                    verification_msg += f"\n   Use your best hypothesis as the transform. Don't ask - just generate it."
                elif best_diff <= 15:
                    verification_msg += f"\n⚠️ MODERATE. Best was {best_diff} cells off: {best_desc[:60]}"
                    verification_msg += f"\n\n✅ ACTION: Call visualization tool with visualization_type='transform'"
                    verification_msg += f"\n   Use your best hypothesis. Generate the transform now."
                else:
                    verification_msg += f"\n❌ All hypotheticals far off (best: {best_diff} cells different): {best_desc[:60]}"
                    
                    # If really bad (>30 cells off) and we have few hypotheticals, suggest trying more
                    if best_diff > 30 and len(self.accumulated_hypotheticals) < 5:
                        verification_msg += f"\n\n🔄 SUGGESTION: Your hypotheticals are all quite far off."
                        verification_msg += f"\n   Consider generating 2-3 MORE hypotheticals with completely different approaches."
                        verification_msg += f"\n   OR if you're confident, generate a 'transform' with your best guess and move forward."
                    else:
                        verification_msg += f"\n\n✅ ACTION: Call visualization tool with visualization_type='transform'"
                        verification_msg += f"\n   Use your best hypothesis anyway. Generate transform to proceed."
                
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
                    
                    # Mark that we found a perfect rule (for validation on remaining examples)
                    if self.current_training_example <= 1:  # Found on training 1 or 2
                        self.perfect_rule_found = True
                        self.perfect_rule_description = best_desc
                        print("  ✨ Perfect rule identified early - will validate on remaining training examples")
                    
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
        self.correct_rules = []  # Track successful rules from training examples
        
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

STEP 2: FORM HYPOTHESES BY GENERATING VISUALIZATIONS
⚠️ YOU MUST USE THE VISUALIZATION TOOL - Don't just describe, actually create the images!

For each hypothesis you explore:
   A. CREATE 3-5 'hypothetical' visualizations testing different pattern interpretations:
      - ⚠️ CRITICAL: Call the visualization tool with type='hypothetical' for EACH hypothesis
      - ⚠️ CRITICAL: ALL hypotheticals in this batch must use the SAME output dimensions
      - Pick ONE dimension size based on the pattern, then test variations of the content
      - Hypothesis 1: Test if pattern is about [specific transformation rule] → CALL TOOL
      - Hypothesis 2: Test if pattern is about [different rule] → CALL TOOL
      - Hypothesis 3: Test if pattern is about [another rule] → CALL TOOL
      - Generate at least 3 hypothetical grids, each testing a distinct interpretation
   
   B. WAIT FOR VERIFICATION:
      - The system will automatically compare your hypotheticals to the actual output
      - You'll receive scores showing which hypothesis is closest
      - Use this feedback to understand which rule works best
   
   C. COMMIT TO BEST HYPOTHESIS:
      - Based on verification scores, choose the best hypothesis
      - Create ONE 'transform' visualization (type='transform') showing your final understanding
      - CALL THE TOOL - don't just describe the transform in text
      - Explain WHY this hypothesis explains the transformation

STEP 3: BUILD UP COMPLEXITY
- If the transformation is complex, break it into sub-steps
- For each sub-step: hypotheticals → transform → next sub-step
- Build incrementally until you fully reconstruct the output

🔑 KEY PRINCIPLES:
- You have the ANSWER (the output). Use it to check your hypotheses!
- YOU MUST GENERATE ACTUAL VISUALIZATIONS - use the tool, don't just write text descriptions
- Generate multiple hypothetical grids to test different transformation rules
- The system will verify your hypotheticals and tell you which is closest
- Every transformation is deterministic and reproducible
- Break complex transformations into smaller, verifiable steps

⚠️ IMPORTANT: Your response MUST include visualization tool calls. Text-only explanations without tool calls will not be accepted.

Here's the first training example:

📸 IMAGE 1: Input grid ({len(task['train'][0]['input'][0])}x{len(task['train'][0]['input'])})
📸 IMAGE 2: Output grid ({len(task['train'][0]['output'][0])}x{len(task['train'][0]['output'])})

Input grid:
{self.format_grid(task['train'][0]['input'])}

Output grid (TARGET to match):
{self.format_grid(task['train'][0]['output'])}

🎯 YOUR TASK: FIND THE TRANSFORMATION RULE
You can see both input and output. Your job is to figure out WHAT RULE transforms the input into this output.

⚠️ CRITICAL: Do NOT just copy the output values!
Instead, TEST DIFFERENT TRANSFORMATION RULES until you find which one produces this output.

PROCESS:
1. Generate 3-5 hypotheticals, each testing a DIFFERENT transformation rule
   - Hypothesis 1: "What if the rule is [specific approach]?" → Apply that rule → Generate grid
   - Hypothesis 2: "What if the rule is [different approach]?" → Apply that rule → Generate grid
   - Hypothesis 3: "What if the rule is [another approach]?" → Apply that rule → Generate grid
   
2. The verification system will tell you which hypothesis MATCHES the target output
   
3. The one that matches reveals the TRUE RULE - that's what you'll use on new inputs later

🔑 EXAMPLES OF GOOD vs BAD HYPOTHESES:
❌ BAD: "Hypothesis: Copy input then modify a few cells" (lazy, not a rule)
❌ BAD: "Hypothesis: Place the exact values I see in the output" (just memorizing)
✅ GOOD: "Hypothesis: For each object in input, place a colored bar in snake order" (testing a rule)
✅ GOOD: "Hypothesis: Connect all red cells using Manhattan distance nearest-neighbor" (testing a rule)
✅ GOOD: "Hypothesis: Mirror the input horizontally then apply color transformation" (testing a rule)

⚠️ CRITICAL RULE DISCOVERY PROCESS:
Your hypotheses should describe TRANSFORMATION RULES, not copying strategies!
- Bad approach: Start with input grid, make small edits
- Good approach: Identify pattern (e.g., "connect objects", "fill regions", "apply symmetry"), then BUILD the output from scratch using that rule
- The rule should be something you can EXPLAIN and APPLY to a completely different input later

OUTPUT DIMENSIONS: {len(task['train'][0]['output'][0])}x{len(task['train'][0]['output'])}
Note: Infer output dimensions from the transformation rule, not by measuring the given output.

START NOW - Generate 3-5 hypotheticals, each testing a different transformation rule!
⚠️ Build output grids from scratch using transformation rules, don't start by copying the input!
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

CRITICAL DIMENSION GUIDANCE:
- First example: Input {len(task['train'][0]['input'][0])}x{len(task['train'][0]['input'])} -> Output {len(task['train'][0]['output'][0])}x{len(task['train'][0]['output'])}
- This input: {len(task['train'][1]['input'][0])}x{len(task['train'][1]['input'])}
- Infer the correct output dimensions from the pattern.

APPLY THE LEARNED PATTERN:

STEP 1: RECALL THE TRANSFORMATION RULE
- What pattern did you identify in example 1?
- What was the core transformation rule?
- How does it apply generally?

STEP 2: APPLY TO NEW INPUT
For each step of applying the transformation:
   
   A. GENERATE HYPOTHETICALS (3-5 variations):
      - ⚠️ CRITICAL: ALL hypotheticals in this batch must use the SAME output dimensions
      - ⚠️ CRITICAL: Each hypothetical must be MEANINGFULLY DIFFERENT, not trivial variations
      - Infer the output size first, then test content variations
      - Apply the learned rule to this new input
      - Test significantly different interpretations (e.g., different traversal orders, different coloring rules, different connection patterns)
      - Show genuinely distinct approaches to applying the core pattern
      - Avoid minor tweaks like "same but shift 1 pixel" - aim for conceptually different applications
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
            
            # Check if training example 2 was solved correctly - if so, add to correct rules
            if verification_results:
                best_overall_diff = min([diff for _, (_, _, diff) in verification_results])
                if best_overall_diff == 0:
                    print("  ✅ Training example 2 solved perfectly! Adding rule to correct_rules stack.")
                    # Extract the rule/pattern from conversation (simplified - gets recent reasoning)
                    self.correct_rules.append({
                        'example': 2,
                        'rule': 'Successfully applied transformation rule to training example 2',
                        'status': 'correct'
                    })
        
        # If there are more training examples (3+), use NO real-time verification strategy
        for i in range(2, len(task['train'])):
            print(f"\n" + "="*80)
            
            # Check if we found a perfect rule earlier - if so, use validation mode
            if self.perfect_rule_found:
                print(f"=== RULE VALIDATION: Testing perfect rule on training example {i+1} ===")
                # Track that this example is being attempted in validation mode
                if i not in self.validation_attempted_examples:
                    self.validation_attempted_examples.append(i)
            else:
                print(f"=== Additional training example {i+1} - apply learned rules (NO real-time verification) ===")
            print("="*80)
            
            self.current_phase = 4  # Set to 4 to DISABLE real-time verification
            self.current_training_example = i
            self.current_expected_output = task['train'][i]['output']  # Set for post-hoc verification
            self.accumulated_hypotheticals = []  # Reset for new phase
            self.current_step_number = 1  # Reset step counter
            self.last_transform_path = None  # Reset transform tracking
            
            img = grid_to_image(task['train'][i]['input'], 30)
            input_img = self.save_visualization(img, "input", visualization_type="transform")
            
            # Build prompt based on whether we have a perfect rule to validate
            if self.perfect_rule_found:
                # Validation mode: Just apply the perfect rule (with retry context if applicable)
                retry_context = ""
                if self.perfect_rule_validation_failures > 0:
                    retry_context = f"""
⚠️ RETRY ATTEMPT {self.perfect_rule_validation_failures}/2:
Your rule failed validation on the previous example. Refine your understanding:
- What edge case or variation did you miss?
- Does the rule need adjustment?
- Are there conditions or exceptions?
"""
                
                prompt_additional = f"""🎯 RULE VALIDATION MODE{retry_context}

You found a PERFECT rule that worked on the previous training examples:
  "{self.perfect_rule_description}"

Now TEST and ADAPT this rule for training example {i+1} with different dimensions/layout.

Training example {i+1} input ({len(task['train'][i]['input'][0])}x{len(task['train'][i]['input'])} - width x height):
{self.format_grid(task['train'][i]['input'])}

CRITICAL DIMENSION GUIDANCE:
Previous examples showed this input→output pattern:
{chr(10).join([f"  Example {j+1}: Input {len(task['train'][j]['input'][0])}x{len(task['train'][j]['input'])} → Output {len(task['train'][j]['output'][0])}x{len(task['train'][j]['output'])}" for j in range(i)])}

This example: Input {len(task['train'][i]['input'][0])}x{len(task['train'][i]['input'])}

Apply the same dimension pattern → Infer the output dimensions
⚠️ ALL your hypotheticals MUST use the SAME inferred output dimensions!

VALIDATION TASK - ADAPT THE RULE:
🔑 KEY INSIGHT: Your perfect rule describes the CORE TRANSFORMATION, but you must ADAPT it to this new input size/layout.

Your rule is the "what" (e.g., "connect centers in snake order"), but you must figure out the "how" for THIS specific input:
- How many centers are there in THIS input?
- What's the snake order for THIS layout?
- What corridors exist in THIS grid?
- How does the output size change for THIS input?

APPROACH:
1. Analyze this new input structure (different from training 1-2)
2. Identify how your rule's PRINCIPLE applies here (same transformation type, different specifics)
3. Generate 3-5 hypotheticals testing different ADAPTATIONS of your rule{' (with refinements based on previous feedback)' if self.perfect_rule_validation_failures > 0 else ''}:
   - ⚠️ CRITICAL: Each hypothetical must be MEANINGFULLY DIFFERENT adaptation
   - ⚠️ CRITICAL: ALL hypotheticals must use the SAME output dimensions
   - Test: How does the rule scale to different grid sizes?
   - Test: How does the rule handle different numbers of objects?
   - Test: How does the rule adapt to different spatial layouts?
4. Pick the best adaptation and create a 'transform'

⚠️ Don't just blindly copy the exact same output pattern - ADAPT the rule to this input's structure!

Apply your rule's principle with appropriate adaptations now.
"""
            else:
                # Normal mode: Apply learned rules
                # Build summary of correct rules from previous examples
                rules_summary = ""
                if self.correct_rules:
                    rules_summary = "\n📚 CORRECT RULES FROM PREVIOUS EXAMPLES:\n"
                    for rule_info in self.correct_rules:
                        rules_summary += f"  ✅ Example {rule_info['example']}: {rule_info['rule']}\n"
                    rules_summary += "\nUse these validated rules to solve this example.\n"
                
                prompt_additional = f"""Now apply your learned rules to training example {i+1}.

⚠️ IMPORTANT: NO real-time verification available. Apply your rules confidently based on what you've learned.
{rules_summary}

Training example {i+1} input ({len(task['train'][i]['input'][0])}x{len(task['train'][i]['input'])} - width x height):
{self.format_grid(task['train'][i]['input'])}

CRITICAL DIMENSION GUIDANCE:
Previous examples showed this input→output pattern:
{chr(10).join([f"  Example {j+1}: Input {len(task['train'][j]['input'][0])}x{len(task['train'][j]['input'])} → Output {len(task['train'][j]['output'][0])}x{len(task['train'][j]['output'])}" for j in range(i)])}

This example: Input {len(task['train'][i]['input'][0])}x{len(task['train'][i]['input'])}
Apply the same dimension pattern: Output should be ___x___ (you must infer from pattern above)

⚠️ ALL your hypotheticals MUST use these same output dimensions!

APPLY YOUR LEARNED RULES:

STEP 1: RECALL THE VALIDATED RULES
- What transformation rules have you confirmed work?
- How have they applied consistently across examples 1 and 2?

STEP 2: APPLY CONFIDENTLY (NO VERIFICATION AVAILABLE)
   - ⚠️ CRITICAL: ALL hypotheticals must use the SAME output dimensions
   - ⚠️ CRITICAL: Make each hypothetical MEANINGFULLY DIFFERENT (not just minor tweaks)
   - Infer the output size from the pattern, commit to it for all hypotheticals
   - Generate hypotheticals exploring genuinely different ways your rules could apply
   - Test distinct interpretations: different orderings, different coloring schemes, different connection strategies
   - Avoid trivial variations - each hypothesis should represent a conceptually different approach
   - Use type='hypothetical' for variations
   - Pick the best application and create a 'transform'
   - No feedback will be given until you complete your prediction

STEP 3: BUILD YOUR FINAL OUTPUT
- Apply transformation step by step
- Trust your learned rules
- Show your complete reasoning process

START NOW: Apply your validated transformation rules to this example (no real-time verification).
"""

            response_additional = self.call_ai_with_image(prompt_additional, [input_img])
            num_phases += 1
            
            # Post-hoc verification (silent during generation, now revealed)
            print(f"\n" + "="*80)
            print(f"=== Reveal & Reflect: Training example {i+1} ===")
            print("="*80)
            
            self.current_phase = 3  # Post-hoc feedback phase
            img = grid_to_image(task['train'][i]['output'], 30)
            output_img = self.save_visualization(img, "output", is_actual_output=True)
            
            # Calculate post-hoc scores for all hypotheticals
            training_dir = os.path.join(self.visualizations_dir, self.current_task_name, f"training_{self.current_training_example + 1}")
            
            verification_results = []
            all_items = os.listdir(training_dir)
            best_overall_diff = float('inf')
            
            # Find all unique step numbers that have hypotheticals
            step_numbers = set()
            for filename in all_items:
                if filename.endswith('_hypothetical.png'):
                    step_num = int(filename.split('_')[0])
                    step_numbers.add(step_num)
            
            # Verify hypotheticals for each step (post-hoc)
            for step_num in sorted(step_numbers):
                result = self.find_best_hypothetical(training_dir, task['train'][i]['output'], step_num)
                if result:
                    verification_results.append((step_num, result))
                    best_grid, best_desc, diff = result
                    if diff < best_overall_diff:
                        best_overall_diff = diff
            
            # Build verification feedback
            verification_feedback = ""
            if verification_results:
                verification_feedback = "\n\n📊 POST-HOC VERIFICATION RESULTS:\n"
                
                # Different feedback for validation mode vs normal mode
                if self.perfect_rule_found:
                    verification_feedback += f"Testing your perfect rule on example {i+1}:\n\n"
                else:
                    verification_feedback += f"I've compared your hypotheticals for example {i+1} to the actual output:\n\n"
                
                for step_num, (best_grid, best_desc, diff) in verification_results:
                    if diff == 0:
                        verification_feedback += f"  ✅ Step {step_num}: PERFECT match! ({best_desc[:50]})\n"
                    elif diff < 10:
                        verification_feedback += f"  ⭐ Step {step_num}: Very close! Only {diff} cells different. ({best_desc[:50]})\n"
                    else:
                        verification_feedback += f"  ❌ Step {step_num}: {diff} cells off. ({best_desc[:50]})\n"
                
                if best_overall_diff == 0:
                    if self.perfect_rule_found:
                        verification_feedback += f"\n🎉 VALIDATION SUCCESS! Your perfect rule works on example {i+1}!"
                        verification_feedback += f"\n✨ This confirms your rule generalizes across all training examples."
                        verification_feedback += f"\n🚀 High confidence for the test case!"
                        # Reset failure counter on success
                        self.perfect_rule_validation_failures = 0
                    else:
                        verification_feedback += f"\n🎯 SUCCESS! You got example {i+1} correct using your learned rules!"
                        verification_feedback += f"\n✅ This confirms your transformation rule is correct."
                else:
                    if self.perfect_rule_found:
                        self.perfect_rule_validation_failures += 1
                        max_retries = 2
                        
                        if self.perfect_rule_validation_failures <= max_retries:
                            verification_feedback += f"\n⚠️ VALIDATION ATTEMPT {self.perfect_rule_validation_failures} FAILED! Your rule was off by {best_overall_diff} cells on example {i+1}."
                            verification_feedback += f"\n� RETRY: You have {max_retries - self.perfect_rule_validation_failures + 1} more attempt(s) to refine your rule."
                            verification_feedback += f"\n🔍 Reflect: What's different about this example?"
                            verification_feedback += f"\n� Consider: Maybe the rule needs adjustment or has edge cases."
                            # Keep the perfect_rule_found flag for retry
                        else:
                            verification_feedback += f"\n❌ VALIDATION FAILED AFTER {max_retries} RETRIES! Your rule was off by {best_overall_diff} cells on example {i+1}."
                            verification_feedback += f"\n🔍 Your 'perfect' rule doesn't generalize - it's overfitted to early examples."
                            verification_feedback += f"\n💭 Saving this rule as a failed hypothesis, will switch to exploratory mode."
                            
                            # Save the failed perfect rule to the stack for future reference
                            print(f"  📝 Saving failed perfect rule to correct_rules stack for reference")
                            self.correct_rules.append({
                                'example': f"1-{i}",  # Range where it was tested
                                'rule': f"FAILED VALIDATION: {self.perfect_rule_description}",
                                'status': 'failed_validation',
                                'validation_failures': self.perfect_rule_validation_failures,
                                'diff': best_overall_diff
                            })
                            
                            # Clear the perfect rule flag after max retries
                            failed_rule_desc = self.perfect_rule_description  # Save for potential re-processing
                            self.perfect_rule_found = False
                            self.perfect_rule_description = None
                            self.perfect_rule_validation_failures = 0
                    else:
                        verification_feedback += f"\n⚠️ Your prediction was off by {best_overall_diff} cells."
                        verification_feedback += f"\n🔍 Reflect on what's different about this example."
            
            # Determine if this example was solved correctly
            success = best_overall_diff == 0
            
            # Build prompt based on validation mode vs normal mode
            if self.perfect_rule_found and success:
                status_msg = "🎉 VALIDATION SUCCESSFUL! Your perfect rule generalizes to this example."
                continue_msg = "Your rule is validated across multiple examples. Continue to test on remaining examples."
            elif self.perfect_rule_found and not success:
                if self.perfect_rule_validation_failures <= 2:
                    status_msg = f"⚠️ VALIDATION ATTEMPT {self.perfect_rule_validation_failures} FAILED - But you can retry!"
                    continue_msg = f"""Your rule needs refinement. Attempt {self.perfect_rule_validation_failures}/2:
- What's different about this example?
- What edge cases did you miss?
- How should you adjust the rule?

🔄 TRY AGAIN with refined understanding. Same validation mode."""
                else:
                    status_msg = "❌ VALIDATION FAILED AFTER RETRIES! Switching to exploratory mode."
                    continue_msg = """Your early rule was overfitted. Major rethink needed:
- What fundamentally did you miss?
- Is there a completely different pattern?
- Should you start fresh with a new hypothesis?

Continue exploring freely."""
            elif success:
                status_msg = "🎉 CORRECT! Your rules work on this example."
                continue_msg = "Continue building confidence - your rules are working consistently."
            else:
                status_msg = "❌ INCORRECT. Reflect on what you missed:"
                continue_msg = """What did you miss or misunderstand?
- Is there an edge case you didn't account for?
- Did you misapply the rule?
- Is there a subtle variation you overlooked?

Refine your understanding and describe what you learned from this mistake."""
            
            prompt_reveal = f"""Here's the actual output for training example {i+1}:

Output grid:
{self.format_grid(task['train'][i]['output'])}
{verification_feedback}

{status_msg}

{continue_msg}
"""

            response_reveal = self.call_ai_with_image(prompt_reveal, [output_img])
            num_phases += 1
            
            # Add to correct rules if successful, or note the failure
            if success:
                print(f"  ✅ Training example {i+1} solved perfectly! Adding to correct_rules stack.")
                self.correct_rules.append({
                    'example': i+1,
                    'rule': f'Successfully applied transformation rule to training example {i+1}',
                    'status': 'correct'
                })
            else:
                print(f"  ⚠️ Training example {i+1} not solved. Reflection captured for next attempt.")
                self.correct_rules.append({
                    'example': i+1,
                    'rule': f'Failed on training example {i+1} - needs refinement',
                    'status': 'failed',
                    'diff': best_overall_diff
                })
            
            # Check if we just switched out of validation mode (perfect rule failed completely)
            # If so, we need to re-process the validation-attempted examples in exploratory mode
            if not self.perfect_rule_found and len(self.validation_attempted_examples) > 0:
                print(f"\n🔄 SWITCHING TO EXPLORATORY MODE - Re-processing examples that were attempted in validation mode")
                examples_to_reprocess = list(self.validation_attempted_examples)
                self.validation_attempted_examples = []  # Clear the list
                
                for reprocess_i in examples_to_reprocess:
                    print(f"\n" + "="*80)
                    print(f"=== EXPLORATORY MODE: Training example {reprocess_i+1} (re-processing after failed validation) ===")
                    print("="*80)
                    
                    # Reset state for this example
                    self.current_phase = 4  # NO real-time verification
                    self.current_training_example = reprocess_i
                    self.current_expected_output = task['train'][reprocess_i]['output']
                    self.accumulated_hypotheticals = []
                    self.current_step_number = 1
                    self.last_transform_path = None
                    
                    # Re-create input image
                    img = grid_to_image(task['train'][reprocess_i]['input'], 30)
                    input_img = self.save_visualization(img, "input", visualization_type="transform")
                    
                    # Build exploratory prompt with context about failed rule
                    rules_summary = ""
                    if self.correct_rules:
                        rules_summary = "\n📚 LEARNED FROM PREVIOUS ATTEMPTS:\n"
                        for rule_info in self.correct_rules:
                            if rule_info['status'] == 'failed_validation':
                                rules_summary += f"  ❌ FAILED RULE: {rule_info['rule']}\n"
                            elif rule_info['status'] == 'correct':
                                rules_summary += f"  ✅ Example {rule_info['example']}: {rule_info['rule']}\n"
                            elif rule_info['status'] == 'failed':
                                rules_summary += f"  ⚠️ Example {rule_info['example']}: {rule_info['rule']}\n"
                        rules_summary += "\n"
                    
                    prompt_reprocess = f"""Now explore this training example with a fresh perspective.

⚠️ CONTEXT: Your previous "perfect" rule failed validation. Try completely different approaches.
{rules_summary}

Training example {reprocess_i+1} input ({len(task['train'][reprocess_i]['input'][0])}x{len(task['train'][reprocess_i]['input'])} - width x height):
{self.format_grid(task['train'][reprocess_i]['input'])}

CRITICAL DIMENSION GUIDANCE:
- Previous training outputs were: {', '.join([f"{len(task['train'][j]['output'][0])}x{len(task['train'][j]['output'])}" for j in range(reprocess_i)])}
- Previous training inputs were: {', '.join([f"{len(task['train'][j]['input'][0])}x{len(task['train'][j]['input'])}" for j in range(reprocess_i)])}
- This input is: {len(task['train'][reprocess_i]['input'][0])}x{len(task['train'][reprocess_i]['input'])}
- Infer the output dimensions from the pattern.

EXPLORATORY APPROACH:

STEP 1: FORGET THE FAILED RULE
- What other patterns could explain the transformations?
- What did the failed rule miss or misinterpret?
- Look for completely different transformation principles

STEP 2: GENERATE DIVERSE HYPOTHETICALS (3-5 variations)
   - ⚠️ CRITICAL: ALL hypotheticals must use the SAME output dimensions
   - ⚠️ CRITICAL: Make each hypothetical MEANINGFULLY DIFFERENT (not just minor tweaks)
   - Try fundamentally different transformation rules
   - Test alternative interpretations of the pattern
   - Use type='hypothetical' for each variation

STEP 3: COMMIT TO BEST APPROACH
   - Pick the transformation that seems most consistent
   - Create ONE 'transform' visualization
   - Use type='transform' for your chosen approach
   - Explain your reasoning

START NOW: Explore this example with fresh eyes and diverse hypotheses.
"""

                    response_reprocess = self.call_ai_with_image(prompt_reprocess, [input_img])
                    num_phases += 1
                    
                    # Post-hoc reveal for re-processed example
                    print(f"\n" + "="*80)
                    print(f"=== Reveal & Reflect: Training example {reprocess_i+1} (re-processed) ===")
                    print("="*80)
                    
                    self.current_phase = 3
                    img = grid_to_image(task['train'][reprocess_i]['output'], 30)
                    output_img = self.save_visualization(img, "output", is_actual_output=True)
                    
                    # Calculate post-hoc scores
                    training_dir = os.path.join(self.visualizations_dir, self.current_task_name, f"training_{self.current_training_example + 1}")
                    
                    verification_results = []
                    all_items = os.listdir(training_dir)
                    best_overall_diff = float('inf')
                    
                    step_numbers = set()
                    for filename in all_items:
                        if filename.endswith('_hypothetical.png'):
                            step_num = int(filename.split('_')[0])
                            step_numbers.add(step_num)
                    
                    for step_num in sorted(step_numbers):
                        result = self.find_best_hypothetical(training_dir, task['train'][reprocess_i]['output'], step_num)
                        if result:
                            verification_results.append((step_num, result))
                            best_grid, best_desc, diff = result
                            if diff < best_overall_diff:
                                best_overall_diff = diff
                    
                    # Build verification feedback for re-processed example
                    verification_feedback = ""
                    if verification_results:
                        verification_feedback = "\n\n📊 POST-HOC VERIFICATION RESULTS (Exploratory Mode):\n"
                        verification_feedback += f"I've compared your new hypotheticals for example {reprocess_i+1}:\n\n"
                        
                        for step_num, (best_grid, best_desc, diff) in verification_results:
                            if diff == 0:
                                verification_feedback += f"  ✅ Step {step_num}: PERFECT match! ({best_desc[:50]})\n"
                            elif diff < 10:
                                verification_feedback += f"  ⭐ Step {step_num}: Very close! Only {diff} cells different. ({best_desc[:50]})\n"
                            else:
                                verification_feedback += f"  ❌ Step {step_num}: {diff} cells off. ({best_desc[:50]})\n"
                        
                        if best_overall_diff == 0:
                            verification_feedback += f"\n🎯 SUCCESS! Your exploratory approach found the correct solution!"
                        else:
                            verification_feedback += f"\n⚠️ Still {best_overall_diff} cells off. Continue refining your understanding."
                    
                    reprocess_success = best_overall_diff == 0
                    
                    if reprocess_success:
                        status_msg = "🎉 CORRECT! Your exploratory approach worked."
                        continue_msg = "This new understanding is more accurate than the failed rule."
                    else:
                        status_msg = "⚠️ Still not perfect, but we're learning."
                        continue_msg = "Continue building understanding from multiple failed attempts."
                    
                    prompt_reveal_reprocess = f"""Here's the actual output for training example {reprocess_i+1}:

Output grid:
{self.format_grid(task['train'][reprocess_i]['output'])}
{verification_feedback}

{status_msg}

{continue_msg}
"""

                    response_reveal_reprocess = self.call_ai_with_image(prompt_reveal_reprocess, [output_img])
                    num_phases += 1
                    
                    # Update correct rules stack
                    if reprocess_success:
                        print(f"  ✅ Training example {reprocess_i+1} solved with exploratory approach!")
                        self.correct_rules.append({
                            'example': reprocess_i+1,
                            'rule': f'Successfully solved training example {reprocess_i+1} after validation failure',
                            'status': 'correct_after_exploration'
                        })
                    else:
                        print(f"  ⚠️ Training example {reprocess_i+1} still not solved in exploratory mode.")
                
                # After re-processing, continue with any remaining examples in normal mode
                # (The loop will continue from where it left off)
        
        # Phase 4: Test input - ask for output
        print("\n" + "="*80)
        print("=== Phase 4: Test input - generate output ===")
        print("="*80)
        
        # Clear perfect rule validation mode before test (validation only for training examples)
        if self.perfect_rule_found:
            print("  ✨ Perfect rule was validated on training examples - will apply to test with confidence")
            self.perfect_rule_found = False  # Don't use validation mode for test
        
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
        
        # Build summary of validated rules
        rules_summary = ""
        
        # If we had a perfect rule that was validated, mention it with confidence
        if self.perfect_rule_description:
            rules_summary = f"\n🎯 VALIDATED PERFECT RULE:\n"
            rules_summary += f'   "{self.perfect_rule_description}"\n'
            rules_summary += f"   This rule was validated across multiple training examples with perfect accuracy.\n\n"
        
        if self.correct_rules:
            correct_count = sum(1 for r in self.correct_rules if r['status'] == 'correct')
            rules_summary += f"📚 TRAINING RESULTS FROM {len(task['train'])} EXAMPLES:\n"
            rules_summary += f"   ✅ {correct_count} examples solved correctly\n"
            for rule_info in self.correct_rules:
                if rule_info['status'] == 'correct':
                    rules_summary += f"   ✅ Example {rule_info['example']}: Rule validated\n"
                else:
                    rules_summary += f"   ⚠️ Example {rule_info['example']}: Had issues (diff: {rule_info.get('diff', '?')})\n"
            rules_summary += "\nApply your validated transformation rule confidently.\n"
        
        prompt_test = f"""This is the TEST. Apply the transformation rule you learned from the training examples.
{rules_summary}

🎯 FINAL APPLICATION: You've learned the pattern from 2+ training examples. Now apply it confidently to this test input.

Test input ({len(task['test'][0]['input'][0])}x{len(task['test'][0]['input'])} - width x height):
{self.format_grid(task['test'][0]['input'])}

CRITICAL DIMENSION INFERENCE:
- Training example outputs were: {', '.join([f"{len(t['output'][0])}x{len(t['output'])}" for t in task['train']])}
- Training example inputs were: {', '.join([f"{len(t['input'][0])}x{len(t['input'])}" for t in task['train']])}
- Test input is: {len(task['test'][0]['input'][0])}x{len(task['test'][0]['input'])}
- You must INFER the correct test output dimensions from the pattern you learned.

SOLVE WITH CONFIDENCE:

STEP 1: RECALL YOUR LEARNED PATTERN
- What transformation rule did you identify?
- How did it apply across both training examples?
- What's the core, consistent pattern?

STEP 2: APPLY TO TEST INPUT
For each step of the transformation:
   
   A. GENERATE HYPOTHETICALS (3-5 applications):
      - ⚠️ CRITICAL: ALL hypotheticals in this batch must use the SAME output dimensions
      - ⚠️ CRITICAL: Each hypothetical must be MEANINGFULLY DIFFERENT, not cosmetic changes
      - Infer the test output dimensions from training examples, commit to ONE size
      - Apply your learned rule to this test input
      - Show the transformation working step by step with genuinely different approaches
      - Test substantially different interpretations: different traversal patterns, different application orders, different edge case handling
      - Each hypothesis should explore a distinct way the rule could work
      - Avoid near-duplicates or trivial parameter tweaks
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