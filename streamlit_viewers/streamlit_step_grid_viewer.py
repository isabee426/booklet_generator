#!/usr/bin/env python3
"""
Streamlit Step Grid Viewer V2
Display puzzles step-by-step with efficient handling of many steps:
- Scrollable horizontal layout for steps
- Grouped by step type
- Efficient rendering for 100+ steps
"""

import streamlit as st
import json
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime
import sys

# Add utils to path for grid_to_image
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
try:
    from arc_visualizer import grid_to_image
except ImportError:
    grid_to_image = None

# Configure page
st.set_page_config(
    page_title="ARC Step Grid Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def find_puzzles():
    """Find all puzzles with visual step results"""
    results_dir = Path("visual_step_results")
    if not results_dir.exists():
        return []
    
    puzzles = []
    for puzzle_dir in results_dir.iterdir():
        if puzzle_dir.is_dir():
            # Check if it has training examples
            training_dirs = [d for d in puzzle_dir.iterdir() 
                           if d.is_dir() and (d.name.startswith("training_") or d.name.startswith("testing_"))]
            if training_dirs:
                puzzles.append(puzzle_dir.name)
    
    return sorted(puzzles)

def load_generalized_patterns(puzzle_id: str):
    """Load generalized patterns/booklet for a puzzle"""
    patterns_file = Path("visual_step_results") / puzzle_id / "generalized_steps" / "generalized_patterns.json"
    
    if not patterns_file.exists():
        return None
    
    try:
        with open(patterns_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading generalized patterns: {e}")
        return None

def load_step_results(puzzle_id: str, training_num: int, is_test: bool = False):
    """Load step results for a specific training example"""
    example_type = "testing" if is_test else "training"
    results_file = Path("visual_step_results") / puzzle_id / f"{example_type}_{training_num:02d}" / "results.json"
    
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data['version'] = data.get('version', 'v2')
            return data
    except json.JSONDecodeError as e:
        st.error(f"Error loading JSON file: {e}")
        st.error(f"File: {results_file}")
        # Try to read and show the problematic area
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                content = f.read()
                error_pos = e.pos if hasattr(e, 'pos') else 0
                start = max(0, error_pos - 200)
                end = min(len(content), error_pos + 200)
                st.code(f"Error around position {error_pos}:\n{content[start:end]}")
        except:
            pass
        return None
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return None

def get_step_image(puzzle_id: str, training_num: int, step_num: int, is_test: bool = False, results_data: Dict = None):
    """Get the image for a specific step"""
    example_type = "testing" if is_test else "training"
    img_path = Path("visual_step_results") / puzzle_id / f"{example_type}_{training_num:02d}" / f"step_{step_num:02d}_final.png"
    
    if img_path.exists():
        img = Image.open(img_path)
        
        # For v3/v4/v5/v6 crop steps, add overlay if metadata available
        if results_data and results_data.get('version') in ['v3', 'v4', 'v5', 'v6']:
            steps = results_data.get('steps', [])
            step_data = next((s for s in steps if s.get('step_num') == step_num), None)
            if step_data and step_data.get('is_crop_step'):
                img = add_crop_overlay(img, step_data.get('crop_metadata'))
        
        return img
    
    return None

def add_crop_overlay(img: Image.Image, crop_metadata: Dict) -> Image.Image:
    """Add visual overlay showing crop region"""
    from PIL import ImageDraw, ImageFont
    
    if not crop_metadata:
        return img
    
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    
    # Get crop bbox (scaled to image size)
    crop_bbox = crop_metadata.get('crop_bbox')
    if crop_bbox:
        min_r, min_c, max_r, max_c = crop_bbox
        # Scale coordinates (assuming 30px per cell based on grid_to_image)
        scale = 30
        x1 = min_c * scale
        y1 = min_r * scale
        x2 = (max_c + 1) * scale
        y2 = (max_r + 1) * scale
        
        # Draw semi-transparent rectangle
        overlay_rect = Image.new('RGBA', overlay.size, (255, 0, 0, 0))
        draw_rect = ImageDraw.Draw(overlay_rect)
        draw_rect.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 30), outline='red', width=2)
        overlay = Image.alpha_composite(overlay.convert('RGBA'), overlay_rect).convert('RGB')
        draw = ImageDraw.Draw(overlay)
        
        # Add label
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        if font:
            draw.text((x1 + 5, y1 + 5), "CROP", fill='red', font=font)
        else:
            draw.text((x1 + 5, y1 + 5), "CROP", fill='red')
    
    return overlay

def get_crop_images(puzzle_id: str, training_num: int, step_num: int, is_test: bool = False):
    """Get cropped images for a crop step"""
    example_type = "testing" if is_test else "training"
    base_path = Path("visual_step_results") / puzzle_id / f"{example_type}_{training_num:02d}"
    
    crop_input_path = base_path / f"step_{step_num:02d}_crop_input.png"
    crop_target_path = base_path / f"step_{step_num:02d}_crop_target.png"
    
    crop_input = Image.open(crop_input_path) if crop_input_path.exists() else None
    crop_target = Image.open(crop_target_path) if crop_target_path.exists() else None
    
    return crop_input, crop_target

def get_crop_transform_image(puzzle_id: str, training_num: int, step_num: int, is_test: bool = False):
    """Get cropped transformation image"""
    example_type = "testing" if is_test else "training"
    crop_transform_path = Path("visual_step_results") / puzzle_id / f"{example_type}_{training_num:02d}" / f"step_{step_num:02d}_crop_transform.png"
    
    if crop_transform_path.exists():
        return Image.open(crop_transform_path)
    return None

def get_ground_truth_image(puzzle_id: str, training_num: int, step_num: int, is_test: bool = False):
    """Get ground truth image for a specific step"""
    example_type = "testing" if is_test else "training"
    gt_path = Path("visual_step_results") / puzzle_id / f"{example_type}_{training_num:02d}" / f"step_{step_num:02d}_ground_truth.png"
    
    if gt_path.exists():
        return Image.open(gt_path)
    return None

def get_training_examples(puzzle_id: str):
    """Get list of training example numbers for a puzzle, sorted by most recently changed first"""
    results_dir = Path("visual_step_results") / puzzle_id
    if not results_dir.exists():
        return [], []
    
    training_examples = []
    testing_examples = []
    
    for example_dir in results_dir.iterdir():
        if example_dir.is_dir():
            results_file = example_dir / "results.json"
            # Get modification time of results.json, or directory if file doesn't exist
            if results_file.exists():
                mtime = results_file.stat().st_mtime
            else:
                mtime = example_dir.stat().st_mtime
            
            if example_dir.name.startswith("training_"):
                try:
                    num = int(example_dir.name.split("_")[1])
                    training_examples.append((num, mtime))
                except:
                    pass
            elif example_dir.name.startswith("testing_"):
                try:
                    num = int(example_dir.name.split("_")[1])
                    testing_examples.append((num, mtime))
                except:
                    pass
    
    # Sort by modification time (most recent first), then by number
    training_nums = [num for num, _ in sorted(training_examples, key=lambda x: (-x[1], x[0]))]
    testing_nums = [num for num, _ in sorted(testing_examples, key=lambda x: (-x[1], x[0]))]
    
    return training_nums, testing_nums

def group_steps_by_type(steps: List[Dict]) -> Dict[str, List[Dict]]:
    """Group steps by type for better organization"""
    grouped = defaultdict(list)
    
    for step in steps:
        step_type = "other"
        
        if step.get('is_crop_step'):
            step_type = "crop"
        elif step.get('is_uncrop_step'):
            step_type = "uncrop"
        elif step.get('is_cropped_view'):
            step_type = "transform"
        elif step.get('is_reference_step'):
            step_type = "reference"
        elif step.get('is_final_step'):
            step_type = "final"
        elif step.get('is_removal_step'):
            step_type = "removal"
        elif step.get('step_num') == 1:
            step_type = "initial"
        
        grouped[step_type].append(step)
    
    return dict(grouped)

def render_step_card(step_data: Dict, puzzle_id: str, training_num: int, 
                    image_size: int, show_gt: bool, show_desc: bool, 
                    version: str, is_test: bool = False, results_data: Dict = None):
    """Render a single step card"""
    step_num = step_data.get('step_num', 0)
    
    # Container for step
    with st.container():
        # Step header
        step_type_icon = {
            'crop': '🔲',
            'uncrop': '🔓',
            'transform': '🔍',
            'reference': '⭐',
            'final': '✅',
            'removal': '🗑️',
            'initial': '🎬',
            'other': '📝'
        }
        
        step_type = "other"
        if step_data.get('is_crop_step'):
            step_type = 'crop'
        elif step_data.get('is_uncrop_step'):
            step_type = 'uncrop'
        elif step_data.get('is_cropped_view'):
            step_type = 'transform'
        elif step_data.get('is_reference_step'):
            step_type = 'reference'
        elif step_data.get('is_final_step'):
            step_type = 'final'
        elif step_data.get('is_removal_step'):
            step_type = 'removal'
        elif step_data.get('step_num') == 1:
            step_type = 'initial'
        
        icon = step_type_icon.get(step_type, '📝')
        st.markdown(f"**{icon} Step {step_num}**")
        
        # Get image
        gen_img = get_step_image(puzzle_id, training_num, step_num, is_test, results_data)
        
        if gen_img:
            # Crop step special handling
            if version in ['v3', 'v4', 'v5', 'v6'] and step_data.get('is_crop_step'):
                st.image(gen_img, width=image_size, use_container_width=False)
                
                # Show cropped images
                crop_input, crop_target = get_crop_images(puzzle_id, training_num, step_num, is_test)
                if crop_input or crop_target:
                    cols = st.columns(2)
                    if crop_input:
                        with cols[0]:
                            st.caption("**Cropped Input:**")
                            st.image(crop_input, width=image_size // 2, use_container_width=False)
                    if crop_target:
                        with cols[1]:
                            st.caption("**Cropped Target (Output):**")
                            st.image(crop_target, width=image_size // 2, use_container_width=False)
                
                # Also show cropped_output from step data if available
                if step_data.get('cropped_output') and grid_to_image:
                    try:
                        cropped_output_grid = step_data['cropped_output']
                        cropped_output_img = grid_to_image(cropped_output_grid, 50)
                        st.caption("**Cropped Output (from step data):**")
                        st.image(cropped_output_img, width=image_size // 2, use_container_width=False)
                    except Exception as e:
                        pass  # Skip if can't render
                
                if step_data.get('is_new_object'):
                    st.caption("✨ **New Object**")
                if step_data.get('object_num'):
                    st.caption(f"🔷 Object #{step_data['object_num']}")
            
            # Cropped transformation view
            elif version in ['v3', 'v4', 'v5', 'v6'] and step_data.get('is_cropped_view'):
                crop_transform_img = get_crop_transform_image(puzzle_id, training_num, step_num, is_test)
                if crop_transform_img:
                    st.image(crop_transform_img, width=image_size // 2, use_container_width=False)
                else:
                    st.image(gen_img, width=image_size // 2, use_container_width=False)
                
                # Show cropped ground truth and cropped output side by side
                if step_data.get('cropped_ground_truth') or step_data.get('cropped_output'):
                    cols = st.columns(2)
                    if step_data.get('cropped_ground_truth') and grid_to_image:
                        try:
                            cropped_gt_grid = step_data['cropped_ground_truth']
                            cropped_gt_img = grid_to_image(cropped_gt_grid, 50)
                            with cols[0]:
                                st.caption("**Cropped Ground Truth:**")
                                st.image(cropped_gt_img, width=image_size // 2, use_container_width=False)
                        except Exception as e:
                            pass
                    if step_data.get('cropped_output') and grid_to_image:
                        try:
                            cropped_output_grid = step_data['cropped_output']
                            cropped_output_img = grid_to_image(cropped_output_grid, 50)
                            with cols[1]:
                                st.caption("**Cropped Output:**")
                                st.image(cropped_output_img, width=image_size // 2, use_container_width=False)
                        except Exception as e:
                            pass
            
            # Regular step
            else:
                st.image(gen_img, width=image_size, use_container_width=False)
        
        # Ground truth
        if show_gt and not is_test:
            gt_img = get_ground_truth_image(puzzle_id, training_num, step_num, is_test)
            if gt_img:
                st.caption("**Ground Truth:**")
                st.image(gt_img, width=image_size, use_container_width=False)
        
        # Description
        if show_desc:
            description = step_data.get('description') or step_data.get('final_description', '')
            if description:
                st.caption(f"*{description[:100]}{'...' if len(description) > 100 else ''}*")
        
        # Metadata
        if step_data.get('object_num'):
            st.caption(f"Object {step_data['object_num']}")
        
        if step_data.get('is_new_object'):
            st.caption("✨ New object")
        
        # Accuracy
        accuracy = step_data.get('accuracy')
        if accuracy is not None:
            if accuracy >= 0.99:
                st.success(f"✓ {accuracy:.1%}")
            elif accuracy >= 0.95:
                st.info(f"{accuracy:.1%}")
            elif accuracy > 0.5:
                st.warning(f"{accuracy:.1%}")
            else:
                st.error(f"{accuracy:.1%}")
        
        if step_data.get('used_ground_truth'):
            st.caption("⚠️ Used GT")

def display_generalized_steps(generalized_patterns: Dict):
    """Display generalized steps/patterns"""
    st.subheader("📋 Generalized Step Sequence")
    
    generalized_steps = generalized_patterns.get('generalized_step_sequence', [])
    transition_determinants = generalized_patterns.get('transition_determinants', {})
    booklet_pattern = generalized_patterns.get('booklet_pattern', {})
    transformation_rule = generalized_patterns.get('transformation_rule', {})
    
    if generalized_steps:
        st.markdown(f"**Found {len(generalized_steps)} generalized steps:**")
        for i, step in enumerate(generalized_steps):
            with st.expander(f"Step {step.get('step_num', i+1)}: {step.get('step_type', 'N/A').upper()}"):
                st.markdown(f"**Description:** {step.get('description', 'N/A')}")
                if step.get('exact_colors'):
                    st.markdown(f"**Colors:** {', '.join(step.get('exact_colors', []))}")
                if step.get('exact_spacing'):
                    st.markdown(f"**Spacing:** {step.get('exact_spacing', 'N/A')}")
                if step.get('exact_pattern'):
                    st.markdown(f"**Pattern:** {step.get('exact_pattern', 'N/A')}")
                if step.get('conditions'):
                    st.markdown(f"**Conditions:** {step.get('conditions', 'N/A')}")
                if step.get('applies_to'):
                    st.markdown(f"**Applies To:** {step.get('applies_to', 'N/A')}")
                if step.get('adaptation'):
                    st.markdown(f"**Adaptation:** {step.get('adaptation', 'N/A')}")
    else:
        st.info("No generalized step sequence found")
    
    if transition_determinants:
        st.subheader("🔀 Transition Determinants")
        determinants = transition_determinants.get('transition_determinants', [])
        if determinants:
            st.markdown(f"**Found {len(determinants)} transition types:**")
            for i, det in enumerate(determinants):
                with st.expander(f"Transition {i+1}: {det.get('transition_type', 'N/A')}"):
                    props = det.get('determining_properties', {})
                    st.markdown(f"**Determining Properties:**")
                    if props.get('colors'):
                        st.markdown(f"- Colors: {', '.join(props.get('colors', []))}")
                    if props.get('shapes'):
                        st.markdown(f"- Shapes: {', '.join(props.get('shapes', []))}")
                    if props.get('positions'):
                        st.markdown(f"- Positions: {props.get('positions', 'N/A')}")
                    
                    conditions = det.get('conditions', [])
                    if conditions:
                        st.markdown(f"**Conditions:**")
                        for cond in conditions:
                            st.markdown(f"- {cond}")
                    
                    trans_rule = det.get('transition_rule', {})
                    if trans_rule:
                        st.markdown(f"**Transition Rule:**")
                        if trans_rule.get('exact_transformation'):
                            st.markdown(f"- Transformation: {trans_rule.get('exact_transformation')}")
                        if trans_rule.get('exact_pattern'):
                            st.markdown(f"- Pattern: {trans_rule.get('exact_pattern')}")
        else:
            st.info("No transition determinants found")
    
    if booklet_pattern:
        st.subheader("📖 Booklet Pattern")
        pattern_gen = booklet_pattern.get('generalization', '')
        if pattern_gen:
            st.markdown(f"**Generalization:** {pattern_gen}")
        
        step_seq = booklet_pattern.get('step_sequence_pattern', [])
        if step_seq:
            st.markdown(f"**Step Sequence Pattern ({len(step_seq)} steps):**")
            for step in step_seq:
                st.markdown(f"- Step {step.get('order', 'N/A')} [{step.get('step_type', 'N/A')}]: {step.get('description', 'N/A')}")
        
        trans_rules = booklet_pattern.get('transformation_rules', [])
        if trans_rules:
            st.markdown(f"**Transformation Rules ({len(trans_rules)}):**")
            for rule in trans_rules:
                st.markdown(f"- {rule.get('rule', 'N/A')} (applies to: {rule.get('applies_to', 'N/A')})")
    
    if transformation_rule:
        st.subheader("⚙️ Transformation Rule")
        rule_desc = transformation_rule.get('rule_description', '')
        if rule_desc:
            st.markdown(f"**Rule:** {rule_desc}")
        
        gen = transformation_rule.get('generalization', {})
        if gen.get('abstract_rule'):
            st.markdown(f"**Abstract Rule:** {gen.get('abstract_rule')}")
        if gen.get('color_logic'):
            st.markdown(f"**Color Logic:** {gen.get('color_logic')}")
        if gen.get('shape_logic'):
            st.markdown(f"**Shape Logic:** {gen.get('shape_logic')}")

def main():
    st.title("📊 ARC Step Grid Viewer")
    st.markdown("**Enhanced viewer with horizontal scrolling** - Scroll horizontally to navigate through all steps")
    
    # Find all puzzles
    puzzles = find_puzzles()
    
    if not puzzles:
        st.error("No visual step results found. Run visual_step_generator.py first.")
        st.code("python scripts/visual_step_generator_v4.py --puzzle <puzzle_id> --all")
        return
    
    # Sidebar - Puzzle selection
    st.sidebar.header("🎯 Select Puzzle")
    selected_puzzle = st.sidebar.selectbox("Puzzle ID", puzzles)
    
    if not selected_puzzle:
        return
    
    # Get training/testing examples
    training_examples, testing_examples = get_training_examples(selected_puzzle)
    
    if not training_examples and not testing_examples:
        st.error(f"No examples found for puzzle {selected_puzzle}")
        return
    
    # Example type selection
    example_type = st.sidebar.radio("Example Type", ["Training", "Testing"], 
                                    index=0 if training_examples else 1)
    is_test = (example_type == "Testing")
    examples = testing_examples if is_test else training_examples
    
    if not examples:
        st.error(f"No {example_type.lower()} examples found")
        return
    
    # Example selection
    selected_example = st.sidebar.selectbox(
        f"{example_type} Example",
        examples,
        format_func=lambda x: f"{example_type} {x}"
    )
    
    # Load results
    results = load_step_results(selected_puzzle, selected_example, is_test=is_test)
    
    if not results:
        st.error(f"No results found for {example_type.lower()} example {selected_example}")
        return
    
    steps = results.get('steps', [])
    version = results.get('version', 'v2')
    
    st.sidebar.metric("📊 Total Steps", len(steps))
    st.sidebar.metric("🔖 Version", version.upper())
    
    # Show step breakdown
    if steps:
        step_types_count = {}
        for step in steps:
            step_type = "other"
            if step.get('is_crop_step'):
                step_type = "crop"
            elif step.get('is_uncrop_step'):
                step_type = "uncrop"
            elif step.get('is_cropped_view'):
                step_type = "transform"
            elif step.get('is_reference_step'):
                step_type = "reference"
            elif step.get('is_final_step'):
                step_type = "final"
            elif step.get('is_removal_step'):
                step_type = "removal"
            elif step.get('step_num') == 1:
                step_type = "initial"
            
            step_types_count[step_type] = step_types_count.get(step_type, 0) + 1
        
        if step_types_count:
            st.sidebar.caption("**Step Breakdown:**")
            for stype, count in sorted(step_types_count.items()):
                st.sidebar.caption(f"  • {stype}: {count}")
    
    # Check for generalized patterns
    generalized_patterns = load_generalized_patterns(selected_puzzle)
    show_generalized = st.sidebar.checkbox("Show Generalized Steps", value=False)
    
    # Display options
    st.sidebar.divider()
    st.sidebar.header("⚙️ Display Options")
    
    view_mode = st.sidebar.radio("View Mode", ["All Steps", "Grouped by Type", "Filtered"])
    show_ground_truth = st.sidebar.checkbox("Show Ground Truth", value=True)
    show_descriptions = st.sidebar.checkbox("Show Descriptions", value=True)
    image_size = st.sidebar.slider("Image Size", min_value=100, max_value=400, value=150, step=50)
    
    # Filter options
    if view_mode == "Filtered":
        st.sidebar.divider()
        st.sidebar.header("🔍 Filters")
        show_crops = st.sidebar.checkbox("Show Crop Steps", value=True)
        show_uncrops = st.sidebar.checkbox("Show Uncrop Steps", value=True)
        show_transforms = st.sidebar.checkbox("Show Transform Steps", value=True)
        show_references = st.sidebar.checkbox("Show Reference Steps", value=True)
        show_others = st.sidebar.checkbox("Show Other Steps", value=True)
    
    # Main display
    if show_generalized and generalized_patterns:
        st.header(f"🧩 Puzzle: {selected_puzzle} - Generalized Steps")
        display_generalized_steps(generalized_patterns)
        st.divider()
    
    st.header(f"🧩 Puzzle: {selected_puzzle} - {example_type} Example {selected_example}")
    
    # Show puzzle info with better layout
    if results.get('input_grid') and results.get('expected_output_grid'):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            input_size = f"{len(results['input_grid'])}×{len(results['input_grid'][0])}"
            st.metric("📥 Input Size", input_size)
        with col2:
            output_size = f"{len(results['expected_output_grid'])}×{len(results['expected_output_grid'][0])}"
            st.metric("📤 Output Size", output_size)
        with col3:
            st.metric("📊 Total Steps", len(steps))
        with col4:
            if results.get('timestamp'):
                try:
                    ts_str = results['timestamp']
                    # Handle different timestamp formats
                    if 'T' in ts_str:
                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00').split('.')[0])
                    else:
                        ts = datetime.fromisoformat(ts_str)
                    st.caption(f"🕒 Generated:\n{ts.strftime('%Y-%m-%d %H:%M')}")
                except:
                    st.caption(f"🕒 {results.get('timestamp', 'N/A')}")
        
        # Display input and output grids
        st.divider()
        st.subheader("📋 Input and Output Grids")
        
        if grid_to_image:
            col_input, col_output = st.columns(2)
            
            with col_input:
                st.markdown("**📥 Input Grid**")
                try:
                    input_grid = results['input_grid']
                    input_img = grid_to_image(input_grid, 50)
                    st.image(input_img, use_container_width=True)
                    st.caption(f"Size: {len(input_grid)}×{len(input_grid[0])}")
                except Exception as e:
                    st.error(f"Error rendering input grid: {e}")
            
            with col_output:
                st.markdown("**📤 Expected Output Grid**")
                try:
                    output_grid = results['expected_output_grid']
                    output_img = grid_to_image(output_grid, 50)
                    st.image(output_img, use_container_width=True)
                    st.caption(f"Size: {len(output_grid)}×{len(output_grid[0])}")
                except Exception as e:
                    st.error(f"Error rendering output grid: {e}")
        else:
            st.warning("⚠️ grid_to_image not available. Install arc_visualizer to see grid images.")
        
        st.divider()
    
    # Group steps
    grouped_steps = group_steps_by_type(steps)
    
    # Filter steps if needed
    if view_mode == "Filtered":
        filtered_steps = []
        for step in steps:
            if step.get('is_crop_step') and show_crops:
                filtered_steps.append(step)
            elif step.get('is_uncrop_step') and show_uncrops:
                filtered_steps.append(step)
            elif step.get('is_cropped_view') and show_transforms:
                filtered_steps.append(step)
            elif step.get('is_reference_step') and show_references:
                filtered_steps.append(step)
            elif not any([step.get('is_crop_step'), step.get('is_uncrop_step'), 
                          step.get('is_cropped_view'), step.get('is_reference_step')]) and show_others:
                filtered_steps.append(step)
        steps_to_show = filtered_steps
    elif view_mode == "Grouped by Type":
        steps_to_show = steps  # Will be grouped in display
    else:
        steps_to_show = steps
    
    # Display steps
    if view_mode == "Grouped by Type":
        # Show grouped by type
        type_order = ['initial', 'reference', 'crop', 'transform', 'uncrop', 'removal', 'final', 'other']
        type_labels = {
            'initial': '🎬 Initial Steps',
            'reference': '⭐ Reference Objects',
            'crop': '🔲 Crop Steps',
            'transform': '🔍 Transformations',
            'uncrop': '🔓 Uncrop Steps',
            'removal': '🗑️ Removal Steps',
            'final': '✅ Final Steps',
            'other': '📝 Other Steps'
        }
        
        for step_type in type_order:
            if step_type in grouped_steps and grouped_steps[step_type]:
                type_steps = grouped_steps[step_type]
                with st.expander(f"{type_labels[step_type]} ({len(type_steps)} steps)", expanded=True):
                    # Inject CSS for this expander's horizontal scrolling
                    st.markdown("""
                    <style>
                    /* Horizontal scroll for grouped steps */
                    div[data-testid="stHorizontalBlock"] {
                        overflow-x: auto;
                        overflow-y: hidden;
                        display: flex;
                        flex-wrap: nowrap;
                        gap: 15px;
                        padding: 10px 0;
                        -webkit-overflow-scrolling: touch;
                        scrollbar-width: thin;
                        scrollbar-color: #888 #f1f1f1;
                    }
                    div[data-testid="stHorizontalBlock"]::-webkit-scrollbar {
                        height: 10px;
                    }
                    div[data-testid="stHorizontalBlock"]::-webkit-scrollbar-track {
                        background: #f1f1f1;
                        border-radius: 8px;
                    }
                    div[data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb {
                        background: #888;
                        border-radius: 8px;
                    }
                    div[data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb:hover {
                        background: #555;
                    }
                    div[data-testid="stHorizontalBlock"] > div {
                        flex: 0 0 auto;
                        min-width: 200px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Show all steps in one scrollable row
                    num_steps_in_type = len(type_steps)
                    max_cols = min(num_steps_in_type, 30)
                    
                    if num_steps_in_type <= max_cols:
                        cols = st.columns(num_steps_in_type)
                        for idx, step in enumerate(type_steps):
                            with cols[idx]:
                                render_step_card(step, selected_puzzle, selected_example, 
                                               image_size, show_ground_truth, show_descriptions,
                                               version, is_test, results)
                    else:
                        # Show in batches but all scrollable
                        for batch_start in range(0, num_steps_in_type, max_cols):
                            batch_end = min(batch_start + max_cols, num_steps_in_type)
                            batch_steps = type_steps[batch_start:batch_end]
                            cols = st.columns(len(batch_steps))
                            for idx, step in enumerate(batch_steps):
                                with cols[idx]:
                                    render_step_card(step, selected_puzzle, selected_example,
                                                   image_size, show_ground_truth, show_descriptions,
                                                   version, is_test, results)
    else:
        # Show all steps in scrollable horizontal layout with scrollbar
        st.markdown(f"**📊 Showing {len(steps_to_show)} steps - Scroll horizontally →**")
        st.caption(f"💡 Use mouse wheel, arrow keys, or drag the scrollbar to navigate through all steps")
        
        if steps_to_show:
            # Inject CSS for horizontal scrolling container
            st.markdown("""
            <style>
            /* Make the main container scrollable */
            .main .block-container {
                max-width: 100%;
            }
            /* Force horizontal scrolling for columns */
            div[data-testid="stHorizontalBlock"] {
                overflow-x: auto;
                overflow-y: hidden;
                display: flex;
                flex-wrap: nowrap;
                gap: 15px;
                padding: 10px 0;
                -webkit-overflow-scrolling: touch;
                scrollbar-width: thin;
                scrollbar-color: #888 #f1f1f1;
            }
            /* Custom scrollbar styling */
            div[data-testid="stHorizontalBlock"]::-webkit-scrollbar {
                height: 12px;
            }
            div[data-testid="stHorizontalBlock"]::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 10px;
            }
            div[data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb {
                background: #888;
                border-radius: 10px;
            }
            div[data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb:hover {
                background: #555;
            }
            /* Prevent column wrapping */
            div[data-testid="stHorizontalBlock"] > div {
                flex: 0 0 auto;
                min-width: 200px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Calculate optimal number of columns to show at once
            # Streamlit has a practical limit, so we'll show a reasonable number
            max_cols_at_once = 30
            num_steps = len(steps_to_show)
            
            if num_steps <= max_cols_at_once:
                # Show all steps in one scrollable row
                cols = st.columns(num_steps)
                for idx, step in enumerate(steps_to_show):
                    with cols[idx]:
                        render_step_card(step, selected_puzzle, selected_example,
                                       image_size, show_ground_truth, show_descriptions,
                                       version, is_test, results)
            else:
                # For many steps, show in a scrollable container
                # We'll create multiple rows if needed, but make each row scrollable
                st.info(f"📋 Showing all {num_steps} steps. Scroll horizontally to see more.")
                
                # Create columns for all steps - Streamlit will handle it
                # We'll batch them into rows of max_cols_at_once
                num_batches = (num_steps + max_cols_at_once - 1) // max_cols_at_once
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * max_cols_at_once
                    end_idx = min(start_idx + max_cols_at_once, num_steps)
                    batch_steps = steps_to_show[start_idx:end_idx]
                    
                    cols = st.columns(len(batch_steps))
                    for col_idx, step in enumerate(batch_steps):
                        with cols[col_idx]:
                            render_step_card(step, selected_puzzle, selected_example,
                                           image_size, show_ground_truth, show_descriptions,
                                           version, is_test, results)
                    
                    # Add a small separator between batches (but they'll scroll together)
                    if batch_idx < num_batches - 1:
                        st.markdown("<br>", unsafe_allow_html=True)
    
    # Display comprehensive analysis if available (v3/v4/v5) - after step visuals
    if results.get('analysis'):
        st.divider()
        with st.expander("📊 Comprehensive Puzzle Analysis", expanded=True):
            analysis = results['analysis']
            
            # Input-Input Analysis
            if analysis.get('input_input'):
                st.subheader("🔍 Input-Input Differences")
                input_input = analysis['input_input']
                if input_input.get('input_pair'):
                    st.caption(f"**Comparing:** {input_input['input_pair']}")
                if input_input.get('differences'):
                    st.write("**Differences:**")
                    for diff in input_input['differences']:
                        st.write(f"• {diff}")
                if input_input.get('output_implications'):
                    st.info(f"**Output Implications:** {input_input['output_implications']}")
            
            # Output-Output Analysis
            if analysis.get('output_output'):
                st.subheader("🔍 Output-Output Similarities")
                output_output = analysis['output_output']
                if output_output.get('output_pair'):
                    st.caption(f"**Comparing:** {output_output['output_pair']}")
                if output_output.get('variations'):
                    st.write("**Variations:**")
                    for var in output_output['variations']:
                        st.write(f"• {var}")
                if output_output.get('input_correlation'):
                    st.info(f"**Input Correlation:** {output_output['input_correlation']}")
            
            # Input-Output Analysis
            if analysis.get('input_output'):
                st.subheader("🔄 Input-Output Transitions")
                input_output = analysis['input_output']
                if input_output.get('type'):
                    st.caption(f"**Type:** {input_output['type']}")
                if input_output.get('description'):
                    st.write(f"**Description:** {input_output['description']}")
                if input_output.get('applies_to'):
                    st.write(f"**Applies To:** {input_output['applies_to']}")
                if input_output.get('conditions'):
                    st.write(f"**Conditions:** {input_output['conditions']}")
                if input_output.get('uses_reference'):
                    st.info(f"**Reference Usage:** {input_output['uses_reference']}")
            
            # Reference Objects
            if analysis.get('reference_objects'):
                st.subheader("⭐ Reference Objects")
                ref_objects = analysis['reference_objects']
                if ref_objects.get('reference_objects'):
                    for idx, ref_obj in enumerate(ref_objects['reference_objects'], 1):
                        with st.container():
                            bbox = ref_obj.get('bbox', [])
                            if bbox:
                                st.write(f"**Reference Object {idx}:** BBox [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
                            if ref_obj.get('usage'):
                                st.caption(f"Usage: {ref_obj['usage']}")
                            if ref_obj.get('description'):
                                st.write(f"{ref_obj['description']}")
                            st.divider()
                if ref_objects.get('how_used'):
                    st.info(f"**How Used:** {ref_objects['how_used']}")
            
            # Grid Size Analysis
            if analysis.get('grid_size'):
                st.subheader("📐 Grid Size Changes")
                grid_size = analysis['grid_size']
                if grid_size.get('size_info'):
                    st.write("**Size Information:**")
                    for size_info in grid_size['size_info']:
                        example_num = size_info.get('example', '?')
                        input_sz = size_info.get('input_size', '?')
                        output_sz = size_info.get('output_size', '?')
                        ratio = size_info.get('ratio', '?')
                        st.write(f"Example {example_num}: {input_sz} → {output_sz} (ratio: {ratio})")
                if grid_size.get('size_change_type'):
                    st.info(f"**Size Change Type:** {grid_size['size_change_type']}")
    
    # Summary statistics
    st.sidebar.divider()
    st.sidebar.header("📈 Summary")
    
    if steps:
        accuracies = [s.get('accuracy', 0) for s in steps if s.get('accuracy') is not None]
        if accuracies:
            avg_acc = np.mean(accuracies)
            final_acc = accuracies[-1] if accuracies else 0
            st.sidebar.metric("Avg Accuracy", f"{avg_acc:.1%}")
            st.sidebar.metric("Final Accuracy", f"{final_acc:.1%}")
        
        # Step type counts
        st.sidebar.caption("**Step Types:**")
        for step_type, type_steps in grouped_steps.items():
            st.sidebar.caption(f"{len(type_steps)} {step_type} steps")

if __name__ == "__main__":
    main()
