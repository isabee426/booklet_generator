#!/usr/bin/env python3
"""
Streamlit App for ARC Booklet Visualizer
Shows step-by-step instructions and visualizations for ARC puzzle solving
"""

import streamlit as st
import json
import os
from pathlib import Path
from PIL import Image
import numpy as np
from arc_visualizer import grid_to_image
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="ARC Booklet Visualizer",
    page_icon="🧩",
    layout="wide"
)

def load_booklet_json(file_path):
    """Load booklet data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_task_json(task_name):
    """Try to load the original task JSON file"""
    # Look in common ARC data directories
    possible_paths = [
        Path(f"ARC-AGI-2/data/training/{task_name}.json"),
        Path(f"ARC-AGI-2/data/evaluation/{task_name}.json"),
        Path(f"ARC-AGI-2/ARC-AGI-2/data/training/{task_name}.json"),
        Path(f"ARC-AGI-2/ARC-AGI-2/data/evaluation/{task_name}.json"),
        Path(f"{task_name}.json"),
    ]
    
    for path in possible_paths:
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
    return None

def find_booklet_files(directory="test"):
    """Find all booklet JSON files in the directory"""
    booklet_files = []
    # Look in test/ directory first, then fallback to current directory
    test_dir = Path(directory)
    if test_dir.exists():
        for file in test_dir.glob("*_booklet.json"):
            booklet_files.append(str(file))
    
    # Also check current directory as fallback
    for file in Path(".").glob("*_booklet.json"):
        booklet_files.append(str(file))
    
    return sorted(list(set(booklet_files)))  # Remove duplicates

def visualize_grid(grid_data, cell_size=30):
    """Create image from grid data"""
    if grid_data is None or len(grid_data) == 0:
        return None
    img = grid_to_image(grid_data, cell_size)
    return img

def compare_grids(grid1, grid2):
    """Compare two grids and return match percentage"""
    if grid1 is None or grid2 is None:
        return None
    if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
        return 0.0
    
    total = len(grid1) * len(grid1[0])
    matches = sum(1 for i in range(len(grid1)) for j in range(len(grid1[0])) 
                  if grid1[i][j] == grid2[i][j])
    return matches / total

def extract_instruction_text(steps):
    """Extract instruction generation and update steps"""
    instructions = []
    for step in steps:
        instruction = step.get('instruction', '')
        if 'Instructions' in instruction or 'INSTRUCTIONS' in instruction or 'Step-by-Step' in instruction:
            instructions.append(step)
    return instructions

def main():
    st.title("🧩 ARC-AGI Booklet Visualizer")
    st.markdown("Visualize step-by-step instructions for solving ARC-AGI puzzles")
    
    # Sidebar for file selection
    st.sidebar.header("📁 Booklet Selection")
    
    # Find booklet files
    booklet_files = find_booklet_files()
    
    if not booklet_files:
        st.error("No booklet files found. Please run the solver first.")
        st.info("To generate booklets, run: `python arc-booklets-solver.py <task_file>`")
        return
    
    # Add sorting options
    sort_by = st.sidebar.selectbox(
        "Sort by:",
        ["Name", "Success (✅ first)", "Failure (❌ first)"]
    )
    
    # Load all booklets for sorting
    booklet_info = []
    for file in booklet_files:
        try:
            data = load_booklet_json(file)
            booklet_info.append({
                'file': file,
                'name': Path(file).name,
                'accuracy': data.get('accuracy'),
                'task_name': data.get('task_name', '')
            })
        except:
            booklet_info.append({
                'file': file,
                'name': Path(file).name,
                'accuracy': None,
                'task_name': ''
            })
    
    # Sort booklets
    if sort_by == "Success (✅ first)":
        booklet_info.sort(key=lambda x: (x['accuracy'] != 1.0, x['name']))
    elif sort_by == "Failure (❌ first)":
        booklet_info.sort(key=lambda x: (x['accuracy'] == 1.0, x['name']))
    else:
        booklet_info.sort(key=lambda x: x['name'])
    
    # Format display names with status
    def format_name(info):
        acc = info['accuracy']
        if acc == 1.0:
            return f"✅ {info['name']}"
        elif acc == 0.0:
            return f"❌ {info['name']}"
        else:
            return f"❔ {info['name']}"
    
    selected_file = st.sidebar.selectbox(
        "Select a booklet file:",
        [info['file'] for info in booklet_info],
        format_func=lambda x: format_name(next(i for i in booklet_info if i['file'] == x))
    )
    
    if not selected_file:
        st.warning("Please select a booklet file from the sidebar.")
        return
    
    # Load the booklet
    try:
        booklet_data = load_booklet_json(selected_file)
    except Exception as e:
        st.error(f"Error loading booklet: {e}")
        return
    
    # Try to load original task data
    task_data = load_task_json(booklet_data['task_name'])
    
    # Display task information
    st.header(f"Task: {booklet_data['task_name']}")
    
    # Enhanced metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Steps", len(booklet_data['steps']))
    
    with col2:
        accuracy = booklet_data.get('accuracy', 'N/A')
        if isinstance(accuracy, (int, float)):
            accuracy_percent = accuracy * 100
            st.metric("Accuracy", f"{accuracy_percent:.0f}%", 
                     delta="✅ Correct" if accuracy == 1.0 else "❌ Incorrect")
        else:
            st.metric("Accuracy", "N/A")
    
    with col3:
        prediction_shape = booklet_data.get('final_prediction_shape', 'N/A')
        st.metric("Prediction Shape", str(prediction_shape))
    
    with col4:
        actual_shape = booklet_data.get('actual_output_shape', 'N/A')
        st.metric("Expected Shape", str(actual_shape))
    
    # Show training set size if available
    if task_data:
        st.info(f"📚 Training examples: {len(task_data.get('train', []))} | Test examples: {len(task_data.get('test', []))}")
    
    st.divider()
    
    # Add tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Step-by-Step", 
        "📊 Overview", 
        "🎯 Final Prediction",
        "📚 Training Data",
        "🧠 Generated Instructions"
    ])
    
    # TAB 1: Step-by-Step Instructions
    with tab1:
        st.header("📋 Step-by-Step Instructions")
        
        # Add navigation
        total_steps = len(booklet_data['steps'])
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("⬅️ Previous"):
                if 'current_step' not in st.session_state:
                    st.session_state.current_step = 0
                st.session_state.current_step = max(0, st.session_state.current_step - 1)
        
        with col2:
            if 'current_step' not in st.session_state:
                st.session_state.current_step = 0
            
            step_number = st.slider(
                "Current Step",
                min_value=1,
                max_value=total_steps,
                value=st.session_state.current_step + 1,
                key="step_slider"
            )
            st.session_state.current_step = step_number - 1
        
        with col3:
            if st.button("➡️ Next"):
                if 'current_step' not in st.session_state:
                    st.session_state.current_step = 0
                st.session_state.current_step = min(total_steps - 1, st.session_state.current_step + 1)
        
        # Display current step
        if st.session_state.current_step < total_steps:
            step = booklet_data['steps'][st.session_state.current_step]
            
            st.subheader(f"Step {step['step_number']}: {step.get('instruction', '')[:50]}...")
            
            # Create two columns for text and visualization
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Instruction")
                st.write(step['instruction'])
                
                # Show grid info if available
                if step.get('has_grid', False):
                    st.info(f"Grid shape: {step.get('grid_shape', 'Unknown')}")
            
            with col2:
                st.markdown("### Visualization")
                
                # First try to use the image_path from the booklet data
                if step.get('has_image', False) and step.get('image_path'):
                    img_path = Path(step['image_path'])
                    if img_path.exists():
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True)
                        st.caption(f"Image: {img_path.name}")
                    else:
                        st.warning(f"Image file not found: {img_path}")
                elif step.get('has_grid', False):
                    # If we have grid data but no image, try to find it
                    task_name = booklet_data['task_name']
                    img_dir = Path("img_tmp")
                    
                    if img_dir.exists():
                        step_num = step['step_number']
                        
                        # Try multiple patterns to find the image
                        patterns = [
                            f"{task_name}_step_{step_num:03d}_*.png",
                            f"{task_name}_*_{step_num:03d}.png",
                            f"step_{step_num:03d}.png",
                            f"{task_name}_step_{step_num:03d}.png",
                            f"{task_name}_train*_{step_num:03d}.png",
                            f"{task_name}_test*_{step_num:03d}.png"
                        ]
                        
                        found_image = False
                        for pattern in patterns:
                            matching_files = list(img_dir.glob(pattern))
                            if matching_files:
                                img_path = matching_files[0]
                                img = Image.open(img_path)
                                st.image(img, use_container_width=True)
                                st.caption(f"Found: {img_path.name}")
                                found_image = True
                                break
                        
                        if not found_image:
                            st.info("No visualization available for this step")
                            st.caption(f"Looked for patterns: {patterns[:3]}...")
                    else:
                        st.warning("Image directory not found")
                else:
                    st.info("No grid data for this step")
    
    # TAB 2: Overview
    with tab2:
        st.header("📊 Solving Process Overview")
        
        # Count different types of steps
        iterations = []
        instruction_steps = []
        prediction_steps = []
        error_steps = []
        
        for step in booklet_data['steps']:
            instruction = step.get('instruction', '')
            if 'ITERATION' in instruction:
                iterations.append(step)
            if 'Instructions' in instruction or 'INSTRUCTIONS' in instruction or 'Step-by-Step' in instruction:
                instruction_steps.append(step)
            if 'Prediction' in instruction or 'PREDICTION' in instruction:
                prediction_steps.append(step)
            if '❌' in instruction or 'INCORRECT' in instruction:
                error_steps.append(step)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Iterations", len(iterations))
        with col2:
            st.metric("Predictions Made", len(prediction_steps))
        with col3:
            st.metric("Errors Detected", len(error_steps))
        
        st.subheader("Process Timeline")
        for i, step in enumerate(iterations):
            with st.expander(f"🔄 {step['instruction']}", expanded=False):
                # Find related steps for this iteration
                start_idx = step['step_number'] - 1
                end_idx = min(start_idx + 20, len(booklet_data['steps']))
                
                related_steps = booklet_data['steps'][start_idx:end_idx]
                for s in related_steps[:5]:  # Show first 5 steps of iteration
                    st.write(f"**Step {s['step_number']}:** {s['instruction'][:100]}...")
        
        st.subheader("All Steps Summary")
        step_df_data = []
        for step in booklet_data['steps']:
            step_df_data.append({
                'Step': step['step_number'],
                'Has Grid': '✅' if step.get('has_grid') else '❌',
                'Has Image': '✅' if step.get('has_image') else '❌',
                'Instruction': step['instruction'][:80] + '...' if len(step['instruction']) > 80 else step['instruction']
            })
        
        st.dataframe(step_df_data, use_container_width=True)
    
    # TAB 3: Final Prediction
    with tab3:
        st.header("🎯 Final Prediction Comparison")
        
        # Find prediction and expected output images
        task_name = booklet_data['task_name']
        img_dir = Path("img_tmp")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Test Input")
            if task_data and task_data.get('test'):
                test_input = task_data['test'][0]['input']
                input_img = visualize_grid(test_input)
                if input_img:
                    st.image(input_img, use_container_width=True)
                    st.caption(f"Shape: {len(test_input)} x {len(test_input[0])}")
            else:
                # Try to find test input image
                if img_dir.exists():
                    test_imgs = list(img_dir.glob(f"{task_name}_test_input_*.png"))
                    if test_imgs:
                        st.image(Image.open(test_imgs[0]), use_container_width=True)
        
        with col2:
            st.subheader("🤖 Predicted Output")
            if booklet_data.get('final_prediction_shape'):
                st.info(f"Shape: {booklet_data['final_prediction_shape']}")
                # Try to find prediction image
                if img_dir.exists():
                    pred_imgs = list(img_dir.glob(f"{task_name}*prediction*.png"))
                    if pred_imgs:
                        st.image(Image.open(pred_imgs[-1]), use_container_width=True)
                    else:
                        st.warning("Prediction image not found")
            else:
                st.info("No prediction available")
        
        with col3:
            st.subheader("✅ Expected Output")
            if task_data and task_data.get('test') and task_data['test'][0].get('output'):
                actual_output = task_data['test'][0]['output']
                output_img = visualize_grid(actual_output)
                if output_img:
                    st.image(output_img, use_container_width=True)
                    st.caption(f"Shape: {len(actual_output)} x {len(actual_output[0])}")
            elif booklet_data.get('actual_output_shape'):
                st.info(f"Shape: {booklet_data['actual_output_shape']}")
                # Try to find expected output image
                if img_dir.exists():
                    output_imgs = list(img_dir.glob(f"{task_name}_test_output_*.png"))
                    if output_imgs:
                        st.image(Image.open(output_imgs[0]), use_container_width=True)
            else:
                st.info("No reference output available")
        
        # Display accuracy details
        if booklet_data.get('accuracy') is not None:
            accuracy = booklet_data['accuracy']
            if accuracy == 1.0:
                st.success("✅ The prediction is CORRECT!")
            else:
                st.error("❌ The prediction does NOT match the expected output.")
        
        # Show all prediction attempts
        st.subheader("All Prediction Attempts")
        prediction_steps = [s for s in booklet_data['steps'] if 'Prediction' in s.get('instruction', '')]
        
        for pred_step in prediction_steps:
            with st.expander(f"Step {pred_step['step_number']}: {pred_step['instruction']}", expanded=False):
                st.write(pred_step['instruction'])
                if pred_step.get('has_image') and pred_step.get('image_path'):
                    img_path = Path(pred_step['image_path'])
                    if img_path.exists():
                        st.image(Image.open(img_path), width=300)
    
    # TAB 4: Training Data
    with tab4:
        st.header("📚 Training Examples")
        
        if task_data and task_data.get('train'):
            for i, example in enumerate(task_data['train']):
                st.subheader(f"Training Example {i+1}")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Input**")
                    input_img = visualize_grid(example['input'])
                    if input_img:
                        st.image(input_img, use_container_width=True)
                    st.caption(f"Shape: {len(example['input'])} x {len(example['input'][0])}")
                
                with col2:
                    st.write("**Output**")
                    output_img = visualize_grid(example['output'])
                    if output_img:
                        st.image(output_img, use_container_width=True)
                    st.caption(f"Shape: {len(example['output'])} x {len(example['output'][0])}")
                
                st.divider()
        else:
            st.warning("Original task data not found. Cannot display training examples.")
            st.info("Training example images may be in img_tmp/ directory")
    
    # TAB 5: Generated Instructions
    with tab5:
        st.header("🧠 AI-Generated Step-by-Step Instructions")
        
        instruction_steps = extract_instruction_text(booklet_data['steps'])
        
        if instruction_steps:
            st.info(f"Found {len(instruction_steps)} instruction generation/update steps")
            
            for i, inst_step in enumerate(instruction_steps):
                # Determine if this is initial instructions or an update
                instruction = inst_step['instruction']
                if i == 0:
                    icon = "🔍"
                    title = "Initial Step-by-Step Instructions (from first training example)"
                else:
                    icon = "🔄"
                    title = f"Updated Step-by-Step Instructions {i} (after {i+1} examples)"
                
                with st.expander(f"{icon} {title}", expanded=(i == len(instruction_steps) - 1)):
                    st.markdown(f"**Step {inst_step['step_number']}: {instruction}**")
                    
                    # Find the next step which usually contains the actual instructions
                    next_step_idx = inst_step['step_number']
                    if next_step_idx < len(booklet_data['steps']):
                        instructions_content = booklet_data['steps'][next_step_idx]['instruction']
                        st.markdown(instructions_content)
        else:
            st.warning("No instruction generation steps found in the booklet.")
    # Sidebar: Additional info and controls
    st.sidebar.divider()
    st.sidebar.header("📊 Statistics")
    
    # Calculate stats
    total_images = len([s for s in booklet_data['steps'] if s.get('has_image')])
    total_grids = len([s for s in booklet_data['steps'] if s.get('has_grid')])
    
    st.sidebar.metric("Steps with Images", total_images)
    st.sidebar.metric("Steps with Grids", total_grids)
    
    # Debug section - show available images
    with st.sidebar.expander("🔍 Debug: Image Files"):
        img_dir = Path("img_tmp")
        if img_dir.exists():
            task_images = list(img_dir.glob(f"{booklet_data['task_name']}*.png"))
            all_images = list(img_dir.glob("*.png"))
            st.write(f"Task images: {len(task_images)}")
            st.write(f"Total images: {len(all_images)}")
            
            if st.checkbox("Show all task images"):
                for img in sorted(task_images):
                    st.write(f"- {img.name}")
        else:
            st.write("img_tmp/ directory not found")
    
    # File info
    with st.sidebar.expander("📄 File Info"):
        st.write(f"**Booklet file:** {Path(selected_file).name}")
        st.write(f"**Task ID:** {booklet_data['task_name']}")
        if os.path.exists(selected_file):
            file_size = os.path.getsize(selected_file)
            st.write(f"**File size:** {file_size:,} bytes")
    
    # Footer
    st.divider()
    st.markdown("---")
    st.markdown(
        """
        **ARC Booklet Solver** | Viewing instructions generated by the meta-instruction algorithm
        """
    )

if __name__ == "__main__":
    main()
