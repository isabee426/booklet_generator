#!/usr/bin/env python3
"""
Comprehensive ARC Booklet Viewer
Organized view of all booklet types with comparison capabilities
"""

import streamlit as st
import json
from pathlib import Path
from PIL import Image
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="ARC Comprehensive Booklet Viewer",
    page_icon="📚",
    layout="wide"
)

def find_all_booklets():
    """Find and categorize all booklets"""
    booklets = {
        'batch_visual': [],
        'single': [],
        'refined': [],
        'ensemble': []
    }
    
    # Batch visual booklets
    batch_dir = Path("batch_visual_booklets")
    if batch_dir.exists():
        for run_dir in batch_dir.iterdir():
            if run_dir.is_dir() and (run_dir / "batch_results.json").exists():
                # Extract timestamp from folder name
                parts = run_dir.name.split('_batch_')
                timestamp_str = parts[1] if len(parts) > 1 else "unknown"
                
                with open(run_dir / "batch_results.json", 'r') as f:
                    meta = json.load(f)
                
                # Find all example booklets
                example_booklets = []
                for item in run_dir.iterdir():
                    if item.is_dir() and (item / "metadata.json").exists():
                        example_booklets.append(str(item))
                
                booklets['batch_visual'].append({
                    'path': str(run_dir),
                    'task_name': meta['task_name'],
                    'timestamp': timestamp_str,
                    'training_success': meta['training_success'],
                    'training_total': meta['training_examples'],
                    'test_success': meta.get('test_success', 0),
                    'test_total': meta.get('total_test_cases', 0),
                    'refinements': meta.get('refinement_iterations', 0),
                    'example_booklets': example_booklets,
                    'approach': 'Batch Visual'
                })
    
    # Single booklets
    single_dir = Path("sample_booklets")
    if single_dir.exists():
        for item in single_dir.iterdir():
            if item.is_dir() and (item / "metadata.json").exists():
                with open(item / "metadata.json", 'r') as f:
                    meta = json.load(f)
                
                booklets['single'].append({
                    'path': str(item),
                    'task_name': meta['task_name'],
                    'timestamp': meta.get('generated_at', 'unknown'),
                    'total_steps': meta['total_steps'],
                    'approach': 'Single Example'
                })
    
    # Refined booklets
    refined_dir = Path("refined_booklets")
    if refined_dir.exists():
        for item in refined_dir.iterdir():
            if item.is_dir() and (item / "refinement_meta.json").exists():
                with open(item / "refinement_meta.json", 'r') as f:
                    meta = json.load(f)
                
                booklets['refined'].append({
                    'path': str(item),
                    'task_name': meta['task_name'],
                    'timestamp': meta.get('generated_at', 'unknown'),
                    'training_total': meta['total_training_examples'],
                    'refinements': meta.get('total_refinements', 0),
                    'approach': 'Iterative Refiner'
                })
    
    return booklets

def load_booklet_metadata(booklet_path):
    """Load metadata for a booklet"""
    meta_path = Path(booklet_path) / "metadata.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None

def show_booklet_comparison_table(booklet_path):
    """Show step-by-step comparison table"""
    meta = load_booklet_metadata(booklet_path)
    if not meta:
        st.error("Could not load booklet metadata")
        return
    
    booklet_dir = Path(booklet_path)
    
    # Header
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Task", meta['task_name'])
    with col2:
        st.metric("Total Steps", meta['total_steps'])
    with col3:
        reached = sum(1 for s in meta['steps'] if s.get('reached_target', False))
        st.metric("Steps Correct", f"{reached}/{meta['total_steps']}")
    with col4:
        st.metric("Final Success", "✅" if meta.get('success', False) else "❌")
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Step Comparison", "🔍 Step Details", "📝 Inputs/Outputs"])
    
    with tab1:
        st.subheader("Step-by-Step Comparison")
        
        # Table
        table_data = []
        for step in meta['steps']:
            step_num = step['step_number']
            table_data.append({
                'Step': step_num + 1,
                'Description': step['description'][:60] + '...' if len(step['description']) > 60 else step['description'],
                'Reached Target': '✅' if step.get('reached_target', False) else '❌',
                'Tries': step.get('tries', 1)
            })
        
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
        
        # Visual comparison
        st.subheader("Visual Comparison")
        step_to_view = st.selectbox("Select step:", range(len(meta['steps'])), 
                                     format_func=lambda x: f"Step {x+1}")
        
        if step_to_view is not None:
            step = meta['steps'][step_to_view]
            step_num = step['step_number']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Model Output**")
                model_img = booklet_dir / f"step_{step_num:03d}.png"
                if model_img.exists():
                    st.image(str(model_img), use_column_width=True)
                else:
                    st.warning("Image not found")
            
            with col2:
                st.write("**Expected Output**")
                expected_img = booklet_dir / f"step_{step_num:03d}_expected.png"
                if expected_img.exists():
                    st.image(str(expected_img), use_column_width=True)
                else:
                    st.success("✅ Model matched target!")
            
            with col3:
                st.write("**Step Info**")
                st.write(f"**Description:**")
                st.write(step['description'])
                st.write(f"**Reached Target:** {'✅' if step.get('reached_target') else '❌'}")
    
    with tab2:
        st.header("All Steps Expanded")
        for i, step in enumerate(meta['steps']):
            with st.expander(f"Step {i+1}: {step['description'][:50]}...", expanded=(i==0)):
                st.write(f"**Full Description:**")
                st.write(step['description'])
                
                col1, col2 = st.columns(2)
                with col1:
                    model_img = booklet_dir / f"step_{step['step_number']:03d}.png"
                    if model_img.exists():
                        st.image(str(model_img), caption="Model", use_column_width=True)
                
                with col2:
                    expected_img = booklet_dir / f"step_{step['step_number']:03d}_expected.png"
                    if expected_img.exists():
                        st.image(str(expected_img), caption="Expected", use_column_width=True)
    
    with tab3:
        st.header("Input and Target Output")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input")
            input_img = booklet_dir / "input.png"
            if input_img.exists():
                st.image(str(input_img), use_column_width=True)
        
        with col2:
            st.subheader("Target Output")
            target_img = booklet_dir / "target_output.png"
            if target_img.exists():
                st.image(str(target_img), use_column_width=True)

def main():
    st.title("📚 Comprehensive ARC Booklet Viewer")
    st.markdown("All booklets organized by approach with comparison tools")
    
    # Find all booklets
    all_booklets = find_all_booklets()
    
    total = sum(len(v) for v in all_booklets.values())
    
    if total == 0:
        st.error("No booklets found. Run a generator first.")
        st.code("""
# Batch Visual (best for generalization)
python arc-booklet-batch-visual.py <task_file>

# Single Example
python arc-booklet-generator.py <task_file>

# Iterative Refiner
python arc-booklet-refiner.py <task_file>
        """)
        return
    
    # Sidebar - Approach selection
    st.sidebar.header("📁 Booklet Organization")
    
    # Show counts
    st.sidebar.metric("Batch Visual Runs", len(all_booklets['batch_visual']))
    st.sidebar.metric("Single Booklets", len(all_booklets['single']))
    st.sidebar.metric("Refined Booklets", len(all_booklets['refined']))
    st.sidebar.metric("Ensemble Booklets", len(all_booklets['ensemble']))
    
    st.sidebar.divider()
    
    # Select approach
    approach_options = []
    if all_booklets['batch_visual']:
        approach_options.append("Batch Visual")
    if all_booklets['single']:
        approach_options.append("Single Example")
    if all_booklets['refined']:
        approach_options.append("Iterative Refiner")
    if all_booklets['ensemble']:
        approach_options.append("Ensemble")
    
    if not approach_options:
        st.warning("No booklets found")
        return
    
    selected_approach = st.sidebar.selectbox("Select Approach:", approach_options)
    
    # Show booklets for selected approach
    if selected_approach == "Batch Visual":
        show_batch_visual_booklets(all_booklets['batch_visual'])
    elif selected_approach == "Single Example":
        show_single_booklets(all_booklets['single'])
    elif selected_approach == "Iterative Refiner":
        show_refined_booklets(all_booklets['refined'])
    elif selected_approach == "Ensemble":
        show_ensemble_booklets(all_booklets['ensemble'])

def show_batch_visual_booklets(booklets):
    """Show batch visual booklets organized by task and run"""
    st.header("🎨 Batch Visual Booklets")
    st.markdown("*Structured reasoning across all training examples*")
    
    if not booklets:
        st.info("No batch visual booklets found")
        return
    
    # Group by task
    by_task = {}
    for b in booklets:
        task = b['task_name']
        if task not in by_task:
            by_task[task] = []
        by_task[task].append(b)
    
    # Sort runs by timestamp (newest first)
    for task in by_task:
        by_task[task] = sorted(by_task[task], key=lambda x: x['timestamp'], reverse=True)
    
    # Task selector
    task_names = list(by_task.keys())
    selected_task = st.selectbox("Select Task:", task_names)
    
    if selected_task:
        runs = by_task[selected_task]
        
        st.subheader(f"Task: {selected_task}")
        st.write(f"**{len(runs)} run(s) found**")
        
        # Show runs
        for run_idx, run in enumerate(runs):
            with st.expander(f"Run {run_idx + 1} - {run['timestamp']} - Train: {run['training_success']}/{run['training_total']}, Test: {run['test_success']}/{run['test_total']}", 
                           expanded=(run_idx == 0)):
                
                # Run stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Training Success", f"{run['training_success']}/{run['training_total']}")
                with col2:
                    st.metric("Test Success", f"{run['test_success']}/{run['test_total']}")
                with col3:
                    st.metric("Refinements", run['refinements'])
                with col4:
                    st.metric("Example Booklets", len(run['example_booklets']))
                
                st.divider()
                
                # Select example booklet to view
                if run['example_booklets']:
                    example_options = [Path(p).name for p in run['example_booklets']]
                    selected_example = st.selectbox(
                        f"Select example booklet (Run {run_idx + 1}):",
                        run['example_booklets'],
                        format_func=lambda x: Path(x).name,
                        key=f"example_{run_idx}"
                    )
                    
                    if selected_example:
                        st.divider()
                        show_booklet_comparison_table(selected_example)

def show_single_booklets(booklets):
    """Show single example booklets"""
    st.header("📖 Single Example Booklets")
    
    if not booklets:
        st.info("No single booklets found")
        return
    
    # Sort by timestamp
    sorted_booklets = sorted(booklets, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    selected = st.selectbox(
        "Select booklet:",
        sorted_booklets,
        format_func=lambda x: f"{x['task_name']} - {x['total_steps']} steps - {x['timestamp'][:19] if len(x['timestamp']) > 19 else x['timestamp']}"
    )
    
    if selected:
        show_booklet_comparison_table(selected['path'])

def show_refined_booklets(booklets):
    """Show refined booklets"""
    st.header("🔄 Iteratively Refined Booklets")
    
    if not booklets:
        st.info("No refined booklets found")
        return
    
    # Sort by timestamp
    sorted_booklets = sorted(booklets, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    selected = st.selectbox(
        "Select booklet:",
        sorted_booklets,
        format_func=lambda x: f"{x['task_name']} - {x['refinements']} refinements - {x['timestamp'][:19] if len(x['timestamp']) > 19 else x['timestamp']}"
    )
    
    if selected:
        st.write("Refined booklets don't have step-by-step visualizations yet")
        st.write(f"Path: {selected['path']}")

def show_ensemble_booklets(booklets):
    """Show ensemble booklets"""
    st.header("🎭 Ensemble Booklets")
    st.info("Ensemble booklets not fully implemented yet")

if __name__ == "__main__":
    main()

