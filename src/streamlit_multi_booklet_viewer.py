#!/usr/bin/env python3
"""
Streamlit Multi-Booklet Viewer
Comprehensive viewer for refined and ensemble booklets
"""

import streamlit as st
import json
from pathlib import Path
from PIL import Image
import pandas as pd

# Configure page
st.set_page_config(
    page_title="ARC Multi-Booklet Viewer",
    page_icon="📚",
    layout="wide"
)

def find_booklets():
    """Find all refined and ensemble booklets"""
    booklets = {
        'refined': [],
        'ensemble': [],
        'single': []
    }
    
    # Find refined booklets
    refined_dir = Path("refined_booklets")
    if refined_dir.exists():
        for item in refined_dir.iterdir():
            if item.is_dir() and (item / "refinement_meta.json").exists():
                booklets['refined'].append(str(item))
    
    # Find ensemble booklets
    ensemble_dir = Path("ensemble_booklets")
    if ensemble_dir.exists():
        for item in ensemble_dir.iterdir():
            if item.is_dir() and (item / "synthesis_meta.json").exists():
                booklets['ensemble'].append(str(item))
    
    # Find single booklets
    single_dir = Path("sample_booklets")
    if single_dir.exists():
        for item in single_dir.iterdir():
            if item.is_dir() and (item / "metadata.json").exists():
                booklets['single'].append(str(item))
    
    return booklets

def load_refinement_meta(booklet_dir):
    """Load refinement metadata"""
    meta_path = Path(booklet_dir) / "refinement_meta.json"
    with open(meta_path, 'r') as f:
        return json.load(f)

def load_synthesis_meta(booklet_dir):
    """Load synthesis metadata"""
    meta_path = Path(booklet_dir) / "synthesis_meta.json"
    with open(meta_path, 'r') as f:
        return json.load(f)

def load_booklet_meta(booklet_dir):
    """Load single booklet metadata"""
    meta_path = Path(booklet_dir) / "metadata.json"
    with open(meta_path, 'r') as f:
        return json.load(f)

def main():
    st.title("📚 ARC Multi-Booklet Viewer")
    st.markdown("Comprehensive viewer for refined, ensemble, and single booklets")
    
    # Find all booklets
    booklets = find_booklets()
    
    total_booklets = sum(len(v) for v in booklets.values())
    
    if total_booklets == 0:
        st.error("No booklets found. Generate booklets first.")
        st.code("""
# Single booklet
python arc-booklet-generator.py <task_file>

# Refined booklet
python arc-booklet-refiner.py <task_file>

# Ensemble booklet
python arc-booklet-ensemble.py <task_file>
        """)
        return
    
    # Sidebar - Booklet selection
    st.sidebar.header("📁 Select Booklet")
    
    # Show counts
    st.sidebar.metric("Refined Booklets", len(booklets['refined']))
    st.sidebar.metric("Ensemble Booklets", len(booklets['ensemble']))
    st.sidebar.metric("Single Booklets", len(booklets['single']))
    
    st.sidebar.divider()
    
    # Booklet type selection
    booklet_type = st.sidebar.radio(
        "Booklet Type:",
        ["Refined (Iterative)", "Ensemble (Multi-Booklet)", "Single Example"],
        index=0 if booklets['refined'] else (1 if booklets['ensemble'] else 2)
    )
    
    # Select specific booklet
    if booklet_type == "Refined (Iterative)":
        if not booklets['refined']:
            st.warning("No refined booklets found. Generate one with arc-booklet-refiner.py")
            return
        
        selected = st.sidebar.selectbox(
            "Select Refined Booklet:",
            booklets['refined'],
            format_func=lambda x: Path(x).name
        )
        
        if selected:
            show_refined_booklet(selected)
    
    elif booklet_type == "Ensemble (Multi-Booklet)":
        if not booklets['ensemble']:
            st.warning("No ensemble booklets found. Generate one with arc-booklet-ensemble.py")
            return
        
        selected = st.sidebar.selectbox(
            "Select Ensemble Booklet:",
            booklets['ensemble'],
            format_func=lambda x: Path(x).name
        )
        
        if selected:
            show_ensemble_booklet(selected)
    
    else:  # Single
        if not booklets['single']:
            st.warning("No single booklets found. Generate one with arc-booklet-generator.py")
            return
        
        selected = st.sidebar.selectbox(
            "Select Single Booklet:",
            booklets['single'],
            format_func=lambda x: Path(x).name
        )
        
        if selected:
            show_single_booklet(selected)

def show_refined_booklet(booklet_dir):
    """Display refined booklet with iteration history"""
    meta = load_refinement_meta(booklet_dir)
    
    st.header(f"🔄 Refined Booklet: {meta['task_name']}")
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Training Examples", meta['total_training_examples'])
    with col2:
        st.metric("Training Success", f"{meta.get('successful_training_tests', 0)}/{meta['total_training_examples']}")
    with col3:
        st.metric("Test Cases Solved", f"{meta.get('test_success_count', 0)}/{meta.get('total_test_cases', 0)}")
    with col4:
        st.metric("Final Steps", len(meta.get('final_steps', [])))
    with col5:
        st.metric("Refinements", meta.get('total_refinements', 0))
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Refinement Evolution", "📋 Final Steps", "📊 Statistics"])
    
    # TAB 1: Refinement Evolution
    with tab1:
        st.subheader("How Steps Evolved Across Training Examples")
        
        for entry in meta['refinement_history']:
            action_emoji = {
                'initial_generation': '🌱',
                'tested_success': '✅',
                'refined_after_failure': '🔧'
            }.get(entry['action'], '📝')
            
            action_label = entry['action'].replace('_', ' ').title()
            
            with st.expander(
                f"{action_emoji} Example {entry['example_number']} - {action_label} - {len(entry['steps'])} steps",
                expanded=(entry['example_number'] == 1)
            ):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("**Steps:**")
                    for i, step in enumerate(entry['steps'], 1):
                        st.write(f"{i}. {step}")
                
                with col2:
                    st.metric("Total Steps", len(entry['steps']))
                    st.write(f"**Action:** {action_label}")
                    st.write(f"**Success:** {'✅ Yes' if entry['success'] else '❌ No'}")
        
        # Show step changes timeline
        st.subheader("Step Count Timeline")
        timeline_data = {
            'Example': [f"Ex {h['example_number']}" for h in meta['refinement_history']],
            'Total Steps': [len(h['steps']) for h in meta['refinement_history']]
        }
        st.line_chart(pd.DataFrame(timeline_data).set_index('Example'))
    
    # TAB 2: Final Steps
    with tab2:
        st.subheader("Final Refined Steps")
        st.info("These are the steps after processing all training examples")
        
        if meta.get('final_steps'):
            for i, step in enumerate(meta['final_steps'], 1):
                st.write(f"**{i}.** {step}")
        else:
            st.warning("No final steps recorded")
    
    # TAB 3: Statistics
    with tab3:
        st.subheader("Refinement Statistics")
        
        # Test Results Section
        if meta.get('test_results'):
            st.write("**Test Case Results:**")
            test_data = []
            for test in meta['test_results']:
                test_data.append({
                    'Test': f"Test {test['test_number']}",
                    'Result': '✅ Success' if test['success'] else '❌ Failed',
                    'Attempts': test['attempts']
                })
            st.dataframe(pd.DataFrame(test_data), use_container_width=True)
            st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Success Rate by Example:**")
            success_data = []
            for h in meta['refinement_history']:
                success_data.append({
                    'Example': f"Ex {h['example_number']}",
                    'Action': h['action'].replace('_', ' ').title(),
                    'Success': 'Yes' if h['success'] else 'No',
                    'Steps': len(h['steps'])
                })
            st.dataframe(pd.DataFrame(success_data), use_container_width=True)
        
        with col2:
            st.write("**Step Evolution:**")
            evolution_data = []
            prev_steps = None
            for h in meta['refinement_history']:
                changed = "Initial" if prev_steps is None else ("Changed" if h['steps'] != prev_steps else "Stable")
                evolution_data.append({
                    'Example': h['example_number'],
                    'Status': changed,
                    'Step Count': len(h['steps'])
                })
                prev_steps = h['steps']
            st.dataframe(pd.DataFrame(evolution_data), use_container_width=True)

def show_ensemble_booklet(booklet_dir):
    """Display ensemble booklet with multiple perspectives"""
    meta = load_synthesis_meta(booklet_dir)
    booklet_path = Path(booklet_dir)
    
    st.header(f"🎭 Ensemble Booklet: {meta['task_name']}")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Booklets", meta['total_booklets_generated'])
    with col2:
        successful = sum(1 for b in meta['booklets'] if b.get('success', False))
        st.metric("Successful", f"{successful}/{meta['total_booklets_generated']}")
    with col3:
        st.metric("Common Steps", len(meta.get('common_steps', [])))
    with col4:
        avg_steps = sum(b['total_steps'] for b in meta['booklets']) / len(meta['booklets'])
        st.metric("Avg Steps", f"{avg_steps:.1f}")
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Common Pattern", "📚 All Booklets", "🔍 Individual View", "📊 Comparison"])
    
    # TAB 1: Common Pattern
    with tab1:
        st.subheader("Common Steps Across All Training Examples")
        
        if meta.get('common_steps'):
            st.success(f"Found {len(meta['common_steps'])} common steps that appear in all booklets!")
            
            for i, step in enumerate(meta['common_steps'], 1):
                st.write(f"**{i}.** {step}")
        else:
            st.warning("No common steps found across all booklets")
            st.info("This suggests the puzzle may have context-dependent variations")
    
    # TAB 2: All Booklets
    with tab2:
        st.subheader("All Individual Booklets")
        
        for booklet in meta['booklets']:
            success_emoji = "✅" if booklet['success'] else "❌"
            
            with st.expander(
                f"{success_emoji} Example {booklet['example_number']} - {booklet['total_steps']} steps",
                expanded=False
            ):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write("**Steps:**")
                    for i, step in enumerate(booklet['steps'], 1):
                        st.write(f"{i}. {step}")
                
                with col2:
                    st.metric("Total Steps", booklet['total_steps'])
                    st.metric("Reached Target", f"{booklet['steps_reaching_target']}/{booklet['total_steps']}")
                
                with col3:
                    st.write(f"**Success:** {success_emoji}")
                    st.write(f"**Path:** `{booklet['booklet_path']}`")
                    
                    # Link to individual booklet
                    booklet_full_path = booklet_path / booklet['booklet_path']
                    if booklet_full_path.exists():
                        st.caption(f"View in file explorer: {booklet_full_path}")
    
    # TAB 3: Individual View
    with tab3:
        st.subheader("View Individual Booklet Details")
        
        # Select which booklet to view in detail
        booklet_options = [
            f"Example {b['example_number']}: {b['total_steps']} steps {'✅' if b['success'] else '❌'}"
            for b in meta['booklets']
        ]
        
        selected_idx = st.selectbox(
            "Select booklet to view:",
            range(len(meta['booklets'])),
            format_func=lambda i: booklet_options[i]
        )
        
        if selected_idx is not None:
            selected_booklet = meta['booklets'][selected_idx]
            booklet_full_path = booklet_path / selected_booklet['booklet_path']
            
            if (booklet_full_path / "metadata.json").exists():
                booklet_data = load_booklet_meta(booklet_full_path)
                
                st.write(f"**Example {selected_booklet['example_number']} Details**")
                
                # Show images
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Input**")
                    input_img = booklet_full_path / "input.png"
                    if input_img.exists():
                        st.image(str(input_img), use_container_width=True)
                
                with col2:
                    st.write("**Final Output**")
                    if booklet_data['steps']:
                        final_step = booklet_data['steps'][-1]
                        final_img = booklet_full_path / f"step_{final_step['step_number']:03d}.png"
                        if final_img.exists():
                            st.image(str(final_img), use_container_width=True)
                
                with col3:
                    st.write("**Expected Output**")
                    target_img = booklet_full_path / "target_output.png"
                    if target_img.exists():
                        st.image(str(target_img), use_container_width=True)
                
                # Show all steps
                st.divider()
                st.write("**All Steps:**")
                
                for step in booklet_data['steps']:
                    with st.expander(f"Step {step['step_number']}: {step['description'][:60]}...", expanded=False):
                        step_img = booklet_full_path / f"step_{step['step_number']:03d}.png"
                        if step_img.exists():
                            st.image(str(step_img), width=300)
                        st.write(f"**Description:** {step['description']}")
                        st.write(f"**Tries:** {step.get('tries', 1)}")
                        st.write(f"**Reached Target:** {'✅' if step.get('reached_target') else '❌'}")
    
    # TAB 4: Comparison
    with tab4:
        st.subheader("Compare All Booklets")
        
        # Create comparison table
        comparison_data = []
        
        for booklet in meta['booklets']:
            comparison_data.append({
                'Example': booklet['example_number'],
                'Steps': booklet['total_steps'],
                'Reached Target': f"{booklet['steps_reaching_target']}/{booklet['total_steps']}",
                'Success Rate': f"{booklet['steps_reaching_target']/booklet['total_steps']*100:.0f}%" if booklet['total_steps'] > 0 else "0%",
                'Success': '✅' if booklet['success'] else '❌'
            })
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        # Step count distribution
        st.subheader("Step Count Distribution")
        step_counts = [b['total_steps'] for b in meta['booklets']]
        st.bar_chart(pd.DataFrame({
            'Example': [f"Ex {b['example_number']}" for b in meta['booklets']],
            'Steps': step_counts
        }).set_index('Example'))
        
        # Success rate chart
        st.subheader("Success Rate by Example")
        success_rates = [
            (b['steps_reaching_target'] / b['total_steps'] * 100) if b['total_steps'] > 0 else 0
            for b in meta['booklets']
        ]
        st.bar_chart(pd.DataFrame({
            'Example': [f"Ex {b['example_number']}" for b in meta['booklets']],
            'Success %': success_rates
        }).set_index('Example'))

def show_single_booklet(booklet_dir):
    """Display single booklet"""
    booklet_data = load_booklet_meta(booklet_dir)
    booklet_path = Path(booklet_dir)
    
    st.header(f"📖 Single Booklet: {booklet_data['task_name']}")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Steps", booklet_data['total_steps'])
    with col2:
        reached = sum(1 for s in booklet_data['steps'] if s.get('reached_target', False))
        st.metric("Steps Reaching Target", f"{reached}/{booklet_data['total_steps']}")
    with col3:
        st.metric("Success Rate", f"{reached/booklet_data['total_steps']*100:.0f}%" if booklet_data['total_steps'] > 0 else "0%")
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Comparison Table", "🔍 Step Details", "📚 Input/Output"])
    
    # TAB 1: Comparison Table
    with tab1:
        st.header("Model vs Expected Comparison")
        
        table_data = []
        
        for step in booklet_data['steps']:
            step_num = step['step_number']
            expected_img_path = booklet_path / f"step_{step_num:03d}_expected.png"
            
            table_data.append({
                'Step': step_num,
                'Description': step['description'][:60] + '...' if len(step['description']) > 60 else step['description'],
                'Tries': step.get('tries', 1),
                'Reached Target': '✅' if step.get('reached_target', False) else '❌',
                'Has Expected Image': '✅' if expected_img_path.exists() else '➖'
            })
        
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
        
        # Visual comparison
        st.subheader("Visual Comparison")
        
        step_to_view = st.selectbox(
            "Select step:",
            range(len(booklet_data['steps'])),
            format_func=lambda x: f"Step {x}: {booklet_data['steps'][x]['description'][:40]}..."
        )
        
        if step_to_view is not None:
            step = booklet_data['steps'][step_to_view]
            
            cols = st.columns(3)
            
            with cols[0]:
                st.write("**Model Output**")
                model_img = booklet_path / f"step_{step['step_number']:03d}.png"
                if model_img.exists():
                    st.image(str(model_img), use_container_width=True)
            
            with cols[1]:
                st.write("**Expected**")
                expected_img = booklet_path / f"step_{step['step_number']:03d}_expected.png"
                if expected_img.exists():
                    st.image(str(expected_img), use_container_width=True)
                else:
                    st.success("Model matched target!")
            
            with cols[2]:
                st.write("**Step Info**")
                st.write(f"**Description:** {step['description']}")
                st.write(f"**Tries:** {step.get('tries', 1)}")
                st.write(f"**Shape:** {step.get('grid_shape', 'N/A')}")
                st.write(f"**Reached Target:** {'✅' if step.get('reached_target') else '❌'}")
    
    # TAB 2: Step Details
    with tab2:
        st.header("All Steps")
        
        for step in booklet_data['steps']:
            with st.expander(f"Step {step['step_number']}: {step['description'][:60]}...", expanded=False):
                cols = st.columns([2, 1])
                
                with cols[0]:
                    st.write(f"**Full Description:** {step['description']}")
                    st.write(f"**Grid Shape:** {step.get('grid_shape', 'N/A')}")
                    st.write(f"**Tries:** {step.get('tries', 1)}")
                    st.write(f"**Reached Target:** {'✅' if step.get('reached_target') else '❌'}")
                
                with cols[1]:
                    model_img = booklet_path / f"step_{step['step_number']:03d}.png"
                    if model_img.exists():
                        st.image(str(model_img), caption="Model", use_container_width=True)
                    
                    expected_img = booklet_path / f"step_{step['step_number']:03d}_expected.png"
                    if expected_img.exists():
                        st.image(str(expected_img), caption="Expected", use_container_width=True)
    
    # TAB 3: Input/Output
    with tab3:
        st.header("Puzzle Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input")
            input_img = booklet_path / "input.png"
            if input_img.exists():
                st.image(str(input_img), use_container_width=True)
        
        with col2:
            st.subheader("Target Output")
            output_img = booklet_path / "target_output.png"
            if output_img.exists():
                st.image(str(output_img), use_container_width=True)
        
        # Final comparison
        st.subheader("Final Result")
        if booklet_data['steps']:
            final_step = booklet_data['steps'][-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Model Final Output**")
                final_img = booklet_path / f"step_{final_step['step_number']:03d}.png"
                if final_img.exists():
                    st.image(str(final_img), use_container_width=True)
            
            with col2:
                st.write("**Expected Output**")
                target_img = booklet_path / "target_output.png"
                if target_img.exists():
                    st.image(str(target_img), use_container_width=True)
            
            with col3:
                st.write("**Result**")
                if final_step.get('reached_target'):
                    st.success("✅ SUCCESS!")
                    st.write("Model output matches expected")
                else:
                    st.error("❌ MISMATCH")
                    st.write(f"Attempts: {final_step.get('tries', 1)}")
                    st.write(f"Model shape: {final_step.get('grid_shape')}")
                    st.write(f"Target shape: {final_step.get('target_shape')}")

if __name__ == "__main__":
    main()

