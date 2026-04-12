#!/usr/bin/env python3
"""
Streamlit Booklet Viewer
Shows step-by-step booklets with model vs ideal comparison table
"""

import streamlit as st
import json
from pathlib import Path
from PIL import Image
import pandas as pd

# Configure page
st.set_page_config(
    page_title="ARC Booklet Viewer",
    page_icon="📚",
    layout="wide"
)

def find_booklets():
    """Find all booklet directories from multiple sources"""
    booklet_dirs = []
    
    # Search in multiple directories
    search_dirs = [
        "sample_booklets",
        "batch_visual_booklets",
        "refined_booklets",
        "ensemble_booklets"
    ]
    
    for directory in search_dirs:
        base_path = Path(directory)
        
        if base_path.exists():
            for item in base_path.iterdir():
                if item.is_dir():
                    # Look for booklet folders (have metadata.json)
                    if (item / "metadata.json").exists():
                        booklet_dirs.append(str(item))
                    # Also look for nested booklet folders (like batch visual)
                    else:
                        for nested in item.iterdir():
                            if nested.is_dir() and (nested / "metadata.json").exists():
                                booklet_dirs.append(str(nested))
    
    return sorted(booklet_dirs)

def load_booklet(booklet_dir):
    """Load booklet metadata"""
    metadata_path = Path(booklet_dir) / "metadata.json"
    with open(metadata_path, 'r') as f:
        return json.load(f)

def main():
    st.title("📚 ARC Booklet Viewer")
    st.markdown("Step-by-step instructional booklets with model vs ideal comparison")
    
    # Sidebar - Booklet selection
    st.sidebar.header("📁 Select Booklet")
    
    booklets = find_booklets()
    
    if not booklets:
        st.error("No booklets found. Generate booklets first using arc-booklet-generator.py")
        st.code("python arc-booklet-generator.py <task_file>")
        return
    
    selected_booklet = st.sidebar.selectbox(
        "Booklet:",
        booklets,
        format_func=lambda x: Path(x).name
    )
    
    if not selected_booklet:
        st.warning("Please select a booklet")
        return
    
    # Load booklet
    booklet_data = load_booklet(selected_booklet)
    booklet_dir = Path(selected_booklet)
    
    # Header
    st.header(f"Task: {booklet_data['task_name']}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Steps", booklet_data['total_steps'])
    with col2:
        reached_target = sum(1 for s in booklet_data['steps'] if s.get('reached_target', False))
        st.metric("Steps Reaching Target", f"{reached_target}/{booklet_data['total_steps']}")
    with col3:
        st.metric("Success Rate", f"{reached_target/booklet_data['total_steps']*100:.0f}%" if booklet_data['total_steps'] > 0 else "N/A")
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Comparison Table", "🔍 Step Details", "📚 Input/Output"])
    
    # TAB 1: Comparison Table
    with tab1:
        st.header("Model vs Ideal Comparison")
        
        # Create comparison table data
        table_data = []
        
        for step in booklet_data['steps']:
            step_num = step['step_number']
            
            # Check if images exist
            model_img_path = booklet_dir / f"step_{step_num:03d}.png"
            ideal_img_path = booklet_dir / f"step_{step_num:03d}_expected.png"
            
            table_data.append({
                'Step': step_num,
                'Description': step['description'][:50] + '...' if len(step['description']) > 50 else step['description'],
                'Tries': step.get('tries', 1),
                'Reached Target': '✅' if step.get('reached_target', False) else '❌',
                'Has Expected Image': '✅' if ideal_img_path.exists() else '➖'
            })
        
        # Display table
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
        
        st.subheader("Visual Comparison")
        
        # Select step to view
        step_to_view = st.selectbox(
            "Select step to view:",
            range(len(booklet_data['steps'])),
            format_func=lambda x: f"Step {x}: {booklet_data['steps'][x]['description'][:40]}..."
        )
        
        if step_to_view is not None:
            step = booklet_data['steps'][step_to_view]
            
            # Display images side by side
            cols = st.columns(3)
            
            with cols[0]:
                st.subheader("Model Output")
                model_img_path = booklet_dir / f"step_{step['step_number']:03d}.png"
                if model_img_path.exists():
                    st.image(str(model_img_path), use_container_width=True)
                    st.caption(f"Step {step['step_number']}")
                else:
                    st.info("Image not found")
            
            with cols[1]:
                st.subheader("Ideal/Target")
                ideal_img_path = booklet_dir / f"step_{step['step_number']:03d}_expected.png"
                if ideal_img_path.exists():
                    st.image(str(ideal_img_path), use_container_width=True)
                    st.caption("Target grid")
                else:
                    st.info("No separate ideal (model matched target)")
            
            with cols[2]:
                st.subheader("Step Info")
                st.write(f"**Description:**")
                st.write(step['description'])
                st.write(f"**Tries:** {step.get('tries', 1)}")
                st.write(f"**Reached Target:** {'✅ Yes' if step.get('reached_target') else '❌ No'}")
                st.write(f"**Grid Shape:** {step.get('grid_shape', 'N/A')}")
    
    # TAB 2: Step Details
    with tab2:
        st.header("Step-by-Step Details")
        
        for i, step in enumerate(booklet_data['steps']):
            with st.expander(f"Step {step['step_number']}: {step['description'][:60]}{'...' if len(step['description']) > 60 else ''}", 
                           expanded=(i == 0)):
                cols = st.columns([2, 1])
                
                with cols[0]:
                    st.write("**Full Description:**")
                    st.write(step['description'])
                    st.write(f"**Grid Shape:** {step.get('grid_shape', 'N/A')}")
                    st.write(f"**Tries:** {step.get('tries', 1)}")
                    st.write(f"**Reached Target:** {'✅ Yes' if step.get('reached_target') else '❌ No'}")
                
                with cols[1]:
                    model_img = booklet_dir / f"step_{step['step_number']:03d}.png"
                    if model_img.exists():
                        st.image(str(model_img), caption="Model Output", use_container_width=True)
                    
                    ideal_img = booklet_dir / f"step_{step['step_number']:03d}_expected.png"
                    if ideal_img.exists():
                        st.image(str(ideal_img), caption="Ideal Target", use_container_width=True)
    
    # TAB 3: Input/Output
    with tab3:
        st.header("Puzzle Input and Target Output")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input")
            input_img = booklet_dir / "input.png"
            if input_img.exists():
                st.image(str(input_img), use_container_width=True)
        
        with col2:
            st.subheader("Target Output")
            output_img = booklet_dir / "target_output.png"
            if output_img.exists():
                st.image(str(output_img), use_container_width=True)
        
        # Show final step for comparison
        st.subheader("Final Step Result")
        if booklet_data['steps']:
            final_step = booklet_data['steps'][-1]
            final_img = booklet_dir / f"step_{final_step['step_number']:03d}.png"
            if final_img.exists():
                st.image(str(final_img), caption="Model's Final Output", width=400)
                if final_step.get('reached_target'):
                    st.success("✅ Model reached the target output!")
                else:
                    st.error("❌ Model did not reach target output")
    
    # Sidebar info
    st.sidebar.divider()
    st.sidebar.header("📊 Booklet Info")
    st.sidebar.write(f"**Task:** {booklet_data['task_name']}")
    st.sidebar.write(f"**Steps:** {booklet_data['total_steps']}")
    st.sidebar.write(f"**Generated:** {booklet_data.get('generated_at', 'N/A')[:19]}")
    
    # Footer
    st.divider()
    st.markdown("**ARC Booklet Generator** | Step-by-step instructional puzzle solutions")


if __name__ == "__main__":
    main()

