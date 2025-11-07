#!/usr/bin/env python3
"""
Streamlit UI for Visual Step Results

Shows:
- Multiple training examples per puzzle
- Each step with all attempts (slide through with arrows)
- Generated vs ground truth side-by-side
- Action type tags (COPY/MOVE/MODIFY)
- Conditional step detection
- Success tags and accuracy metrics
- Overview of all training examples
- Organized visual layout

Usage:
    streamlit run scripts/view_visual_steps.py
"""

import streamlit as st
import json
from pathlib import Path
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from arc_visualizer import grid_to_image

st.set_page_config(page_title="Visual Step Results", layout="wide")

# Custom CSS
st.markdown("""
<style>
.success-tag {
    background: #28a745;
    color: white;
    padding: 5px 15px;
    border-radius: 5px;
    font-weight: bold;
    display: inline-block;
    margin: 10px 0;
}
.fail-tag {
    background: #dc3545;
    color: white;
    padding: 5px 15px;
    border-radius: 5px;
    font-weight: bold;
    display: inline-block;
    margin: 10px 0;
}
.attempt-box {
    border: 2px solid #ddd;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
}
.ground-truth-box {
    border: 2px solid #28a745;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    background: #f8f9fa;
}
</style>
""", unsafe_allow_html=True)

st.title("🎨 Visual Step Generator - Results")

# Source selector
st.sidebar.markdown("## 📂 Data Source")
data_source = st.sidebar.radio(
    "View from:",
    ["Recent Results", "Successful Runs"],
    label_visibility="collapsed"
)

# Find results
if data_source == "Successful Runs":
    results_dir = Path("successful_runs")
    if not results_dir.exists():
        st.warning("⚠️ No successful runs yet. Mark runs as successful with:\n`python scripts/mark_successful.py --puzzle ID`")
        st.stop()
else:
    results_dir = Path("visual_step_results")
    if not results_dir.exists():
        st.warning("⚠️ No results found. Run: `python scripts/visual_step_generator.py --puzzle 05f2a901`")
        st.stop()

puzzles = [p.name for p in results_dir.iterdir() if p.is_dir()]

if not puzzles:
    if data_source == "Successful Runs":
        st.warning("No successful runs yet")
    else:
        st.warning("No puzzles generated yet")
    st.stop()

# Select puzzle
st.sidebar.markdown("## 📁 Select Puzzle")
selected_puzzle = st.sidebar.selectbox("Puzzle ID", sorted(puzzles), label_visibility="collapsed")

if not selected_puzzle:
    st.stop()

# Show success badge if viewing successful runs
if data_source == "Successful Runs":
    metadata_file = results_dir / selected_puzzle / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            meta = json.load(f)
        st.sidebar.success(f"✅ Successful Run\nMarked: {meta['marked_successful'][:10]}")
        if 'stats' in meta:
            stats = meta['stats']
            st.sidebar.metric("Performance", f"{stats['average_accuracy']:.1%}")

# Find all training and test examples for this puzzle
puzzle_dir = results_dir / selected_puzzle
training_dirs = sorted([d for d in puzzle_dir.iterdir() if d.is_dir() and d.name.startswith("training_")])
test_dirs = sorted([d for d in puzzle_dir.iterdir() if d.is_dir() and d.name.startswith("test_")])

if not training_dirs and not test_dirs:
    st.warning(f"No examples found for puzzle {selected_puzzle}")
    st.stop()

# Select example type
st.sidebar.markdown("## 🎯 Example Type")
example_types = []
if training_dirs:
    example_types.append("Training Examples")
if test_dirs:
    example_types.append("Test Examples")

example_type = st.sidebar.radio(
    "Type",
    example_types,
    label_visibility="collapsed"
)

# Select specific example
if example_type == "Training Examples":
    st.sidebar.markdown("## 📚 Select Training Example")
    training_options = {}
    for td in training_dirs:
        training_num = int(td.name.split("_")[1])
        training_options[f"Training Example {training_num}"] = training_num
    
    selected_label = st.sidebar.selectbox(
        "Training",
        list(training_options.keys()),
        label_visibility="collapsed"
    )
    selected_num = training_options[selected_label]
    current_dir = puzzle_dir / f"training_{selected_num:02d}"
    is_test = False
    
else:  # Test Examples
    st.sidebar.markdown("## 🧪 Select Test Example")
    test_options = {}
    for td in test_dirs:
        test_num = int(td.name.split("_")[1])
        test_options[f"Test Example {test_num}"] = test_num
    
    selected_label = st.sidebar.selectbox(
        "Test",
        list(test_options.keys()),
        label_visibility="collapsed"
    )
    selected_num = test_options[selected_label]
    current_dir = puzzle_dir / f"test_{selected_num:02d}"
    is_test = True

# Load data
results_file = current_dir / "results.json"

if not results_file.exists():
    st.error(f"Results file not found: {results_file}")
    st.stop()

with open(results_file) as f:
    data = json.load(f)

if is_test:
    st.header(f"Puzzle: {selected_puzzle} - 🧪 Test Example {selected_num}")
    st.caption(f"Model: {data.get('model', 'unknown')} • Generated: {data['timestamp'][:10]}")
    
    if 'final_accuracy' in data and data['final_accuracy'] is not None:
        st.success(f"✅ Final Accuracy: {data['final_accuracy']:.1%}")
    
    # Validate final output for temporary colors
    if data.get('steps'):
        final_step = data['steps'][-1]
        if final_step.get('grid'):
            import numpy as np
            unique_colors = set(np.array(final_step['grid']).flatten())
            # Get valid colors from test_input_grid
            valid_colors = set()
            if data.get('test_input_grid'):
                for row in data['test_input_grid']:
                    valid_colors.update(row)
            if data.get('test_output_grid'):
                for row in data['test_output_grid']:
                    valid_colors.update(row)
            invalid_colors = unique_colors - valid_colors
            if invalid_colors:
                st.error(f"⚠️ WARNING: Temporary colors detected in final output: {sorted(invalid_colors)}")
                st.error("All HIGHLIGHT temporary colors should be removed by the final step!")
    
    if 'num_training_booklets_used' in data:
        st.info(f"📚 Learned pattern from {data['num_training_booklets_used']} training examples")
else:
    st.header(f"Puzzle: {selected_puzzle} - 📚 Training Example {selected_num}")
    st.caption(f"Model: {data.get('model', 'unknown')} • Generated: {data['timestamp'][:10]}")
    
    # Validate final output for temporary colors
    if data.get('steps'):
        final_step = data['steps'][-1]
        if final_step.get('final_grid'):
            import numpy as np
            unique_colors = set(np.array(final_step['final_grid']).flatten())
            # Get valid colors from input/output grids
            valid_colors = set()
            if data.get('input_grid'):
                for row in data['input_grid']:
                    valid_colors.update(row)
            if data.get('expected_output_grid'):
                for row in data['expected_output_grid']:
                    valid_colors.update(row)
            invalid_colors = unique_colors - valid_colors
            if invalid_colors:
                st.error(f"⚠️ WARNING: Temporary colors detected in final output: {sorted(invalid_colors)}")
                st.error("All HIGHLIGHT temporary colors should be removed by the final step!")
    
    # Show training example info if multiple exist
    if len(training_dirs) > 1:
        st.info(f"📊 This puzzle has {len(training_dirs)} training examples. Use the sidebar to switch between them.")

# Show puzzle overview (input/output)
st.subheader("🎯 Puzzle Overview")

col_in, col_out = st.columns(2)

with col_in:
    st.markdown("**📥 INPUT**")
    if is_test:
        input_img_path = current_dir / "test_input.png"
        input_grid_key = 'test_input_grid'
    else:
        input_img_path = current_dir / "input.png"
        input_grid_key = 'input_grid'
    
    if input_img_path.exists():
        st.image(str(input_img_path), use_column_width=True)
    
    with st.expander("Show input grid"):
        if input_grid_key in data:
            st.code(str(data[input_grid_key]))

with col_out:
    st.markdown("**📤 EXPECTED OUTPUT (Final Goal)**")
    if is_test:
        output_img_path = current_dir / "test_output.png"
        output_grid_key = 'test_output_grid'
    else:
        output_img_path = current_dir / "expected_output.png"
        output_grid_key = 'expected_output_grid'
    
    if output_img_path.exists():
        st.image(str(output_img_path), use_column_width=True)
    elif is_test:
        st.info("Ground truth not available (evaluation set)")
    
    with st.expander("Show output grid"):
        if output_grid_key in data and data[output_grid_key]:
            st.code(str(data[output_grid_key]))
        elif is_test:
            st.write("Not available")

# Show Phase 1 analysis
with st.expander("📋 Phase 1: Transformation Analysis (Shared Across All Training Examples)", expanded=False):
    # Check if this is a multi-training puzzle
    if len(training_dirs) > 1:
        st.info(f"ℹ️ This analysis was generated ONCE and is shared across all {len(training_dirs)} training examples for this puzzle.")
    
    analysis_text = data['phase1_analysis']
    
    # Highlight if variable steps are mentioned
    if "VARIABLE STEP COUNTS" in analysis_text or "different numbers of steps" in analysis_text.lower():
        st.warning("⚠️ This puzzle has variable step counts across training examples. The model uses conditional language to handle this.")
    
    st.text(analysis_text)

# Show pattern summary for test examples
if is_test and 'pattern_summary' in data:
    with st.expander("🎯 Learned Pattern (from Training Examples)", expanded=True):
        st.markdown(data['pattern_summary'])

st.divider()

# Summary stats
steps = data.get('steps', [])
total_steps = len(steps)

if total_steps == 0:
    st.error("No steps found in this example's results.")
    st.info("This might be an incomplete or failed run.")
    st.stop()

if is_test:
    # Test examples don't have comparison data
    col1, col2 = st.columns(2)
    col1.metric("Total Steps Generated", total_steps)
    if 'final_accuracy' in data and data['final_accuracy'] is not None:
        col2.metric("Final Accuracy", f"{data['final_accuracy']:.1%}")
    else:
        col2.metric("Ground Truth", "Not Available")
else:
    # Training examples have full comparison data
    perfect_matches = sum(1 for s in steps if s['comparison']['match'])
    used_gt_count = sum(1 for s in steps if s['used_ground_truth'])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Steps", total_steps)
    col2.metric("Perfect Matches", f"{perfect_matches}/{total_steps}")
    col3.metric("Avg Accuracy", f"{sum(s['comparison']['accuracy'] for s in steps)/total_steps:.1%}")
    col4.metric("Used GT Fallback", used_gt_count)

st.divider()

# Step selector
if is_test:
    # Test steps may have sub-steps like "3.1", display them properly
    step_options = [(i+1, step.get('step_num', i+1)) for i, step in enumerate(steps)]
    selected_idx = st.selectbox(
        "🔢 Select Step",
        range(len(step_options)),
        format_func=lambda x: f"Step {step_options[x][1]}"
    )
    step_num = selected_idx + 1  # For indexing
    actual_step_id = step_options[selected_idx][1]
else:
    # Training steps are regular integers
    step_num = st.selectbox(
        "🔢 Select Step",
        range(1, total_steps + 1),
        format_func=lambda x: f"Step {x}"
    )
    actual_step_id = step_num

step_data = steps[step_num - 1]

if is_test:
    st.header(f"Step {actual_step_id}")
else:
    st.header(f"Step {step_num}")

# Handle test vs training display differently
if is_test:
    # Test examples just have a single step result
    if step_data.get('success', True):
        st.markdown('<div class="success-tag">✅ Step Generated</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="fail-tag">❌ Generation Failed</div>', unsafe_allow_html=True)
else:
    # Training examples have comparison data
    attempts = step_data['attempts']
    successful = any(a['success'] for a in attempts)
    comparison = step_data['comparison']
    
    if comparison['match']:
        st.markdown('<div class="success-tag">✅ SUCCESS! Perfect Match</div>', unsafe_allow_html=True)
    elif successful and comparison['accuracy'] > 0.8:
        st.markdown(f'<div class="success-tag">✅ Close Match ({comparison["accuracy"]:.1%})</div>', unsafe_allow_html=True)
    elif successful:
        st.markdown(f'<div class="fail-tag">⚠️ Partial Match ({comparison["accuracy"]:.1%})</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="fail-tag">❌ All Attempts Failed</div>', unsafe_allow_html=True)

# Display depends on test vs training
if is_test:
    # Test examples: just show the generated step
    st.subheader("🧪 Generated Step")
    
    st.markdown('<div class="attempt-box">', unsafe_allow_html=True)
    
    # Get actual step identifier (may be like "3.1")
    actual_step_id = step_data.get('step_num', step_num)
    st.markdown(f"### 🤖 Generated Step {actual_step_id}")
    
    # Show image (handle both integer and sub-step format)
    step_filename = str(actual_step_id).replace('.', '_')
    step_img_path = current_dir / f"step_{step_filename}.png"
    if step_img_path.exists():
        st.image(str(step_img_path), use_column_width=True)
    else:
        st.info("No grid generated")
    
    # Show description
    st.markdown("**Description:**")
    desc = step_data.get('description', 'No description')
    
    # Detect action type
    if desc.startswith("COPY:"):
        st.success("📋 COPY")
    elif desc.startswith("MOVE:"):
        st.info("➡️ MOVE")
    elif desc.startswith("EXPAND:"):
        st.info("🔍 EXPAND")
    elif desc.startswith("HIGHLIGHT:"):
        st.warning("🔦 HIGHLIGHT")
    elif desc.startswith("TRANSFORM:"):
        st.success("🔄 TRANSFORM")
    elif desc.startswith("FILL:"):
        st.info("🎨 FILL")
    elif desc.startswith("MODIFY:"):
        st.warning("✏️ MODIFY")
    elif desc.startswith("NO-OP:"):
        st.info("⏭️ NO-OP")
    
    # Highlight 3-step subprocess
    if desc.startswith("HIGHLIGHT:"):
        st.info("📍 Step 1 of 3-step subprocess: HIGHLIGHT → TRANSFORM → Complete")
    elif desc.startswith("TRANSFORM:") and "highlighted" in desc.lower():
        st.info("📍 Step 2 of 3-step subprocess: Applying transformation to highlighted cells")
    
    st.write(desc)
    
    # Show grid array
    with st.expander("Show grid array"):
        if step_data.get('grid'):
            st.code(str(step_data['grid']))
        else:
            st.write("No grid")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
else:
    # Training examples: show attempts and ground truth
    # Attempt selector (slide through attempts)
    st.subheader("🔄 Model Attempts")
    
    if 'attempt_idx' not in st.session_state:
        st.session_state.attempt_idx = 0
    
    attempts = step_data['attempts']
    comparison = step_data['comparison']
    
    col_left, col_mid, col_right = st.columns([1, 10, 1])
    
    with col_left:
        if st.button("◀", key="prev_attempt"):
            if st.session_state.attempt_idx > 0:
                st.session_state.attempt_idx -= 1
                st.rerun()
    
    with col_mid:
        st.markdown(f"**Attempt {st.session_state.attempt_idx + 1} of {len(attempts)}**")
    
    with col_right:
        if st.button("▶", key="next_attempt"):
            if st.session_state.attempt_idx < len(attempts) - 1:
                st.session_state.attempt_idx += 1
                st.rerun()
    
    # Show current attempt
    attempt_idx = st.session_state.attempt_idx
    attempt = attempts[attempt_idx]
    
    # Two columns: Generated vs Ground Truth
    col_gen, col_gt = st.columns(2)

    with col_gen:
        st.markdown('<div class="attempt-box">', unsafe_allow_html=True)
        st.markdown(f"### 🤖 Generated (Attempt {attempt['attempt']})")
        
        if attempt['success']:
            st.success("✓ Valid")
        else:
            st.error(f"✗ Failed: {', '.join(attempt.get('errors', ['Unknown error']))}")
        
        # Show image
        attempt_img_path = current_dir / f"step_{step_num:02d}_attempt_{attempt['attempt']}.png"
        if attempt_img_path.exists():
            st.image(str(attempt_img_path), use_column_width=True)
        else:
            st.info("No grid generated")
        
        # Show description
        st.markdown("**Description:**")
        desc = attempt.get('description', 'No description')
        
        # Detect action type
        action_type = None
        if desc.startswith("COPY:"):
            action_type = "📋 COPY"
            st.success(action_type)
        elif desc.startswith("MOVE:"):
            action_type = "➡️ MOVE"
            st.info(action_type)
        elif desc.startswith("EXPAND:"):
            action_type = "🔍 EXPAND"
            st.info(action_type)
        elif desc.startswith("HIGHLIGHT:"):
            action_type = "🔦 HIGHLIGHT"
            st.warning(action_type)
        elif desc.startswith("TRANSFORM:"):
            action_type = "🔄 TRANSFORM"
            st.success(action_type)
        elif desc.startswith("FILL:"):
            action_type = "🎨 FILL"
            st.info(action_type)
        elif desc.startswith("MODIFY:"):
            action_type = "✏️ MODIFY"
            st.warning(action_type)
        elif desc.startswith("NO-OP:"):
            action_type = "⏭️ NO-OP"
            st.info(action_type)
        
        # Highlight 3-step subprocess
        if desc.startswith("HIGHLIGHT:"):
            st.info("📍 Step 1 of 3-step subprocess: HIGHLIGHT → TRANSFORM → Complete")
        elif desc.startswith("TRANSFORM:") and "highlighted" in desc.lower():
            st.info("📍 Step 2 of 3-step subprocess: Applying transformation to highlighted cells")
        
        # Highlight conditional steps
        if any(phrase in desc.lower() for phrase in ["if they exist", "if exist", "otherwise do nothing", "any remaining", "any additional"]):
            st.info("🔀 Conditional - adapts to available objects")
        
        st.write(desc)
        
        # Show grid array
        with st.expander("Show grid array"):
            if attempt.get('grid'):
                st.code(str(attempt['grid']))
            else:
                st.write("No grid")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_gt:
        st.markdown('<div class="ground-truth-box">', unsafe_allow_html=True)
        st.markdown("### ✅ Ground Truth")
        
        # Show image
        gt_img_path = current_dir / f"step_{step_num:02d}_ground_truth.png"
        if gt_img_path.exists():
            st.image(str(gt_img_path), use_column_width=True)
        
        # Show description
        st.markdown("**Description:**")
        st.write(step_data['ground_truth']['description'])
        
        # Show grid array
        with st.expander("Show grid array"):
            st.code(str(step_data['ground_truth']['grid']))
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Comparison details
    if comparison['accuracy'] < 1.0 and comparison.get('differences'):
        st.subheader("🔍 Differences")
        
        diff_data = []
        for diff in comparison['differences'][:10]:
            diff_data.append({
                "Position": f"({diff['position'][0]}, {diff['position'][1]})",
                "Generated": f"color {diff['generated']}",
                "Expected": f"color {diff['expected']}"
            })
        
        st.table(diff_data)

# Navigation between steps
st.divider()

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if step_num > 1:
        if st.button("⬅️ Previous Step"):
            st.session_state.attempt_idx = 0  # Reset attempt index
            # Streamlit will rerun and step selector will change

with col3:
    if step_num < total_steps:
        if st.button("Next Step ➡️"):
            st.session_state.attempt_idx = 0  # Reset attempt index
            # Streamlit will rerun and step selector will change

# Sidebar: All steps overview for current example
st.sidebar.markdown("---")
if is_test:
    st.sidebar.subheader("🧪 Test Steps (Current)")
else:
    st.sidebar.subheader("📊 Steps Summary (Current)")

for i, step in enumerate(steps, 1):
    if is_test:
        # Test examples: show step identifier (may be "3.1")
        success_icon = "✅" if step.get('success', True) else "❌"
        step_id = step.get('step_num', i)
        st.sidebar.write(f"{success_icon} Step {step_id}")
    else:
        # Training examples: show accuracy
        comp = step['comparison']
        if comp['match']:
            icon = "✅"
        elif comp['accuracy'] > 0.8:
            icon = "⚠️"
        else:
            icon = "❌"
        
        used_gt = " 🔄" if step['used_ground_truth'] else ""
        
        st.sidebar.write(f"{icon} Step {i}: {comp['accuracy']:.0%}{used_gt}")

if not is_test:
    st.sidebar.caption("🔄 = Used ground truth fallback")

# Show overview of all examples (training and test)
if len(training_dirs) > 1 or len(test_dirs) > 0:
    st.sidebar.markdown("---")
    
    # Show training examples overview
    if len(training_dirs) > 1:
        st.sidebar.subheader("📈 All Training Examples")
        
        for td in training_dirs:
            training_num = int(td.name.split("_")[1])
            results_path = td / "results.json"
            
            if results_path.exists():
                with open(results_path) as f:
                    t_data = json.load(f)
                
                t_steps = t_data['steps']
                t_perfect = sum(1 for s in t_steps if s['comparison']['match'])
                t_avg_acc = sum(s['comparison']['accuracy'] for s in t_steps) / len(t_steps) if t_steps else 0
                
                if not is_test and training_num == selected_num:
                    st.sidebar.markdown(f"**→ Training {training_num}** (viewing)")
                else:
                    st.sidebar.write(f"Training {training_num}:")
                
                st.sidebar.write(f"  {len(t_steps)} steps • {t_perfect} perfect • {t_avg_acc:.0%} avg")
    
    # Show test examples overview
    if len(test_dirs) > 0:
        st.sidebar.subheader("🧪 Test Examples")
        
        for td in test_dirs:
            test_num = int(td.name.split("_")[1])
            results_path = td / "results.json"
            
            if results_path.exists():
                with open(results_path) as f:
                    t_data = json.load(f)
                
                t_steps = t_data['steps']
                
                if is_test and test_num == selected_num:
                    st.sidebar.markdown(f"**→ Test {test_num}** (viewing)")
                else:
                    st.sidebar.write(f"Test {test_num}:")
                
                if 'final_accuracy' in t_data and t_data['final_accuracy'] is not None:
                    st.sidebar.write(f"  {len(t_steps)} steps • {t_data['final_accuracy']:.0%} final")
                else:
                    st.sidebar.write(f"  {len(t_steps)} steps • No GT")


