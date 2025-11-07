#!/usr/bin/env python3
"""
ARC Puzzle Creator - Flask Web Application
Clean, modern UI matching arcprize.org/play
"""

from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
import json
import numpy as np
from PIL import Image
from datetime import datetime
import io
import re
import os
import shutil

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not installed. AI rewrite feature will not work.")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'arc-puzzle-creator-secret-key'

# ARC color palette
ARC_COLORS = {
    0: (0, 0, 0), 1: (0, 116, 217), 2: (255, 65, 54), 3: (46, 204, 64), 4: (255, 220, 0),
    5: (255, 133, 27), 6: (240, 18, 190), 7: (127, 219, 255), 8: (135, 12, 37), 9: (149, 117, 205),
}

def load_arc_data():
    """Load all ARC tasks"""
    base_path = Path("../saturn-arc/ARC-AGI-2/ARC-AGI-1/data")
    tasks = {'training': {}, 'evaluation': {}}
    
    for dataset in ['training', 'evaluation']:
        dataset_path = base_path / dataset
        if dataset_path.exists():
            for file_path in sorted(dataset_path.glob("*.json")):
                try:
                    with open(file_path, 'r') as f:
                        tasks[dataset][file_path.stem] = json.load(f)
                except:
                    pass
    
    return tasks

def grid_to_image(grid, cell_size=30):
    """Convert grid to PIL Image"""
    height = len(grid)
    width = len(grid[0]) if grid else 0
    
    img_array = np.zeros((height * cell_size, width * cell_size, 3), dtype=np.uint8)
    
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            color = ARC_COLORS.get(value, (128, 128, 128))
            img_array[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size] = color
    
    return Image.fromarray(img_array)

def get_universal_steps(task_id):
    """Get universal steps for a puzzle"""
    universal_path = Path("visual_traces") / task_id / "universal_steps.json"
    
    if not universal_path.exists():
        return []
    
    try:
        with open(universal_path, 'r') as f:
            return json.load(f)
    except:
        return []

def save_universal_steps(task_id, steps):
    """Save universal steps for a puzzle"""
    puzzle_path = Path("visual_traces") / task_id
    puzzle_path.mkdir(parents=True, exist_ok=True)
    
    universal_path = puzzle_path / "universal_steps.json"
    with open(universal_path, 'w') as f:
        json.dump(steps, f, indent=2)

def get_example_steps(task_id, example_type, example_num):
    """Get all steps for an example"""
    example_path = Path("visual_traces") / task_id / f"{example_type}_{example_num:02d}"
    
    if not example_path.exists():
        return []
    
    steps = []
    for step_dir in sorted(example_path.glob("step_*")):
        if step_dir.is_dir():
            json_path = step_dir / "step.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    step_data = json.load(f)
                    match = re.match(r"step_(\d+)", step_dir.name)
                    if match:
                        step_data["step_num"] = int(match.group(1))
                    steps.append(step_data)
    
    return steps

def save_step(task_id, example_type, example_num, step_data, step_num=None):
    """Save a step (creates new or updates existing)"""
    example_path = Path("visual_traces") / task_id / f"{example_type}_{example_num:02d}"
    example_path.mkdir(parents=True, exist_ok=True)
    
    if step_num is None:
        # Create new step
        existing_steps = sorted(example_path.glob("step_*"))
        next_step_num = len([s for s in existing_steps if s.is_dir()]) + 1
    else:
        # Update existing step
        next_step_num = step_num
    
    step_dir = example_path / f"step_{next_step_num:02d}"
    step_dir.mkdir(exist_ok=True)
    
    # Save JSON
    with open(step_dir / "step.json", 'w') as f:
        json.dump(step_data, f, indent=2)
    
    # Save grid image
    if step_data.get('grid'):
        img = grid_to_image(step_data['grid'])
        img.save(step_dir / "grid.png")
    
    # Save metadata
    with open(step_dir / "metadata.txt", 'w') as f:
        f.write(f"Task ID: {task_id}\n")
        f.write(f"Example Type: {example_type}\n")
        f.write(f"Example Number: {example_num:02d}\n")
        f.write(f"Step: {next_step_num:02d}\n")
        f.write(f"Timestamp: {step_data.get('timestamp', '')}\n")
        f.write(f"Step Name: {step_data.get('step_name', '')}\n\n")
        f.write(f"Description:\n{step_data.get('description', '')}\n")
    
    return next_step_num

# Routes
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/viewer')
def viewer():
    """Completed booklets viewer page"""
    return render_template('viewer.html')

@app.route('/api/tasks')
def get_tasks():
    """Get all available tasks"""
    tasks = load_arc_data()
    return jsonify({
        'training': list(tasks['training'].keys()),
        'evaluation': list(tasks['evaluation'].keys())
    })

@app.route('/api/task/<dataset>/<task_id>')
def get_task(dataset, task_id):
    """Get specific task data"""
    tasks = load_arc_data()
    if dataset in tasks and task_id in tasks[dataset]:
        return jsonify(tasks[dataset][task_id])
    return jsonify({'error': 'Task not found'}), 404

@app.route('/api/universal_steps/<task_id>')
def get_universal_steps_api(task_id):
    """Get universal steps for a puzzle"""
    steps = get_universal_steps(task_id)
    return jsonify(steps)

@app.route('/api/steps/<task_id>/<example_type>/<int:example_num>')
def get_steps(task_id, example_type, example_num):
    """Get steps for an example"""
    steps = get_example_steps(task_id, example_type, example_num)
    return jsonify(steps)

@app.route('/api/save_step', methods=['POST'])
def api_save_step():
    """Save a new step"""
    data = request.json
    
    task_id = data.get('task_id')
    example_type = data.get('example_type')
    example_num = data.get('example_num')
    step_num = data.get('step_num')
    
    step_data = {
        'task_id': task_id,
        'example_type': example_type,
        'example_num': example_num,
        'timestamp': datetime.now().isoformat(),
        'step_name': data.get('step_name', ''),
        'description': data.get('description', ''),
        'grid': data.get('grid'),
        'original_input': data.get('original_input'),
        'original_output': data.get('original_output')
    }
    
    saved_step_num = save_step(task_id, example_type, example_num, step_data, step_num)
    
    # Update universal steps - ALWAYS use the most recent version
    universal_steps = get_universal_steps(task_id)
    
    # Create/update universal step with most recent data
    universal_step = {
        'step_num': saved_step_num,
        'step_name': data.get('step_name', ''),
        'description': data.get('description', ''),
        'last_updated': datetime.now().isoformat(),
        'updated_from': f"{example_type}_{example_num:02d}"
    }
    
    # Find and update, or add new
    found = False
    for i, ustep in enumerate(universal_steps):
        if ustep.get('step_num') == saved_step_num:
            # Always replace with most recent version
            universal_steps[i] = universal_step
            found = True
            break
    
    if not found:
        universal_steps.append(universal_step)
    
    # Sort by step number
    universal_steps.sort(key=lambda x: x.get('step_num', 0))
    
    # Save universal steps
    save_universal_steps(task_id, universal_steps)
    
    return jsonify({
        'success': True,
        'step_num': saved_step_num,
        'message': f'Step {saved_step_num:02d} saved successfully'
    })

@app.route('/api/completed_puzzles')
def get_completed_puzzles():
    """Get list of all puzzles with saved steps"""
    visual_traces_path = Path("visual_traces")
    
    if not visual_traces_path.exists():
        return jsonify({'puzzles': []})
    
    puzzles = []
    for puzzle_dir in sorted(visual_traces_path.iterdir()):
        if puzzle_dir.is_dir():
            puzzle_id = puzzle_dir.name
            
            # Count examples and steps
            training_examples = len(list(puzzle_dir.glob("training_*")))
            testing_examples = len(list(puzzle_dir.glob("testing_*")))
            
            training_steps = sum(
                len(list(ex_dir.glob("step_*"))) 
                for ex_dir in puzzle_dir.glob("training_*") 
                if ex_dir.is_dir()
            )
            
            testing_steps = sum(
                len(list(ex_dir.glob("step_*"))) 
                for ex_dir in puzzle_dir.glob("testing_*") 
                if ex_dir.is_dir()
            )
            
            if training_steps + testing_steps > 0:
                puzzles.append({
                    'id': puzzle_id,
                    'training_examples': training_examples,
                    'testing_examples': testing_examples,
                    'training_steps': training_steps,
                    'testing_steps': testing_steps
                })
    
    return jsonify({'puzzles': puzzles})

@app.route('/api/completed_puzzle/<puzzle_id>')
def get_completed_puzzle(puzzle_id):
    """Get all steps for a completed puzzle"""
    puzzle_path = Path("visual_traces") / puzzle_id
    
    if not puzzle_path.exists():
        return jsonify({'error': 'Puzzle not found'}), 404
    
    # Load universal steps
    universal_steps = get_universal_steps(puzzle_id)
    
    # Load training examples
    training = []
    for ex_dir in sorted(puzzle_path.glob("training_*")):
        if ex_dir.is_dir():
            match = re.match(r"training_(\d+)", ex_dir.name)
            if match:
                ex_num = int(match.group(1))
                steps = get_example_steps(puzzle_id, 'training', ex_num)
                if steps:
                    training.append({
                        'number': ex_num,
                        'steps': steps
                    })
    
    # Load testing examples
    testing = []
    for ex_dir in sorted(puzzle_path.glob("testing_*")):
        if ex_dir.is_dir():
            match = re.match(r"testing_(\d+)", ex_dir.name)
            if match:
                ex_num = int(match.group(1))
                steps = get_example_steps(puzzle_id, 'testing', ex_num)
                if steps:
                    testing.append({
                        'number': ex_num,
                        'steps': steps
                    })
    
    return jsonify({
        'puzzle_id': puzzle_id,
        'universal_steps': universal_steps,
        'training': training,
        'testing': testing
    })

@app.route('/api/grid_image', methods=['POST'])
def get_grid_image():
    """Generate grid image"""
    data = request.json
    grid = data.get('grid', [[]])
    
    img = grid_to_image(grid, cell_size=30)
    
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

@app.route('/api/reorder_steps', methods=['POST'])
def reorder_steps():
    """Reorder, delete, or insert steps"""
    data = request.json
    
    task_id = data.get('task_id')
    example_type = data.get('example_type')
    example_num = data.get('example_num')
    steps = data.get('steps', [])
    
    example_path = Path("visual_traces") / task_id / f"{example_type}_{example_num:02d}"
    
    if not example_path.exists():
        return jsonify({'success': False, 'error': 'Example not found'}), 404
    
    try:
        # Create temp directory for reorganization
        temp_path = example_path.parent / f"_temp_{example_type}_{example_num:02d}"
        temp_path.mkdir(exist_ok=True)
        
        # Process steps in new order
        new_step_num = 1
        for step_data in steps:
            if step_data.get('markedForDeletion'):
                # Skip deleted steps (don't copy)
                continue
            
            if step_data.get('isNew'):
                # Create placeholder for new step
                new_step_dir = temp_path / f"step_{new_step_num:02d}"
                new_step_dir.mkdir(exist_ok=True)
                
                # Create placeholder JSON
                placeholder = {
                    'task_id': task_id,
                    'example_type': example_type,
                    'example_num': example_num,
                    'step_num': new_step_num,
                    'step_name': step_data.get('step_name', f'Step {new_step_num}'),
                    'description': 'PLACEHOLDER - Edit this step to add content',
                    'grid': [[0]],
                    'timestamp': datetime.now().isoformat()
                }
                
                with open(new_step_dir / "step.json", 'w') as f:
                    json.dump(placeholder, f, indent=2)
                
                # Create placeholder grid
                placeholder_grid = [[0]]
                img = grid_to_image(placeholder_grid)
                img.save(new_step_dir / "grid.png")
                
                with open(new_step_dir / "metadata.txt", 'w') as f:
                    f.write(f"Task ID: {task_id}\n")
                    f.write(f"Example Type: {example_type}\n")
                    f.write(f"Example Number: {example_num:02d}\n")
                    f.write(f"Step: {new_step_num:02d}\n")
                    f.write(f"Status: PLACEHOLDER - Edit to add content\n")
            else:
                # Copy existing step
                old_step_num = step_data.get('old_step_num')
                old_step_dir = example_path / f"step_{old_step_num:02d}"
                
                if old_step_dir.exists():
                    new_step_dir = temp_path / f"step_{new_step_num:02d}"
                    shutil.copytree(old_step_dir, new_step_dir)
                    
                    # Update step number in JSON
                    json_path = new_step_dir / "step.json"
                    if json_path.exists():
                        with open(json_path, 'r') as f:
                            step_json = json.load(f)
                        step_json['step_num'] = new_step_num
                        with open(json_path, 'w') as f:
                            json.dump(step_json, f, indent=2)
                    
                    # Update metadata
                    meta_path = new_step_dir / "metadata.txt"
                    if meta_path.exists():
                        with open(meta_path, 'r') as f:
                            content = f.read()
                        # Update step number in metadata
                        content = re.sub(r'Step: \d+', f'Step: {new_step_num:02d}', content)
                        with open(meta_path, 'w') as f:
                            f.write(content)
            
            new_step_num += 1
        
        # Delete old example directory
        shutil.rmtree(example_path)
        
        # Rename temp to final
        temp_path.rename(example_path)
        
        # Update universal steps
        universal_steps = get_universal_steps(task_id)
        example_steps = get_example_steps(task_id, example_type, example_num)
        
        for step in example_steps:
            step_num = step['step_num']
            universal_step = {
                'step_num': step_num,
                'step_name': step.get('step_name', ''),
                'description': step.get('description', ''),
                'last_updated': datetime.now().isoformat(),
                'updated_from': f"{example_type}_{example_num:02d}"
            }
            
            # Update or add
            found = False
            for i, ustep in enumerate(universal_steps):
                if ustep.get('step_num') == step_num:
                    universal_steps[i] = universal_step
                    found = True
                    break
            
            if not found:
                universal_steps.append(universal_step)
        
        # Sort and save
        universal_steps.sort(key=lambda x: x.get('step_num', 0))
        save_universal_steps(task_id, universal_steps)
        
        return jsonify({
            'success': True,
            'message': f'Steps reorganized successfully! {new_step_num - 1} steps remain.'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ai_rewrite', methods=['POST'])
def ai_rewrite():
    """Rewrite step description using AI with context from all steps"""
    if not OPENAI_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'OpenAI not installed. Run: pip install openai'
        }), 500
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return jsonify({
            'success': False,
            'error': 'OPENAI_API_KEY not set. Set it in environment variables.'
        }), 500
    
    data = request.json
    description = data.get('description', '')
    task_id = data.get('task_id', '')
    current_step_num = data.get('step_num', 1)
    step_name = data.get('step_name', '')
    
    if not description:
        return jsonify({
            'success': False,
            'error': 'No description provided'
        }), 400
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Get all universal steps for context
        universal_steps = get_universal_steps(task_id)
        
        # Build context from previous steps
        context = ""
        if universal_steps:
            context = "\n\nContext - Previous steps in this puzzle:\n"
            for step in universal_steps:
                if step['step_num'] < current_step_num:
                    context += f"Step {step['step_num']}: {step['step_name']} - {step['description']}\n"
        
        prompt = f"""You are rewriting step descriptions for an ARC puzzle transformation.

Puzzle ID: {task_id}
Current Step: {current_step_num} - {step_name}
{context}

Original description for Step {current_step_num}:
{description}

Rewrite this description to be:
1. Clearer and more understandable
2. Limited to 2-3 sentences maximum
3. Technically accurate
4. Reference previous steps if relevant (e.g., "Building on Step 1...")
5. Use precise language (no "basically", "like", "kind of")
6. Focus on the specific transformation in THIS step

Rewritten description (2-3 sentences):"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at explaining ARC puzzle transformations clearly and concisely. You understand how to reference previous steps and build logical progression."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=250
        )
        
        rewritten = response.choices[0].message.content.strip()
        
        return jsonify({
            'success': True,
            'rewritten': rewritten,
            'original_length': len(description),
            'new_length': len(rewritten),
            'context_steps': len([s for s in universal_steps if s['step_num'] < current_step_num])
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

