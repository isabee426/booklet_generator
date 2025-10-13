#!/usr/bin/env python3
"""
Extract training data from visualization metadata for RL/imitation learning

This script processes saved visualizations and extracts:
1. All hypotheticals generated per step
2. Verification scores (distances from expected output)
3. Which transform was chosen
4. Full reasoning traces

The goal: Train a student model to internalize the verification function
- Learn which hypotheticals are good WITHOUT needing to verify them
- Pick best approaches based on learned patterns
- Reason step-by-step like the teacher model

Output: JSON dataset ready for training a student model
"""

import json
import os
import sys
from typing import List, Dict, Any, Tuple
from arc_visual_solver import ARCVisualSolver


def extract_step_data(vizs: List[Dict], expected_output: List[List[int]]) -> List[Dict]:
    """Extract structured step data from visualizations"""
    steps = []
    current_step = None
    step_number = 0
    
    for viz in vizs:
        filename = viz['filename']
        metadata = viz['metadata']
        
        # Check if this is a hypotheticals directory or file in it
        if '_hypotheticals' in viz['path']:
            # Extract step number from path
            parts = os.path.basename(os.path.dirname(viz['path']))
            if parts.endswith('_hypotheticals'):
                step_num = int(parts.split('_')[0])
                
                # Initialize step if needed
                if current_step is None or current_step['step_number'] != step_num:
                    if current_step is not None:
                        steps.append(current_step)
                    
                    current_step = {
                        'step_number': step_num,
                        'hypotheticals': [],
                        'transform': None
                    }
            
            # Add hypothetical to current step
            if 'h_' in filename and 'grid_data' in metadata:
                solver = ARCVisualSolver()
                grid = metadata['grid_data']
                description = metadata.get('description', 'Unknown')
                
                # Calculate verification score
                verification_score = solver.calculate_grid_difference(grid, expected_output)
                
                current_step['hypotheticals'].append({
                    'description': description,
                    'grid': grid,
                    'verification_score': verification_score,
                    'path': viz['path']
                })
        
        # Check if this is a transform
        elif 'transform' in filename and 'grid_data' in metadata:
            if current_step is not None:
                current_step['transform'] = {
                    'description': metadata.get('description', 'Unknown'),
                    'grid': metadata['grid_data'],
                    'path': viz['path']
                }
                steps.append(current_step)
                current_step = None
    
    # Add any remaining step
    if current_step is not None:
        steps.append(current_step)
    
    return steps


def extract_task_data(task_id: str, task_file: str) -> Dict[str, Any]:
    """Extract all reasoning traces for a task"""
    solver = ARCVisualSolver()
    
    # Load task to get expected outputs
    task = solver.load_task(task_file)
    
    task_data = {
        'task_id': task_id,
        'training_examples': [],
        'test_example': None
    }
    
    # Process training examples (0-2)
    for ex_num in range(len(task['train'])):
        vizs = solver.list_visualizations(task_id, training_example=ex_num)
        if not vizs:
            continue
        
        expected_output = task['train'][ex_num]['output']
        steps = extract_step_data(vizs, expected_output)
        
        # Analyze each step
        for step in steps:
            if step['hypotheticals']:
                # Find best hypothetical
                best_idx = min(range(len(step['hypotheticals'])),
                             key=lambda i: step['hypotheticals'][i]['verification_score'])
                
                # Mark which was best
                for i, hyp in enumerate(step['hypotheticals']):
                    hyp['is_best'] = (i == best_idx)
                
                # Check if transform matched best
                if step['transform']:
                    step['transform']['matched_best'] = (
                        step['transform']['grid'] == step['hypotheticals'][best_idx]['grid']
                    )
        
        task_data['training_examples'].append({
            'example_number': ex_num,
            'input': task['train'][ex_num]['input'],
            'output': task['train'][ex_num]['output'],
            'reasoning_steps': steps
        })
    
    # Process test example
    if len(task['test']) > 0:
        test_vizs = solver.list_visualizations(task_id, training_example=len(task['train']))
        if test_vizs:
            expected_output = task['test'][0].get('output', None)
            if expected_output:
                test_steps = extract_step_data(test_vizs, expected_output)
                
                task_data['test_example'] = {
                    'input': task['test'][0]['input'],
                    'output': expected_output,
                    'reasoning_steps': test_steps
                }
    
    return task_data


def analyze_dataset(dataset: List[Dict]) -> Dict[str, Any]:
    """Analyze extracted dataset for quality metrics"""
    total_steps = 0
    total_hypotheticals = 0
    perfect_hypotheticals = 0
    close_hypotheticals = 0
    chosen_was_best = 0
    total_transforms = 0
    
    for task in dataset:
        for example in task['training_examples']:
            for step in example['reasoning_steps']:
                total_steps += 1
                
                for hyp in step['hypotheticals']:
                    total_hypotheticals += 1
                    if hyp['verification_score'] == 0:
                        perfect_hypotheticals += 1
                    elif hyp['verification_score'] <= 3:
                        close_hypotheticals += 1
                
                if step['transform']:
                    total_transforms += 1
                    if step['transform'].get('matched_best', False):
                        chosen_was_best += 1
    
    return {
        'total_tasks': len(dataset),
        'total_steps': total_steps,
        'total_hypotheticals': total_hypotheticals,
        'avg_hypotheticals_per_step': total_hypotheticals / max(total_steps, 1),
        'perfect_hypotheticals': perfect_hypotheticals,
        'close_hypotheticals': close_hypotheticals,
        'chosen_was_best_rate': chosen_was_best / max(total_transforms, 1)
    }


def create_training_pairs(task_data: Dict) -> List[Dict]:
    """
    Create input/output pairs for training student model
    
    For each step, create:
    - Input: Current state + context
    - Output: Distribution over hypotheticals (weighted by verification scores)
    """
    training_pairs = []
    
    for example in task_data['training_examples']:
        for step in example['reasoning_steps']:
            if not step['hypotheticals']:
                continue
            
            # Input: Previous context + current state
            input_data = {
                'task_id': task_data['task_id'],
                'example_input': example['input'],
                'example_output': example['output'],
                'step_number': step['step_number'],
                'hypotheticals': [
                    {
                        'description': h['description'],
                        'grid': h['grid']
                    }
                    for h in step['hypotheticals']
                ]
            }
            
            # Output: Target distribution (softmax of negative scores)
            import math
            scores = [h['verification_score'] for h in step['hypotheticals']]
            
            # Convert scores to probabilities (lower score = higher probability)
            # Use temperature to control distribution sharpness
            temperature = 2.0
            exp_scores = [math.exp(-s / temperature) for s in scores]
            sum_exp = sum(exp_scores)
            target_probs = [e / sum_exp for e in exp_scores]
            
            # Also include which one was actually chosen
            chosen_idx = None
            if step['transform']:
                for i, hyp in enumerate(step['hypotheticals']):
                    if hyp['grid'] == step['transform']['grid']:
                        chosen_idx = i
                        break
            
            output_data = {
                'target_distribution': target_probs,  # For imitation learning
                'verification_scores': scores,  # For RL
                'best_idx': scores.index(min(scores)),  # Ground truth best
                'chosen_idx': chosen_idx  # What teacher actually picked
            }
            
            training_pairs.append({
                'input': input_data,
                'output': output_data
            })
    
    return training_pairs


def main():
    """Main extraction pipeline"""
    if len(sys.argv) < 2:
        print("Usage: python extract_training_data.py <output_file> [task_files...]")
        print("\nExample:")
        print('  python extract_training_data.py training_data.json "ARC-AGI-2/ARC-AGI-2/data/training/*.json"')
        sys.exit(1)
    
    output_file = sys.argv[1]
    task_files = sys.argv[2:] if len(sys.argv) > 2 else []
    
    # If no task files specified, try to find solved tasks
    if not task_files:
        viz_dir = os.path.join(os.path.dirname(__file__), 'visualizations')
        if os.path.exists(viz_dir):
            task_ids = [d for d in os.listdir(viz_dir) 
                       if os.path.isdir(os.path.join(viz_dir, d))]
            print(f"Found {len(task_ids)} tasks with visualizations")
            task_files = [f"ARC-AGI-2/ARC-AGI-2/data/evaluation/{tid}.json" 
                         for tid in task_ids]
    
    # Extract data from each task
    dataset = []
    for task_file in task_files:
        if not os.path.exists(task_file):
            print(f"Warning: {task_file} not found, skipping")
            continue
        
        task_id = os.path.splitext(os.path.basename(task_file))[0]
        print(f"Processing {task_id}...")
        
        try:
            task_data = extract_task_data(task_id, task_file)
            if task_data['training_examples']:
                dataset.append(task_data)
                print(f"  ✓ Extracted {len(task_data['training_examples'])} training examples")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Analyze dataset
    print(f"\n{'='*60}")
    print("Dataset Analysis")
    print(f"{'='*60}")
    analysis = analyze_dataset(dataset)
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Create training pairs
    all_training_pairs = []
    for task in dataset:
        pairs = create_training_pairs(task)
        all_training_pairs.extend(pairs)
    
    print(f"\nCreated {len(all_training_pairs)} training pairs")
    
    # Save dataset
    output_data = {
        'metadata': {
            'num_tasks': len(dataset),
            'num_training_pairs': len(all_training_pairs),
            'analysis': analysis
        },
        'tasks': dataset,
        'training_pairs': all_training_pairs
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved to {output_file}")
    print(f"\nDataset ready for training!")
    print(f"  - Load training_pairs for supervised learning")
    print(f"  - Use verification_scores for RL rewards")
    print(f"  - Use target_distribution for imitation learning")


if __name__ == "__main__":
    main()
