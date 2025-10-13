# RL Training Data Generation

## Overview
The visual solver now generates rich training data suitable for reinforcement learning. The key insight: the verifier provides feedback BEFORE the model picks its transform, creating a supervised learning signal.

## The Workflow

### Step 1: Model Creates Hypotheticals
```
Model generates multiple possible approaches:
- Hypothetical 1: "Rotate 90 degrees clockwise"
- Hypothetical 2: "Flip horizontally"  
- Hypothetical 3: "Mirror across diagonal"
```

### Step 2: Verifier Runs (BEFORE Transform)
```
System compares each hypothetical to expected output:
✅ PERFECT: Flip horizontally (0 cells different)
🔸 CLOSE: Rotate 90 degrees (5 cells different)
❌ OFF: Mirror across diagonal (15 cells different)

💡 Best match: "Flip horizontally"
```

### Step 3: Model Picks Transform (WITH Feedback)
```
Model receives verification feedback and chooses:
Transform: "Flip horizontally" (the verified best option)
```

## RL Training Data Structure

Each step generates:

### 1. **State**: 
- Current grid
- Context from previous steps
- Task description

### 2. **Actions** (Hypotheticals):
- Multiple candidate transformations
- Each with description and grid output
- Stored in: `XX_hypotheticals/h_01.png`, `h_02.png`, etc.

### 3. **Rewards** (Verification):
- Distance from expected output
- 0 = perfect, higher = worse
- Stored in metadata: `grid_data` field

### 4. **Chosen Action** (Transform):
- The transform the model picked
- Whether it matched the best hypothetical
- Stored as: `XX_transform.png`

## Training a Student Model

### Supervised Learning Approach
Train a model to:
1. **Input**: Current grid + task context
2. **Output**: Predict which transformation to apply
3. **Loss**: Cross-entropy against the verified best hypothetical

### Imitation Learning
- Learn to mimic the reasoning pattern
- Without needing the verifier at inference time
- Model learns: "In situation X, approach Y works best"

### Policy Gradient RL
Use verification scores as rewards:
```python
reward = -1 * cells_different  # Negative distance
# 0 = best (perfect match)
# -5 = 5 cells off
# -15 = 15 cells off
```

## Data Format for Training

### Per-Step Training Example:
```json
{
  "training_example": 1,
  "step": 2,
  "state": {
    "current_grid": [[0,1,2], [3,4,5]],
    "task_description": "...",
    "previous_steps": [...]
  },
  "hypotheticals": [
    {
      "description": "Rotate 90 degrees",
      "grid": [[...], [...]],
      "verification_score": -5,
      "path": "02_hypotheticals/h_01.png"
    },
    {
      "description": "Flip horizontally",
      "grid": [[...], [...]],
      "verification_score": 0,
      "path": "02_hypotheticals/h_02.png"
    }
  ],
  "chosen_transform": {
    "description": "Flip horizontally",
    "grid": [[...], [...]],
    "was_best": true,
    "path": "02_transform.png"
  },
  "expected_output": [[...], [...]]
}
```

## Extracting Training Data

### Python Script:
```python
from arc_visual_solver import ARCVisualSolver
import json

solver = ARCVisualSolver()

def extract_training_data(task_name, training_example=0):
    """Extract all step data for RL training"""
    vizs = solver.list_visualizations(task_name, training_example)
    
    training_data = []
    current_step = None
    
    for viz in vizs:
        metadata = viz['metadata']
        
        if viz['filename'].endswith('_hypotheticals'):
            # Start of new step
            current_step = {
                'hypotheticals': [],
                'step_num': int(viz['filename'].split('_')[0])
            }
        elif 'h_' in viz['filename'] and current_step:
            # Hypothetical
            current_step['hypotheticals'].append({
                'description': metadata.get('description'),
                'grid': metadata.get('grid_data'),
                'path': viz['path']
            })
        elif 'transform' in viz['filename'] and current_step:
            # Transform chosen
            current_step['transform'] = {
                'description': metadata.get('description'),
                'grid': metadata.get('grid_data'),
                'path': viz['path']
            }
            training_data.append(current_step)
            current_step = None
    
    return training_data
```

## Key Advantages for RL

### 1. **Rich Signal**
- Not just "right/wrong" on final output
- Intermediate step feedback
- Multiple alternatives per step
- Verification scores for each alternative

### 2. **Reasoning Pattern Transfer**
- Model learns HOW to think, not just WHAT answer
- "Generate hypotheticals → verify → pick best" pattern
- Generalizes to new tasks

### 3. **Self-Supervised**
- Verification uses expected output (available in training)
- Student model learns to replicate this reasoning
- At inference: no verifier needed, just learned patterns

### 4. **Explainable**
- Each step has descriptions
- Can trace reasoning path
- Understand why model picked each transform

## Example RL Training Loop

```python
# 1. Collect trajectories
for task in training_tasks:
    solver.solve(task)
    trajectories.append(extract_training_data(task))

# 2. Train student model
for trajectory in trajectories:
    for step in trajectory:
        # Input: state
        state = encode_state(step)
        
        # Target: best hypothetical
        best_idx = find_best_hypothetical(step)
        
        # Train
        loss = cross_entropy(
            model(state),
            best_idx
        )
        loss.backward()

# 3. At inference (no verifier needed)
for step in new_task:
    state = encode_state(step)
    action = model(state)  # Learned which approach works
    apply_transformation(action)
```

## Files Generated Per Task

```
visualizations/task_id/training_1/
  01_input.png              # Initial state
  02_hypotheticals/         # Multiple options explored
    h_01.png (metadata: grid_data, description, score via comparison)
    h_02.png 
    h_03.png
  02_transform.png          # Chosen action (after verification)
  03_hypotheticals/         # Next step options
    h_01.png
    h_02.png
  03_transform.png          # Next chosen action
  ...
  XX_model_output.png       # Final result
  YY_actual_output.png      # Ground truth
```

## Next Steps

1. **Run solver on training set**: Generate trajectories
2. **Extract step data**: Parse metadata from all images
3. **Build dataset**: Create input/output pairs
4. **Train student model**: Learn to pick best approach
5. **Test inference**: Run student without verifier
6. **Evaluate transfer**: Test on held-out tasks

The key insight: verification happens BEFORE the model picks the transform, so the model receives the learning signal in real-time and can incorporate it into its reasoning process.
