# Training a Reasoning Model from Step-by-Step Visualizations

## Overview
Train a model to replicate the reasoning pattern: analyze training examples step-by-step, then apply that reasoning to solve test cases.

## The Meta-Learning Setup

### What the Teacher Model Does:
```
Input: 3 training examples + 1 test input
Process:
  For each training example:
    1. Shows input
    2. Shows output (ground truth)
    3. Generates hypotheticals (multiple approaches)
    4. Gets verification (which was closest)
    5. Picks transform (ideally the verified best)
    6. Iterates until reaches output
  
  For test example:
    1. Shows input only
    2. Uses learned pattern from training examples
    3. Generates solution step-by-step
    4. Produces final output
```

### What the Student Model Learns:
```
Given: Same 3 training examples + 1 test input
Learn to:
  1. Recognize patterns from training examples
  2. Generate appropriate hypotheticals
  3. Pick best approaches (learned from verification)
  4. Apply to test case
WITHOUT needing verification at inference!
```

## Training Data Structure

### From One Task:
```json
{
  "task_id": "1ae2feb7",
  "training_examples": [
    {
      "example_num": 0,
      "input": [[0,1,2], [3,4,5]],
      "output": [[5,4,3], [2,1,0]],
      "reasoning_steps": [
        {
          "step": 1,
          "hypotheticals": [
            {
              "description": "Flip horizontally",
              "grid": [[2,1,0], [5,4,3]],
              "verification_score": 0  // ← PERFECT!
            },
            {
              "description": "Rotate 180",
              "grid": [[5,4,3], [2,1,0]],
              "verification_score": 4
            }
          ],
          "chosen_transform": "Flip horizontally",
          "was_verified_best": true  // ← Important for learning!
        }
      ]
    },
    // ... examples 1 and 2 ...
  ],
  "test_example": {
    "input": [[6,7,8], [9,10,11]],
    "model_reasoning": [
      {
        "step": 1,
        "hypotheticals": [...],
        "chosen_transform": "Flip horizontally",
        "reasoning": "Based on pattern from training examples"
      }
    ],
    "predicted_output": [[8,7,6], [11,10,9]],
    "actual_output": [[8,7,6], [11,10,9]],
    "success": true
  }
}
```

## Training Approach 1: Imitation Learning

### Goal: 
Learn to generate the same reasoning steps as the teacher model.

### Architecture:
```python
class ReasoningModel(nn.Module):
    def __init__(self):
        self.encoder = GridEncoder()  # Encode grids to embeddings
        self.pattern_extractor = TransformerEncoder()  # Learn from training examples
        self.hypothesis_generator = TransformerDecoder()  # Generate hypotheticals
        self.selector = AttentionSelector()  # Pick best hypothetical
    
    def forward(self, training_examples, test_input):
        # 1. Encode all training examples
        training_patterns = []
        for ex in training_examples:
            input_emb = self.encoder(ex.input)
            output_emb = self.encoder(ex.output)
            
            # Learn what transforms work for this task
            pattern = self.pattern_extractor(input_emb, output_emb)
            training_patterns.append(pattern)
        
        # 2. Aggregate learned patterns
        task_pattern = aggregate(training_patterns)
        
        # 3. Generate hypotheticals for test input
        test_emb = self.encoder(test_input)
        hypotheticals = self.hypothesis_generator(test_emb, task_pattern)
        
        # 4. Select best hypothetical (learned from verification)
        chosen = self.selector(hypotheticals, task_pattern)
        
        return chosen
```

### Training Loop:
```python
for task in dataset:
    # Get teacher's reasoning traces
    teacher_traces = load_teacher_reasoning(task)
    
    for example_num, example in enumerate(task.training_examples):
        # Get what teacher did for this example
        teacher_steps = teacher_traces[example_num]
        
        # Train model to replicate
        for step in teacher_steps:
            # Input: current state + previous steps
            state = encode_state(example, step.previous_steps)
            
            # Target: which hypothetical teacher picked
            # Weight by verification score
            target_probs = softmax_from_verification_scores(
                step.hypotheticals
            )
            
            # Predict
            student_probs = model.predict_hypothesis(state)
            
            # Loss: KL divergence (student should match teacher's weighted choices)
            loss = kl_divergence(student_probs, target_probs)
            loss.backward()
    
    # Also train on test case
    test_loss = train_test_case(task.test_example)
```

## Training Approach 2: Reinforcement Learning

### Goal:
Learn to pick hypotheticals that minimize error, using verification scores as rewards.

### Reward Signal:
```python
def compute_reward(hypothetical, expected_output):
    """Reward based on how close to expected output"""
    diff = grid_difference(hypothetical.grid, expected_output)
    
    # Exponential reward (heavily favor exact matches)
    if diff == 0:
        return 1.0  # Perfect!
    elif diff <= 2:
        return 0.8  # Very close
    elif diff <= 5:
        return 0.5  # Close
    else:
        return -0.1 * diff  # Penalty for being far off
```

### Policy Gradient Training:
```python
class PolicyModel(nn.Module):
    def forward(self, state):
        """Returns probability distribution over hypotheticals"""
        return self.policy_net(state)

# Training
for task in dataset:
    for example in task.training_examples:
        for step in example.reasoning_steps:
            state = encode_state(step)
            
            # Model picks hypothetical
            action_probs = policy_model(state)
            action_idx = sample(action_probs)
            
            # Get reward from verification
            reward = compute_reward(
                step.hypotheticals[action_idx],
                example.output
            )
            
            # Policy gradient update
            loss = -log(action_probs[action_idx]) * reward
            loss.backward()
```

## Training Approach 3: Sequence-to-Sequence

### Goal:
Treat entire reasoning chain as sequence generation.

### Input Sequence:
```
[TRAIN_INPUT_1] [TRAIN_OUTPUT_1] [TRAIN_INPUT_2] [TRAIN_OUTPUT_2] 
[TRAIN_INPUT_3] [TRAIN_OUTPUT_3] [TEST_INPUT] [GENERATE]
```

### Output Sequence:
```
[HYPOTHESIS_1: "flip"] [HYPOTHESIS_2: "rotate"] [CHOOSE: "flip"]
[APPLY: grid_1] [HYPOTHESIS_1: "scale"] [CHOOSE: "scale"]
[APPLY: grid_2] [FINAL: test_output]
```

### Training:
```python
# Concatenate all reasoning steps into one sequence
input_seq = tokenize([
    train_ex_1.input, train_ex_1.output,
    train_ex_2.input, train_ex_2.output,
    train_ex_3.input, train_ex_3.output,
    test_input
])

target_seq = tokenize([
    step_1.hypotheticals,
    step_1.chosen_transform,
    step_2.hypotheticals,
    step_2.chosen_transform,
    ...,
    final_output
])

loss = cross_entropy(model(input_seq), target_seq)
```

## Key Training Signals

### 1. **Verification Scores** (Most Important!)
```python
# Weight hypotheticals by how good they were
for hyp in step.hypotheticals:
    weight = exp(-hyp.verification_score / temperature)
    # Higher weight for lower score (fewer cells different)
```

### 2. **Whether Chosen Transform Matched Best Hypothetical**
```python
# Binary signal: Did model pick the verified best?
label = 1 if chosen == best_verified else 0
loss = binary_cross_entropy(model_choice, label)
```

### 3. **Final Task Success**
```python
# Did the whole reasoning chain lead to correct answer?
task_reward = 1.0 if predicted == actual else 0.0
# Propagate reward back through all steps
```

### 4. **Step-Level Intermediate Rewards**
```python
# Each step gets reward based on progress toward goal
for step in reasoning_chain:
    step_reward = similarity(step.output, final_goal)
```

## Data Augmentation

### Generate More Training Examples:
```python
# From one task, generate multiple trajectories
for task in original_tasks:
    # Run teacher model multiple times with different seeds
    for seed in range(10):
        set_random_seed(seed)
        trajectory = teacher_model.solve(task)
        # Get different hypotheticals each time
        training_data.append(trajectory)
```

### Synthetic Task Generation:
```python
# Create variations of existing tasks
def augment_task(task):
    variations = []
    
    # Rotate all grids
    variations.append(rotate_task(task, 90))
    
    # Mirror all grids
    variations.append(mirror_task(task))
    
    # Color permutation
    variations.append(permute_colors(task))
    
    return variations
```

## Evaluation

### Metrics to Track:

1. **Hypothesis Quality**
   - Average verification score of generated hypotheticals
   - Does student generate diverse hypotheticals?

2. **Selection Accuracy**
   - How often does student pick the best verified hypothetical?
   - Compare to random selection baseline

3. **Reasoning Transfer**
   - Test on held-out tasks
   - Can student apply learned reasoning patterns?

4. **Final Task Accuracy**
   - Percentage of test cases solved correctly
   - Compare to teacher model

5. **Efficiency**
   - How many steps does student need vs teacher?
   - Can student learn shortcuts?

## Expected Results

### Stage 1: Hypothesis Imitation
```
Student learns to generate similar hypotheticals as teacher
→ Similar diversity and creativity in approaches
→ But may not pick best ones yet
```

### Stage 2: Selection Learning
```
Student learns which hypotheticals work (from verification)
→ Picks verified best more often
→ Improves accuracy on training examples
```

### Stage 3: Pattern Transfer
```
Student recognizes similar patterns across tasks
→ "This looks like a flip task" → generates flip hypotheticals
→ Generalizes to new tasks
```

### Stage 4: Reasoning Shortcuts
```
Student learns efficient reasoning paths
→ Needs fewer hypotheticals than teacher
→ Directly generates likely-best approach
→ Faster inference
```

## Implementation Roadmap

### Phase 1: Data Collection
1. Run teacher model on 100-1000 training tasks
2. Save all reasoning traces with metadata
3. Extract step-by-step data from visualizations

### Phase 2: Baseline Model
1. Train simple model: input → output (no reasoning)
2. Establish baseline accuracy
3. Shows value of reasoning traces

### Phase 3: Reasoning Model
1. Train with hypothesis generation
2. Train with selection (using verification)
3. Compare to baseline

### Phase 4: Transfer Evaluation
1. Test on held-out tasks
2. Measure generalization
3. Analyze failure modes

### Phase 5: Iterative Improvement
1. Identify where model fails
2. Augment training data for those patterns
3. Retrain and evaluate

## Code Example: Full Training Script

```python
# train_reasoning_model.py
import torch
from arc_visual_solver import ARCVisualSolver
from reasoning_model import ReasoningModel

def extract_training_data(task_files):
    """Extract reasoning traces from saved visualizations"""
    solver = ARCVisualSolver()
    training_data = []
    
    for task_file in task_files:
        task_id = os.path.basename(task_file).split('.')[0]
        
        # Get all training examples for this task
        for ex_num in range(4):  # 0-2 training, 3 test
            vizs = solver.list_visualizations(task_id, ex_num)
            
            example_data = {
                'task_id': task_id,
                'example_num': ex_num,
                'steps': []
            }
            
            current_step = None
            for viz in vizs:
                if 'hypotheticals' in viz['filename']:
                    # Start new step
                    current_step = {'hypotheticals': []}
                elif 'h_' in viz['filename']:
                    # Add hypothetical
                    metadata = viz['metadata']
                    current_step['hypotheticals'].append({
                        'grid': metadata['grid_data'],
                        'description': metadata['description'],
                        'score': calculate_score(metadata)
                    })
                elif 'transform' in viz['filename']:
                    # Chosen transform
                    metadata = viz['metadata']
                    current_step['transform'] = {
                        'grid': metadata['grid_data'],
                        'description': metadata['description']
                    }
                    example_data['steps'].append(current_step)
            
            training_data.append(example_data)
    
    return training_data

def train():
    # Load data
    training_data = extract_training_data(task_files)
    
    # Initialize model
    model = ReasoningModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop
    for epoch in range(100):
        for batch in batch_iterator(training_data):
            # Forward pass
            predictions = model(batch)
            
            # Compute loss (weighted by verification scores)
            loss = compute_weighted_loss(predictions, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate
        accuracy = evaluate(model, test_data)
        print(f"Epoch {epoch}: {accuracy:.2%}")

if __name__ == "__main__":
    train()
```

## Conclusion

**Yes, this approach is highly effective for training a reasoning model!**

The key advantages:
1. **Rich supervision**: Verification scores guide learning
2. **Compositional**: Learns step-by-step reasoning
3. **Transferable**: Pattern recognition across tasks
4. **Explainable**: Can trace reasoning process
5. **Efficient**: Student learns shortcuts from teacher

The student model learns the meta-skill: "Given examples, reason step-by-step to solve new cases" - which is exactly what you want for ARC-AGI tasks!
