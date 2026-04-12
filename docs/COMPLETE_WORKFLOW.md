# Complete Workflow: ARC Visual Solver with Verification

## System Overview

The ARC Visual Solver uses a **hypothesis-verify-select** loop to solve puzzles step-by-step, generating rich training data for reinforcement learning.

---

## 🔄 Complete Workflow

### Phase 1: Task Setup
```
1. Load ARC task JSON file
   - Contains: 3-4 training examples (input/output pairs)
   - Contains: 1+ test examples (input only, need to predict output)

2. Initialize solver
   - Set up OpenAI API client
   - Create visualization directories
   - Reset conversation history
```

### Phase 2: Training Examples Analysis

For each training example (usually 3):

#### Step A: Show Input/Output Pair
```python
Training Example 1:
├─ Input: [[0,1,2], [3,4,5]]
├─ Output: [[5,4,3], [2,1,0]]  ← Ground truth (expected output)
└─ Set: self.current_expected_output = output  # For verification
```

#### Step B: Model Analyzes Transformation

**The model reasons step-by-step:**

```
Step 1: Initial Analysis
├─ Model: "I think this might involve flipping..."
└─ Generates hypotheticals

Step 2: Hypothesis Generation (Model creates multiple options)
├─ Hypothetical 1: type='hypothetical', "Flip horizontally"
│   └─ Grid: [[2,1,0], [5,4,3]]
│   └─ Saved: 02_hypotheticals/h_01.png (with metadata)
│
├─ Hypothetical 2: type='hypothetical', "Flip vertically"
│   └─ Grid: [[3,4,5], [0,1,2]]
│   └─ Saved: 02_hypotheticals/h_02.png
│
├─ Hypothetical 3: type='hypothetical', "Rotate 180°"
│   └─ Grid: [[5,4,3], [2,1,0]]
│   └─ Saved: 02_hypotheticals/h_03.png
│
└─ ... (can create more)
```

#### Step C: 🔍 AUTOMATIC VERIFICATION (Happens Between Hypotheticals and Transform)

**System intercepts immediately after hypotheticals are created:**

```python
# Tracks all hypotheticals created in this iteration
hypotheticals_created_this_iteration = [
    (grid1, "Flip horizontally", path1),
    (grid2, "Flip vertically", path2),
    (grid3, "Rotate 180°", path3)
]

# For each hypothetical:
for hyp_grid, hyp_desc, hyp_path in hypotheticals_created_this_iteration:
    # Compare to expected output
    diff = calculate_grid_difference(hyp_grid, expected_output)
    # diff = number of cells that are different
    
# Results:
# Hypothetical 1: 4 cells different
# Hypothetical 2: 6 cells different
# Hypothetical 3: 0 cells different ← PERFECT!
```

**Verification Feedback Generated:**
```
📊 VERIFICATION RESULTS:
I compared your 3 hypotheticals to the expected output:

🔸 CLOSE (4 cells off): Flip horizontally
❌ OFF (6 cells off): Flip vertically
✅ PERFECT: Rotate 180°

🎯 One of your hypotheticals is PERFECT! Consider using: Rotate 180°
```

#### Step D: Feedback Injection

**System injects verification back into conversation:**
```python
call_params["input"].append({
    "role": "user",
    "content": verification_msg
})

# Model's next API call includes this verification feedback
# Model can now see which hypothetical was best BEFORE picking transform
```

#### Step E: Transform Selection (Model's Decision)

**Model receives verification and responds:**
```
Model: "Based on verification, Rotate 180° was perfect. I'll use that."

Creates: type='transform', "Rotate 180°"
└─ Grid: [[5,4,3], [2,1,0]]
└─ Saved: 02_transform.png
```

#### Step F: Next Step (If Needed)

**If solution isn't complete yet:**
```
Model: "Now I need to apply color mapping..."

Step 3: New Hypotheticals
├─ Hypothetical 1: "Map 0→9"
├─ Hypothetical 2: "Map odd→even"
└─ Hypothetical 3: "Invert colors"

→ Verification runs again
→ Model picks best as transform
→ Continue until output matches
```

### Phase 3: Test Example (Final Challenge)

**Apply learned pattern to new input:**

```
Test Input: [[6,7,8], [9,10,11]]
Expected Output: [[11,10,9], [8,7,6]]  ← Set for verification

Model reasoning:
Step 1: Generate hypotheticals
├─ "Apply same rotation pattern from training"
├─ "Apply flip pattern"
└─ "Combination approach"

→ Verification: Shows which is closest to expected
→ Model: Picks best as transform
→ Final output generated
```

---

## 🎯 Key Features

### 1. Real-Time Verification
- Happens **BETWEEN** hypothetical generation and transform selection
- Model **sees feedback** before making decision
- Enables learning which approaches work

### 2. Metadata Storage
Every visualization PNG stores:
```json
{
    "description": "Rotate 180 degrees",
    "visualization_type": "hypothetical",
    "grid_data": [[5,4,3], [2,1,0]],
    "grid_dimensions": "2x3",
    "timestamp": "2025-10-12T14:30:45",
    "step_details": {
        "iteration": 5,
        "training_example": 0,
        "task": "1ae2feb7"
    }
}
```

### 3. File Organization
```
visualizations/1ae2feb7/training_1/
├─ 01_input.png                    # Starting state
├─ 02_hypotheticals/               # Step 1 exploration
│  ├─ h_01.png (metadata: grid, desc, score)
│  ├─ h_02.png
│  └─ h_03.png
├─ 02_transform.png                # Step 1 chosen (after verification)
├─ 03_hypotheticals/               # Step 2 exploration
│  ├─ h_01.png
│  └─ h_02.png
├─ 03_transform.png                # Step 2 chosen
├─ XX_model_output.png             # Final prediction
└─ YY_actual_output.png            # Ground truth
```

---

## 📊 Training Data Generated

### Per Step:
```json
{
  "step": 2,
  "state": {
    "current_grid": [[0,1,2], [3,4,5]],
    "previous_steps": [...]
  },
  "hypotheticals": [
    {
      "description": "Flip horizontally",
      "grid": [[2,1,0], [5,4,3]],
      "verification_score": 4  // ← How far from expected
    },
    {
      "description": "Rotate 180°",
      "grid": [[5,4,3], [2,1,0]],
      "verification_score": 0  // ← PERFECT!
    }
  ],
  "verification_feedback": "✅ PERFECT: Rotate 180°",
  "chosen_transform": {
    "description": "Rotate 180°",
    "grid": [[5,4,3], [2,1,0]]
  },
  "chose_best": true  // Model picked the 0-score option
}
```

---

## 🔬 How This Enables RL Training

### The Learning Signal

**State**: Current grid + task context
```
Input: [[0,1,2], [3,4,5]]
Context: "From training examples, learned this is a flip/rotate task"
```

**Actions**: Multiple hypothetical approaches
```
Action 1: Flip horizontally
Action 2: Flip vertically  
Action 3: Rotate 180°
```

**Rewards**: Verification scores (lower is better)
```
Reward 1: -4  (4 cells different)
Reward 2: -6  (6 cells different)
Reward 3: 0   (perfect match!) ← Best reward
```

**Policy**: Which action did model choose?
```
Model chose: Action 3 (Rotate 180°)
Was it optimal? YES (matched best reward)
```

### Student Model Training

**Goal**: Learn to pick good hypotheticals WITHOUT verification

**Training Loop**:
```python
for step in training_data:
    # Input: State
    state = encode(step.current_grid, step.context)
    
    # Target: Probability distribution favoring low-score hypotheticals
    target_probs = softmax([
        -hyp.verification_score for hyp in step.hypotheticals
    ])
    # [4, 6, 0] → [-4, -6, 0] → [low, lower, HIGH]
    # Student learns: "In this state, approach 3 is best"
    
    # Predict
    predicted_probs = student_model(state)
    
    # Loss: Cross-entropy
    loss = cross_entropy(predicted_probs, target_probs)
    loss.backward()
```

**Result**: Student internalizes verification function
- Learns which hypotheticals get good scores
- Picks best approaches without needing to verify
- Applies reasoning pattern to new tasks

---

## 🚀 The Complete Cycle

```
1. TASK LOADED
   └─ Training examples + Test example

2. FOR EACH TRAINING EXAMPLE:
   ├─ Show input/output to model
   ├─ Set expected_output for verification
   │
   └─ REASONING LOOP (repeat until solution found):
       │
       ├─ Model generates 2-5 hypotheticals
       │  └─ Saved: XX_hypotheticals/h_01.png, h_02.png, ...
       │
       ├─ 🔍 VERIFICATION (automatic)
       │  ├─ Compare each to expected output
       │  ├─ Calculate differences
       │  └─ Generate feedback message
       │
       ├─ Feedback injected to model
       │  └─ Model sees: "✅ PERFECT: Hypothesis 3"
       │
       ├─ Model picks transform
       │  └─ Saved: XX_transform.png
       │
       └─ Repeat with transform as new state

3. FOR TEST EXAMPLE:
   ├─ Model applies learned pattern
   ├─ Verification still runs (if output available)
   └─ Final prediction generated

4. TRAINING DATA EXTRACTED:
   ├─ All hypotheticals with scores
   ├─ All transforms chosen
   └─ Complete reasoning traces

5. STUDENT MODEL TRAINED:
   ├─ Learns from verification scores
   ├─ Internalizes "what works"
   └─ Can reason without verification!
```

---

## 💡 Key Insights

### Why This Works:

1. **Exploration**: Multiple hypotheticals = diverse approaches explored
2. **Feedback**: Verification = immediate signal on quality
3. **Learning**: Model sees what works before deciding
4. **Transferable**: Student learns patterns, not task-specific solutions
5. **Progressive**: Each step builds on previous best choice

### What Makes It Powerful:

- **Rich Training Signal**: Not just right/wrong, but degree of correctness
- **Step-Level Learning**: Learn good reasoning at each step
- **Self-Supervised**: Verification uses available ground truth
- **Generalizable**: Student learns meta-reasoning patterns

---

## 📈 Expected Performance

### Teacher Model (with verification):
- Explores multiple approaches
- Gets feedback on quality
- Picks best with guidance
- **Result**: ~70-80% task success

### Student Model (without verification):
- Learns from teacher's exploration
- Internalizes verification function
- Picks best from learned patterns
- **Result**: ~60-75% task success (no verification needed!)

### Key Achievement:
Student learns to **reason like the teacher** without needing the verifier at inference time!

---

## 🛠️ To Run:

```bash
# Generate training data
python arc_visual_solver.py "path/to/task.json"

# Extract for training
python extract_training_data.py

# Train student model
python train_student_model.py --data arc_training_data.jsonl

# Evaluate student
python eval_student.py --model student.pt --test test_tasks.json
```

The student model then becomes a fast, self-contained reasoner that doesn't need the verification loop!
