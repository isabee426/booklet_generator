# Training Data Collection Improvements

## Summary of Changes

This document describes the improvements made to collect high-quality training data for meta-learning / imitation learning.

---

## 1. Verification Metadata Added to Hypotheticals

### What Was Added:
Every hypothetical now gets marked with:
- `verification_score`: How many cells different from expected output (-1 if dimension mismatch)
- `is_chosen`: Boolean indicating if this was selected as the best hypothesis
- `competing_hypotheticals`: List of all other hypotheticals in the same batch with their scores
- `verified_at`: Timestamp of verification

### Implementation:
```python
def add_verification_metadata_to_image(image_path, verification_score, is_chosen, all_hypotheticals)
```

### Why This Matters:
- Student model can learn which hypotheticals were good/bad
- Provides ranking signal for training
- Enables contrastive learning (good vs bad hypotheses)
- Creates implicit reward model

---

## 2. Output Dimension Consistency Enforced

### What Changed:
1. **Validation at generation time**: Rejects hypotheticals/transforms with wrong dimensions
2. **Clear error messages**: Tells model exactly what dimensions are required and why
3. **Dimension reminders in prompts**: All phases explicitly state output dimensions

### Key Messages:
```
"Output dimensions are ALWAYS consistent across all examples in ARC tasks"
"Your output MUST be {width}x{height} (same as training examples)"
"Based on training examples, all outputs have dimensions {width}x{height}"
```

### Why This Matters:
- Prevents infinite verification scores from dimension mismatches
- Teaches model that output dimensions are task-invariant
- Forces model to learn transformation, not dimension changes
- Matches how humans solve ARC (output size is constrained)

---

## 3. Training Data Structure

### What Each Hypothetical Contains:

#### Core Metadata:
- `description`: Full reasoning text
- `step_description`: "Step 2 - Hypothetical 3: [description]"
- `visualization_type`: "hypothetical" or "transform"
- `grid_data`: JSON-encoded grid array
- `grid_dimensions`: "width x height"

#### Training-Specific Metadata:
- `verification_score`: Cells different from expected output
- `is_chosen`: Was this selected as best?
- `competing_hypotheticals`: All other options with scores
- `verified_at`: Timestamp

#### Context Metadata:
- `step_number`: Which step in the solution process
- `hypothetical_number`: Which hypothetical in this step
- `training_example`: Which training example (0, 1, 2...)
- `task`: Task ID
- `timestamp`: When created

---

## 4. Training Data Format for ML

### Extracted Format (JSONL):
```json
{
  "task_id": "abc123",
  "training_example": 1,
  "step": 2,
  "context": {
    "seen_examples": [
      {"input": [[...]], "output": [[...]]}
    ],
    "previous_steps": ["01_input.png", "02_transform.png"]
  },
  "current_state": {
    "image": "02_transform.png",
    "grid": [[...]]
  },
  "hypotheticals": [
    {
      "image": "03_hypotheticals/h_01.png",
      "description": "Group by color",
      "grid": [[...]],
      "score": 15,
      "chosen": false
    },
    {
      "image": "03_hypotheticals/h_02.png",
      "description": "Group and center",
      "grid": [[...]],
      "score": 2,
      "chosen": true
    }
  ],
  "expected_output": [[...]]
}
```

---

## 5. How Student Model Uses This Data

### Training Objective:
Learn `P(hypotheticals, chosen_action | context, query)`

### What Student Learns:
1. **Hypothesis Generation**: How to create diverse, reasonable hypotheses
2. **Evaluation Strategy**: Which hypotheses tend to be better
3. **Decision Making**: How to choose best hypothesis without external verifier
4. **Reasoning Patterns**: Common approaches that work across tasks

### Key Insight:
Student doesn't need explicit verification at inference because it learns:
- Characteristics of good hypotheses (from `is_chosen` labels)
- Relative quality (from `verification_score` rankings)
- Consistency patterns (which hypotheses explain all examples)

---

## 6. Meta-Learning Setup

### Input at Test Time (Few-Shot):
```
Context: 3-4 training examples (input → output pairs)
Query: New input (no output shown)
Task: Generate hypotheses and choose best
```

### What Model Does:
```python
1. Analyze context examples
   - Identify invariants
   - Find transformation patterns
   
2. Generate hypotheses for query
   - Apply variations of learned patterns
   - Create 3-5 diverse approaches
   
3. Evaluate hypotheses (internally)
   - Check consistency with context
   - Estimate confidence
   
4. Choose best hypothesis
   - Pick most consistent with all context examples
   - Output final grid
```

### Generalization to Novel Tasks:
- ✅ Learns reasoning strategies, not specific transformations
- ✅ Compositional: can combine learned primitives
- ✅ Consistent with few-shot learning paradigm
- ✅ Matches test-time setup (given examples, solve new case)

---

## 7. Critical Success Factors

### Data Collection:
- ✅ Diverse tasks (100s-1000s)
- ✅ Full reasoning chains captured
- ✅ Verification scores for all hypotheses
- ✅ Chosen actions marked
- ✅ Rich context (previous steps, examples)

### Training:
- Need multi-task objective:
  - Generate diverse hypotheses
  - Rank hypotheses correctly
  - Choose best hypothesis
  - Generate correct grids

### Model Architecture:
- Vision encoder (for grid images)
- Reasoning module (transformer)
- Grid generator (decoder)
- Implicit evaluation (learned from data)

---

## 8. Next Steps

### Data Extraction:
Create `extract_training_data.py` to:
1. Parse all visualization metadata
2. Build context for each step
3. Extract hypothetical batches
4. Create JSONL training file

### Model Training:
1. Fine-tune vision-language model on collected data
2. Train with multi-task objective
3. Evaluate on held-out ARC tasks
4. Compare to GPT baseline

### Evaluation:
- Test on novel ARC tasks (never seen in training)
- Compare reasoning quality to GPT
- Measure if model can solve without verification
- Analyze which reasoning patterns transferred

---

## Summary

This implementation collects **distillation data** from GPT's reasoning process:
- **Teacher (GPT)**: Generates hypotheses WITH verification feedback
- **Student**: Learns to replicate behavior WITHOUT verification
- **Training**: Supervised learning from GPT's demonstrations
- **Goal**: Internalize reasoning strategies that generalize to novel tasks

The key innovation: **Learning meta-reasoning patterns** (how to solve ARC tasks) rather than specific transformations (what to do for each task).
