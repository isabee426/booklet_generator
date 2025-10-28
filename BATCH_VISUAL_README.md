# Batch Visual Booklet Generator

## 🎯 Core Concept

Instead of learning sequentially (one example at a time), the **Batch Visual** approach shows the AI the **entire problem space upfront** to generate universal steps from the start.

## 📊 How It Works

### Traditional Sequential Approach (Refiner)
```
Example 1 → Generate steps
Example 2 → Test steps → Fail → Refine
Example 3 → Test steps → Fail → Refine
Example 4 → Test steps → Success
```
**Problem**: Steps get patched incrementally, may overfit to early examples

### Batch Visual Approach (This)
```
Show: Example 1 (input + output) + ALL other inputs simultaneously
↓
Analyze visual similarity across ALL inputs
↓
Generate universal steps considering entire problem space
↓
Validate on all training examples
↓
Refine if needed (seeing all failures at once)
```

## 🎨 Five Phases

### Phase 1: Visual Similarity Analysis
- Shows AI:
  - Example 1: Input → Output (solved)
  - Example 2: Input only
  - Example 3: Input only
  - Example N: Input only
- AI analyzes:
  - What's **CONSTANT** across all inputs?
  - What **VARIES** across inputs?
  - What general rule would work for ALL?

### Phase 2: Generate Universal Steps
- Shows all input/output pairs
- AI generates 3-7 steps that are:
  - **General from the start** (not specific to Example 1)
  - **Handle variation** ("for each object" not "for the 3 objects")
  - **Visual and high-level** ("move each blue shape" not coordinates)

### Phase 3: Validate on All Training Examples
- Applies universal steps to EVERY training example
- Identifies which examples pass/fail

### Phase 4: Batch Refinement (if needed, up to 3 iterations)
- Shows ALL failed examples simultaneously
- AI sees:
  - Input → Your Output → Expected Output (for each failure)
  - Visual differences across all failures
- Refines steps to fix all failures at once

### Phase 5: Test Cases
- Applies final universal steps to test inputs
- 3 attempts per test case

## 🚀 Why This May Improve Generalization

### 1. **Upfront Problem Space View**
- AI sees all variations before committing to steps
- Can identify the true universal pattern
- Avoids premature optimization

### 2. **Prevents Sequential Bias**
```
Sequential: "Works for Ex1... works for Ex2... oh, Ex3 breaks it, patch..."
Batch:      "What pattern explains ALL of these at once?"
```

### 3. **Visual Similarity Analysis**
- Explicitly reasons about commonalities vs differences
- Forces high-level thinking before generating steps
- Identifies what's essential vs incidental

### 4. **Batch Refinement**
- When refining, sees ALL failures simultaneously
- Can identify systemic issues, not just individual failures
- Fixes root cause, not symptoms

### 5. **Cognitive Alignment**
- Mimics how humans solve ARC puzzles:
  - Look at all examples first
  - Find the common pattern
  - Generate solution that works for all

## 📈 Expected Benefits

| Aspect | Sequential | Batch Visual |
|--------|-----------|--------------|
| Generalization | Incremental patches | Universal from start |
| Overfitting Risk | High (early examples) | Low (sees all upfront) |
| Step Quality | May be example-specific | Designed for variation |
| Refinement | One failure at a time | All failures together |
| Cognitive Load | Lower (one example) | Higher (all examples) |
| Pattern Recognition | Sequential discovery | Holistic analysis |

## 🎯 When to Use This Approach

**Best for:**
- Puzzles with clear visual patterns across examples
- Tasks where examples vary in size/count but follow same rule
- Cases where sequential learning might overfit

**Consider Sequential for:**
- Puzzles where examples build on each other
- Tasks with very different example types
- When computational cost is a concern (batch uses more tokens)

## 🔧 Usage

```bash
# Activate environment
.\arc-solver-env\Scripts\Activate.ps1

# Run batch visual generator
python arc-booklet-batch-visual.py <task_file> [output_dir]

# Example
python arc-booklet-batch-visual.py ..\saturn-arc\ARC-AGI-2\ARC-AGI-1\data\training\00d62c1b.json

# Output goes to batch_visual_booklets/
```

## 📁 Output Structure

```
batch_visual_booklets/
  <task_name>_batch/
    batch_results.json    # Full results including similarity analysis
    README.txt            # Human-readable summary
```

## 🧠 Theoretical Foundation

This approach is inspired by:
1. **Human cognitive strategy** - people look at all examples before solving
2. **Transfer learning** - learn general patterns, not specific instances
3. **Visual reasoning** - pattern matching across multiple inputs
4. **Meta-learning** - learning to learn across problem variations

## 🔬 Experimental Hypothesis

**Hypothesis**: By seeing all training inputs simultaneously, the AI will:
- Generate more abstract, generalizable steps
- Avoid overfitting to early examples
- Identify the true underlying transformation rule
- Produce higher test accuracy with fewer refinement iterations

**Test this by comparing**:
- Batch Visual vs Sequential Refiner
- On same set of 20 puzzles
- Measure: Training accuracy, Test accuracy, Refinement iterations needed

