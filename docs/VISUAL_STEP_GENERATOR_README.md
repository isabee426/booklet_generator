# 🎨 Visual Step Generator - Complete System

**Status:** ✅ READY

## What It Does

Generates ARC puzzle steps using **visual-first reasoning**:
1. Model analyzes ALL training examples (images)
2. Model generates empty output grid
3. Model generates steps one-by-one (visual reasoning → grid modification)
4. Compares to your ground truth booklets
5. Beautiful Streamlit UI to review results

---

## ⚡ Quick Start

### 1. Run Generator

```bash
python scripts/visual_step_generator.py --puzzle 05f2a901
```

**This will:**
- Load ground truth from `visual_traces/05f2a901/`
- Load ARC puzzle data
- Phase 1: Analyze all training examples (images)
- Step 0: Generate empty output grid
- Steps 1-N: Generate each step (3 attempts each)
- Compare to ground truth
- Save everything

### 2. View Results

```bash
streamlit run scripts/view_visual_steps.py
```

**UI Features:**
- ✅ Select puzzle from dropdown
- ✅ Select step (1, 2, 3...)
- ✅ Slide through attempts (◀ ▶ arrows)
- ✅ Generated vs Ground Truth side-by-side
- ✅ Success tags when matches
- ✅ Accuracy scores
- ✅ Difference highlighting

---

## 🔄 Complete Flow

### Phase 1: Transformation Analysis

**Model Receives:**
```
6 IMAGES (3 training examples):
- Training 1: Input → Output
- Training 2: Input → Output  
- Training 3: Input → Output

PROMPT:
"Analyze these examples.
1. What's common/different in inputs?
2. What's common/different in outputs?
3. What's the transformation?
4. Grid size rule?
5. One-sentence summary?"
```

**Model Outputs:**
```
"1. INPUTS: All have color 8 and color 2 objects...
2. OUTPUTS: All same size, objects repositioned...
3. TRANSFORMATION: Color 2 moves toward color 8...
4. GRID SIZE: Same as input (9×9)
5. RULE: Move color 2 until touching color 8"
```

### Step 0: Empty Grid

**Model Receives:**
```
ANALYSIS: [Phase 1 output]
Input size: 9×9

TASK: Generate empty output grid
```

**Model Outputs:**
```
Grid size: 9×9

GRID:
[[0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0],
 ... all zeros]
```

### Step 1: First Transformation

**Model Receives:**
```
IMAGES:
- Image 1: Expected OUTPUT (final goal)
- Image 2: Original INPUT (source objects)

ANALYSIS: [Full Phase 1 analysis]

PREVIOUS STEPS: None

CURRENT GRID: [[0,0,0,...]] (empty)

TASK: Make ONE incremental change
```

**Model Outputs (Attempt 1):**
```
Description: Copy the color 8 object from input to output at same position

GRID:
[[0,0,0,0,0,0,0,0,0],
 [0,0,0,8,8,0,0,0,0],
 [0,0,0,8,8,0,0,0,0],
 ...]
```

**Code Compares to Ground Truth:**
```
Accuracy: 92%
Differences: 
  (5,3): got 0 expected 8
  (5,4): got 0 expected 8
```

**If Success (>99%):** Continue to Step 2  
**If Fail:** Attempt 2 (max 3 attempts)  
**If All 3 Fail:** Use ground truth, mark as fallback

### Step 2: Next Transformation

**Model Receives:**
```
IMAGES:
- Image 1: Expected OUTPUT
- Image 2: Original INPUT
- Image 3: Step 1 result

ANALYSIS: [Full Phase 1 analysis]

PREVIOUS STEPS:
Step 1: Copy the color 8 object from input...

CURRENT GRID: [[0,0,0,8,8,...]] (result from Step 1)

TASK: Next incremental change
```

And so on...

---

## 📊 Streamlit UI Layout

### **Main View:**

```
┌─────────────────────────────────────────────────────────────────┐
│ Puzzle: 05f2a901                                                │
│ ✅ SUCCESS! Perfect Match (Step 3)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Step [1 ▼]    ◀ Attempt 1 of 3 ▶                              │
│                                                                  │
├──────────────────────────┬──────────────────────────────────────┤
│  🤖 GENERATED            │  ✅ GROUND TRUTH                     │
│  (Attempt 1)             │                                       │
│  ┌──────────────┐        │  ┌──────────────┐                   │
│  │  [Grid PNG]  │        │  │  [Grid PNG]  │                   │
│  │              │        │  │              │                   │
│  └──────────────┘        │  └──────────────┘                   │
│                          │                                       │
│  Description:            │  Description:                         │
│  "Copy color 8 object    │  "Maintain the dimensions           │
│   from input to output"  │   of the input grid"                │
│                          │                                       │
│  ✓ Valid grid            │  92% Accuracy                        │
│  [Show grid array ▼]     │  [Show grid array ▼]                │
│                          │                                       │
└──────────────────────────┴──────────────────────────────────────┘

Differences: (5,3): color 0 → color 8, (5,4): color 0 → color 8

──────────────────────────────────────────────────────────────────
         ⬅️ Previous Step          Next Step ➡️
```

### **Sidebar:**

```
📁 Select Puzzle
   ▼ 05f2a901

───────────────
📊 All Steps Summary

✅ Step 1: 92%
✅ Step 2: 100%
⚠️ Step 3: 78%
❌ Step 4: 45% 🔄

🔄 = Used GT fallback
```

### **Navigation:**

- **Dropdown:** Select which step (1, 2, 3...)
- **◀ ▶ Arrows:** Slide through attempts for current step
- **⬅️ ➡️ Buttons:** Previous/Next step
- **Expandable:** Show grid arrays

---

## 📁 Output Structure

```
visual_step_results/05f2a901/
├── results.json                    ← All data
├── step_01_attempt_1.png          ← First attempt
├── step_01_attempt_2.png          ← Second attempt (if needed)
├── step_01_attempt_3.png          ← Third attempt (if needed)
├── step_01_ground_truth.png       ← Your manual work
├── step_01_final.png              ← What was used for next step
├── step_02_attempt_1.png
├── step_02_attempt_2.png
├── step_02_ground_truth.png
├── step_02_final.png
...
```

### **results.json Structure:**

```json
{
  "puzzle_id": "05f2a901",
  "timestamp": "2025-11-04T...",
  "model": "gpt-5-mini",
  "phase1_analysis": "1. INPUTS: All have color 8 and color 2...",
  "steps": [
    {
      "step_num": 1,
      "attempts": [
        {
          "attempt": 1,
          "response": "Full model response...",
          "description": "Copy color 8 object...",
          "grid": [[0,0,0,8,8,...]],
          "success": true,
          "errors": []
        },
        {
          "attempt": 2,
          "response": "...",
          "success": false,
          "errors": ["Invalid colors"]
        }
      ],
      "used_ground_truth": false,
      "final_grid": [[0,0,0,8,8,...]],
      "final_description": "Copy color 8 object...",
      "ground_truth": {
        "grid": [[0,0,0,8,8,...]],
        "description": "Maintain dimensions..."
      },
      "comparison": {
        "match": false,
        "accuracy": 0.92,
        "differences": [
          {"position": [5,3], "generated": 0, "expected": 8}
        ]
      }
    }
  ]
}
```

---

## 🎯 What Makes This Visual-First

### **1. Images are Primary Input**

Model sees PNG grids FIRST:
- Visual context
- Spatial relationships
- Object patterns

### **2. Grid for Precision**

Then sees grid array:
- Exact color values
- Precise modifications
- No ambiguity

### **3. Color Consistency**

- Extracts valid colors from puzzle: `{0, 2, 8}`
- Enforces in prompt: "Only use [0, 2, 8]"
- Validates before accepting
- Describes as "color 0", "color 2" (not "black", "red")

### **4. Analysis Guides Everything**

Phase 1 analysis sent to EVERY step:
- What the rule is
- What objects to work with
- What the goal is

### **5. Ground Truth Fallback**

If all 3 attempts fail:
- Don't give up
- Use your ground truth step
- Continue from there
- Mark as "used_gt" for analysis

---

## 🚀 Usage Examples

### Test One Puzzle

```bash
python scripts/visual_step_generator.py --puzzle 05f2a901
```

### View Results

```bash
streamlit run scripts/view_visual_steps.py
```

### Test Multiple Puzzles

```bash
# Create batch runner
for puzzle in 05f2a901 007bbfb7 017c7c7b; do
    python scripts/visual_step_generator.py --puzzle $puzzle
done
```

### Different Model

```bash
python scripts/visual_step_generator.py --puzzle 05f2a901 --model gpt-4o
```

---

## 📊 Research Analysis

### Metrics Collected

**Per Step:**
- Number of attempts needed (1, 2, or 3)
- Success/failure
- Accuracy vs ground truth
- Specific differences

**Per Puzzle:**
- Total steps
- Perfect matches
- Average accuracy
- Ground truth fallback rate

### Questions This Answers

1. **Can visual reasoning solve ARC?**
   - % of steps model gets right
   - % needing ground truth fallback

2. **What's hard for visual reasoning?**
   - Which steps fail most?
   - What types of transformations challenging?

3. **How close to human performance?**
   - Accuracy vs your manual booklets
   - Where does model diverge?

---

## ✅ System Features

- ✅ Visual-first (images before grids)
- ✅ Color consistency enforced ("color 0", "color 1"...)
- ✅ Analysis provided to every step
- ✅ Output image shown (the goal)
- ✅ All previous steps shown
- ✅ 3 retry attempts
- ✅ Ground truth fallback
- ✅ Complete attempt history
- ✅ Streamlit UI with slides
- ✅ Uses your arc_visualizer.py
- ✅ Ground truth from your visual_traces

---

## 🎯 Ready to Test!

```bash
# Generate
python scripts/visual_step_generator.py --puzzle 05f2a901

# View
streamlit run scripts/view_visual_steps.py
```

**See how AI compares to your manual work!** 🚀





