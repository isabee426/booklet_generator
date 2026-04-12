# Multi-Example Booklet Systems

Two approaches to process ALL training examples in an ARC puzzle, not just the first one.

---

## 🔄 Option 1: Iterative Refinement

**File:** `arc-booklet-refiner.py`

**Concept:** One booklet that evolves across all training examples

### How It Works:
```
Example 1 → Generate initial steps
Example 2 → Apply steps, refine if wrong
Example 3 → Apply steps, refine if wrong
Example 4 → Apply steps, refine if wrong
Example 5 → Apply steps, refine if wrong
Final → Refined steps ready for test
```

### Run It:
```bash
cd booklets_ARCAGI

# Set API key
$env:OPENAI_API_KEY="sk-..."

# Run refiner on a task
python arc-booklet-refiner.py ..\saturn-arc\ARC-AGI-2\ARC-AGI-1\data\training\00d62c1b.json
```

### Output:
```
refined_booklets/
└── 00d62c1b_refined/
    ├── refinement_meta.json    # Complete refinement history
    └── README.txt               # Human-readable summary
```

### What You Get:
- **Refinement History:** Shows how steps evolved across examples
- **Version Tracking:** See what changed and when
- **Success Metrics:** Which examples were solved successfully
- **Final Steps:** The best refined version after all training

---

## 🎭 Option 2: Ensemble Synthesis

**File:** `arc-booklet-ensemble.py`

**Concept:** Multiple booklets, one per example, then find commonalities

### How It Works:
```
Example 1 → Booklet 1 (Steps A, B, C)
Example 2 → Booklet 2 (Steps A, D, E)
Example 3 → Booklet 3 (Steps A, B, F)
Example 4 → Booklet 4 (Steps A, G, H)
Example 5 → Booklet 5 (Steps A, B, I)

Synthesis → Common pattern: Steps A, B appear in most
```

### Run It:
```bash
cd booklets_ARCAGI

# Set API key
$env:OPENAI_API_KEY="sk-..."

# Run ensemble on a task
python arc-booklet-ensemble.py ..\saturn-arc\ARC-AGI-2\ARC-AGI-1\data\training\00d62c1b.json
```

### Output:
```
ensemble_booklets/
└── 00d62c1b_ensemble/
    ├── example_1_booklet/      # Full booklet for example 1
    │   ├── input.png
    │   ├── target_output.png
    │   ├── step_000.png
    │   └── metadata.json
    ├── example_2_booklet/      # Full booklet for example 2
    ├── example_3_booklet/      # Full booklet for example 3
    ├── example_4_booklet/      # Full booklet for example 4
    ├── example_5_booklet/      # Full booklet for example 5
    ├── synthesis_meta.json     # Synthesis results
    ├── comparison.html         # Visual comparison page
    └── README.txt              # Summary
```

### What You Get:
- **Individual Booklets:** Complete step-by-step for each example
- **Common Pattern:** Steps that appear across all booklets
- **Visual Comparison:** HTML page comparing all approaches
- **Diversity Analysis:** See different solving strategies

---

## 📊 Comparison

| Feature | Option 1: Refiner | Option 2: Ensemble |
|---------|-------------------|-------------------|
| **Output** | One refined booklet | Multiple booklets + synthesis |
| **Approach** | Iterative evolution | Independent + merge |
| **Best For** | Finding single best rule | Seeing all perspectives |
| **Output Size** | Smaller | Larger (one booklet per example) |
| **Processing** | Sequential | Can be parallelized |
| **Insights** | Shows learning process | Shows diversity of approaches |

---

## 💡 When to Use Which?

### Use **Refiner** (Option 1) when:
- ✅ You want to see how understanding evolves
- ✅ You need one final set of steps
- ✅ You want to track changes across examples
- ✅ Storage space is limited

### Use **Ensemble** (Option 2) when:
- ✅ You want to see all possible interpretations
- ✅ You need to compare different approaches
- ✅ You want complete booklets for each example
- ✅ You're analyzing puzzle ambiguity

---

## 🎯 Examples

### Task with 5 training examples:

**Refiner Output:**
```
Example 1: Generated "Fill green-enclosed areas with yellow"
Example 2: Refined to "Fill fully-enclosed green areas with yellow"
Example 3: Stable (no changes)
Example 4: Stable (no changes)
Example 5: Stable (no changes)

Final Rule: "Fill fully-enclosed green areas with yellow"
```

**Ensemble Output:**
```
Booklet 1: "Fill enclosed area with yellow"
Booklet 2: "Fill hole inside green border with yellow"
Booklet 3: "Find green frame, fill interior with yellow"
Booklet 4: "Detect green rectangle, fill center with yellow"
Booklet 5: "Fill background inside green outline with yellow"

Common Pattern: "Fill interior/enclosed areas with yellow"
```

---

## ⚙️ Advanced Usage

### Batch Process Multiple Tasks

**Refiner:**
```bash
# Process multiple tasks
for file in ..\saturn-arc\ARC-AGI-2\ARC-AGI-1\data\training\*.json
do
    python arc-booklet-refiner.py "$file"
done
```

**Ensemble:**
```bash
# Process with custom output directory
python arc-booklet-ensemble.py task.json my_custom_dir
```

### View Results

**Refiner:**
- Read `README.txt` for summary
- Check `refinement_meta.json` for programmatic access

**Ensemble:**
- Open `comparison.html` in browser for visual comparison
- Browse `example_N_booklet/` folders individually
- Use Streamlit booklet viewer on individual booklets

---

## 🔧 Technical Notes

### Both Tools:
- Call `arc-booklet-generator.py` internally
- Require OpenAI API key
- Generate temporary task files
- Clean up after themselves

### Timeouts:
- Each example has 10 minute timeout
- 5 examples = up to 50 minutes total
- Adjust in code if needed

### Error Handling:
- Continues if one example fails
- Reports which examples succeeded/failed
- Saves partial results

---

## 📝 Output Format

### Refiner Metadata:
```json
{
  "task_name": "00d62c1b",
  "refinement_history": [
    {
      "example_number": 1,
      "steps": ["step1", "step2"],
      "success": true,
      "total_steps": 5
    }
  ],
  "final_steps": ["refined step1", "refined step2"]
}
```

### Ensemble Metadata:
```json
{
  "task_name": "00d62c1b",
  "booklets": [
    {
      "example_number": 1,
      "booklet_path": "example_1_booklet",
      "steps": ["step1", "step2"],
      "success": true
    }
  ],
  "common_steps": ["common step1"]
}
```

---

## 🚀 Quick Start

```bash
# Option 1: Iterative Refinement
python arc-booklet-refiner.py ../saturn-arc/ARC-AGI-2/ARC-AGI-1/data/training/00d62c1b.json

# Option 2: Ensemble Synthesis
python arc-booklet-ensemble.py ../saturn-arc/ARC-AGI-2/ARC-AGI-1/data/training/00d62c1b.json

# View ensemble results
start ensemble_booklets/00d62c1b_ensemble/comparison.html
```

Both approaches complement each other - use the refiner to find the best single solution, and the ensemble to understand the full problem space!

