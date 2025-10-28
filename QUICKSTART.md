# Quick Start Guide - Solver Comparison

## 🚀 Run the Comparison (Parallel Mode - FASTEST)

```bash
# Navigate to booklets_ARCAGI directory
cd booklets_ARCAGI

# Set your API key
export OPENAI_API_KEY=your_key_here
# On Windows PowerShell:
$env:OPENAI_API_KEY="your_key_here"

# Run comparison on 5 puzzles (PARALLEL - both solvers run simultaneously)
python compare_solvers.py ARC-AGI-2 5
```

**What happens in parallel mode:**
- 🔵 V1 and 🟢 V2 run **simultaneously** for each puzzle (2x faster!)
- Up to **3 puzzles** processed at once (6 solver instances total)
- Results displayed as they complete
- Takes ~10-15 minutes for 5 puzzles instead of 20-30 minutes

## 📊 View Results in Streamlit

**After comparison finishes:**

```bash
# Launch Streamlit viewer
streamlit run streamlit_app.py
```

**In the Streamlit UI:**
1. Click **"Sort by: Success (✅ first)"** to see best results first
2. Select a puzzle booklet from dropdown:
   - `puzzle_name_booklet.json` = V1 results
   - `puzzle_name_booklet_v2.json` = V2 results
3. Compare the same puzzle in both versions!

**Tabs to check:**
- 📋 **Step-by-Step** - See how each solver worked through it
- 📊 **Overview** - Process statistics
- 🎯 **Final Prediction** - Visual comparison of predictions
- 📚 **Training Data** - Original puzzle examples
- 🧠 **Generated Instructions** - The rule/operations

## 🐌 Sequential Mode (For Debugging)

If you want to see detailed output or debug:

```bash
python compare_solvers.py ARC-AGI-2 5 --sequential
```

This runs one solver at a time so output is clearer.

## 📁 What Gets Generated

After running, you'll have:

**Test directory** (`test/`):
- `puzzle1_booklet.json` (V1)
- `puzzle1_booklet_v2.json` (V2)
- `puzzle2_booklet.json` (V1)
- `puzzle2_booklet_v2.json` (V2)
- etc.

**Images** (`img_tmp/`):
- All grid visualizations
- Training examples
- Intermediate steps
- Final predictions

**Comparison results**:
- `comparison_results_<timestamp>.json` - Detailed data

## 🔍 Analyzing Results

### In Console:
```
COMPARISON SUMMARY
Average Accuracy:            V1: 45.0%    V2: 55.0%
Perfect Predictions:         V1: 2/5      V2: 3/5
V1 Wins: 1 / 5  (20.0%)
V2 Wins: 3 / 5  (60.0%)
Ties:    1 / 5  (20.0%)
```

### In Streamlit:
1. **Pick a success** - See how the winning approach solved it
2. **Pick a failure** - Understand where it went wrong
3. **Compare same puzzle** - Load V1, then V2 for same puzzle
   - How did their instructions differ?
   - Where did execution diverge?
   - Which intermediate steps were wrong?

### In JSON file:
```json
{
  "v1": [
    {"puzzle": "abc123", "accuracy": 0.85, "time": 45.2, ...},
    ...
  ],
  "v2": [...],
  "summary": {
    "v1_avg_accuracy": 0.45,
    "v2_avg_accuracy": 0.55,
    "v1_wins": 1,
    "v2_wins": 3
  }
}
```

## ⚡ Performance Expectations

**For 5 puzzles (parallel mode):**
- Runtime: ~10-15 minutes total
- API costs: ~$4-6 (assuming $0.40-0.60 per puzzle per solver)
- Files generated: 10 booklets + ~50-100 images

**For 20 puzzles (parallel mode):**
- Runtime: ~40-60 minutes total  
- API costs: ~$16-24
- Files generated: 40 booklets + ~200-400 images

## 🎯 Quick Example Session

```bash
# 1. Run comparison
python compare_solvers.py ARC-AGI-2 5

# Wait 10-15 minutes...

# 2. View results in Streamlit
streamlit run streamlit_app.py

# 3. In Streamlit:
#    - Sort by "Success ✅ first"
#    - Pick a puzzle that V2 won
#    - Click "🧠 Generated Instructions" tab
#    - See V2's concise operations vs V1's longer instructions
#    - Click "📋 Step-by-Step" tab
#    - Navigate through V2's intermediate grids

# 4. Compare same puzzle in both versions:
#    - Select "puzzle_booklet.json" (V1)
#    - Look at "🎯 Final Prediction"
#    - Note accuracy %
#    - Select "puzzle_booklet_v2.json" (V2)
#    - Compare accuracy % and approach
```

## 🛠️ Troubleshooting

**"No booklet files found"** in Streamlit:
- Make sure you're in the `booklets_ARCAGI` directory when running streamlit
- Check that `test/` directory exists and has `*_booklet*.json` files

**Comparison hangs:**
- Use `--sequential` mode to see which puzzle is stuck
- Check OpenAI API key is set
- Some puzzles timeout after 5 minutes (normal)

**Want to test specific puzzles:**
- Edit `compare_solvers.py` line 334
- Replace `comparison.get_puzzle_files('training', num_puzzles)`
- With: `['ARC-AGI-2/data/training/00576224.json', ...]`

## 📊 Next Steps After Comparison

Based on results, you can:
- **If V1 wins**: Focus on improving holistic reasoning
- **If V2 wins**: Build on step-by-step approach
- **If close**: Create ensemble or hybrid approach
- **Look at failures**: Understand puzzle types each struggles with

