# Solver Comparison: V1 vs V2

## Two Approaches

### V1: Holistic Visual Reasoning (`arc-booklets-solver-v1.py`)
- Identifies ONE transformation rule
- Generates comprehensive visual instructions
- Applies holistically to test input
- Emphasizes pattern recognition
- Uses iterative refinement

### V2: Step-by-Step Execution (`arc-booklets-solver-v2-stepwise.py`)
- Identifies THE RULE (like V1)
- Breaks into 3-7 concrete operations
- Executes each operation incrementally
- Shows intermediate grids
- Like the sample booklet approach

## Running the Comparison

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_key_here

# Run comparison on 20 random training puzzles
python compare_solvers.py ARC-AGI-2 20

# Or specify fewer puzzles for quick test
python compare_solvers.py ARC-AGI-2 5
```

## What Gets Compared

For each puzzle, both solvers run and we compare:
- **Accuracy %** - Cell-by-cell matching
- **Perfect predictions** - 100% accuracy count
- **Time taken** - Execution speed
- **Step count** - How many steps generated
- **Head-to-head** - Which solver wins on each puzzle

## Output

The script produces:
1. **Console output** with progress and summary
2. **JSON file** (`comparison_results_<timestamp>.json`) with detailed data
3. **Booklet files** in `test/` directory:
   - `<puzzle>_booklet.json` (V1)
   - `<puzzle>_booklet_v2.json` (V2)

## Example Output

```
COMPARISON SUMMARY
================================================================================

Metric                         V1 (Holistic)        V2 (Step-by-Step)   
----------------------------------------------------------------------
Average Accuracy                           45.0%                55.0%
Perfect Predictions                          5 / 20                8 / 20
Average Time (seconds)                     120.5               150.3

HEAD-TO-HEAD RESULTS
================================================================================

V1 Wins:     6 / 20  (30.0%)
V2 Wins:    10 / 20  (50.0%)
Ties:        4 / 20  (20.0%)

CONCLUSION
================================================================================

✅ V2 (Step-by-Step) performs BETTER by 10.0% on average
```

## Analyzing Results

After running, you can:

1. **View console summary** - Quick overview of which approach wins
2. **Check JSON file** - Detailed per-puzzle results
3. **Open booklets in Streamlit** - Visual comparison of solving process:
   ```bash
   streamlit run streamlit_app.py
   ```
4. **Look at failure cases** - Where each approach struggles

## Tips

- **Start small** (5 puzzles) to verify everything works
- **Use evaluation set** for final comparison (not training)
- **Watch for timeouts** - Some puzzles take longer
- **Check img_tmp/** - All visualization images saved there
- **Review both booklets** for same puzzle to understand differences

## Next Steps Based on Results

If **V1 wins**:
- Holistic reasoning is better
- Keep simplification features
- Maybe add more visual emphasis

If **V2 wins**:
- Step-by-step execution helps
- Consider hybrid: V1 for planning, V2 for execution
- Improve operation parsing

If **tie**:
- Both have strengths
- Use V1 for pattern-heavy puzzles
- Use V2 for procedural puzzles
- Build ensemble that picks approach per puzzle

