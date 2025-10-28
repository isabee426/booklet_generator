# Batch Generate Booklets

Easy way to generate multiple booklets at once.

## Quick Run

```powershell
# Make sure API key is set
$env:OPENAI_API_KEY="sk-..."

# Run batch script
.\batch_generate_booklets.ps1
```

Or run Python directly:
```powershell
python batch_generate_booklets.py
```

## What It Does

Generates booklets for:
- 2 training puzzles (known examples)
- 3 evaluation puzzles (challenge set)

All saved to `sample_booklets/` folder, which Streamlit can access.

## After Running

View all booklets:
```powershell
streamlit run streamlit_booklet_viewer.py
```

Then select a booklet from the dropdown to see:
- Comparison table (model vs ideal)
- Step-by-step details
- Input/output visualization

