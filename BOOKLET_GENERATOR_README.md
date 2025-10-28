# ARC Booklet Generator

Generate step-by-step instructional booklets for ARC puzzles with visual comparisons.

## Quick Start

### 1. Set API Key
```bash
$env:OPENAI_API_KEY="your-api-key-here"
```

### 2. Generate a Booklet
```bash
cd booklets_ARCAGI
python arc-booklet-generator.py ..\saturn-arc\ARC-AGI-2\ARC-AGI-2\data\training\3e6067c3.json
```

### 3. View in Streamlit
```bash
streamlit run streamlit_booklet_viewer.py
```

## What It Does

### Visual-First Approach
1. **Visual Analysis** - Analyzes input image (no output shown)
2. **Identify Operations** - Breaks transformation into 3-7 human-level steps
3. **Execute Step-by-Step** - Applies each operation incrementally
4. **Save Booklet** - Creates folder with images and metadata

### Meta-Instructions Used
Based on highlighted prompts from V1 solver:
- Think VISUALLY (images, not coordinates)
- HIGH-LEVEL but PRECISE
- Identify THE SINGLE RULE
- Consistency enforcement
- Output size inference

## Booklet Structure

Generated booklet folder contains:

```
sample_booklets/
└── taskname_booklet/
    ├── input.png           # Original input
    ├── target_output.png   # Expected solution
    ├── step_000.png        # Model output after step 0
    ├── step_001.png        # Model output after step 1
    ├── step_002.png        # Model output after step 2
    ├── step_000_ideal.png  # Target for step 0 (if different)
    ├── metadata.json       # Complete step data
    └── README.txt          # Human-readable summary
```

## Streamlit Viewer

### Tab 1: Comparison Table
Shows table with columns:
- **Step** - Step number
- **Description** - Operation description
- **Tries** - Number of attempts
- **Reached Ideal** - ✅/❌ indicator
- **Has Ideal Image** - Whether separate ideal exists

Below table: Side-by-side comparison of:
- Model Output (what AI generated)
- Ideal/Target (what it should be)
- Step Info (description, tries, shape)

### Tab 2: Step Details
Expandable list of all steps with:
- Full description
- Model output image
- Ideal image (if different)
- Metadata

### Tab 3: Input/Output
- Original input image
- Target output image
- Final step result
- Success/failure indicator

## Example Operations (Human-Level)

Good operations the generator creates:

```
1. Locate the reference row at the bottom showing the color sequence
2. For each color in the sequence, find the matching colored block in the grid
3. Draw a horizontal bridge of that color connecting consecutive blocks, 
   making the bridge the same width as the blocks
4. Ensure bridges do not overwrite the block borders
```

**Not too abstract:** "Transform the grid" ❌
**Not too detailed:** "Set cell [5,0] to 2..." ❌
**Just right:** References visual elements with clear actions ✅

## Comparison with V1/V2 Solvers

| Feature | V1/V2 Solvers | Booklet Generator |
|---------|--------------|-------------------|
| Purpose | Compare approaches | Create teaching materials |
| Training examples | All examples | First example only |
| Output | JSON booklet | Folder with images |
| Viewer | streamlit_app.py | streamlit_booklet_viewer.py |
| Format | Comprehensive logs | Clean step-by-step |
| Ideal comparison | No | Yes (model vs target) |

## Use Cases

**Booklet Generator is best for:**
- 📚 Creating teaching materials
- 🔍 Understanding how AI solves step-by-step
- 👁️ Visual comparison of attempts vs targets
- 📖 Sharing solutions with others

**V1/V2 Solvers are best for:**
- 🧪 Testing accuracy
- 📊 Comparing approaches
- 🔄 Iterative refinement across examples
- ⚙️ Production solving

## Tips

1. **Start with simple puzzles** - Easier to generate good booklets
2. **Review in Streamlit** - The comparison table shows where AI struggles
3. **Use sample_booklet1** as reference - Shows ideal booklet structure
4. **Check tries column** - High tries = AI struggled with that operation

## Files Generated

Each booklet run creates:
- Images: Input, output, each step, ideals
- Metadata: JSON with all step info
- README: Human-readable summary

All saved in `sample_booklets/<taskname>_booklet/`

