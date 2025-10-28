# Setup Guide

## Issue: Missing Dependencies

The solvers need the `openai` module and other dependencies. You have two options:

## Option 1: Use Existing Virtual Environment (Fastest)

You already have a virtual environment with dependencies at `saturn-arc/arc-solver-env/`:

```bash
# Activate it (PowerShell)
..\saturn-arc\arc-solver-env\Scripts\Activate.ps1

# Now run solvers
python arc-booklets-solver-v1.py ..\saturn-arc\ARC-AGI-2\ARC-AGI-2\data\training\00576224.json
```

## Option 2: Install Dependencies in booklets_ARCAGI

```bash
# Create requirements.txt
# (already exists in saturn-arc, can copy)

pip install openai pillow numpy streamlit

# Then run solvers
python arc-booklets-solver-v1.py ..\saturn-arc\ARC-AGI-2\ARC-AGI-2\data\training\00576224.json
```

## Option 3: Copy Files to saturn-arc Directory

Since saturn-arc already has the environment setup:

```bash
# Copy solver files to saturn-arc
copy arc-booklets-solver-v1.py ..\saturn-arc\
copy arc-booklets-solver-v2-stepwise.py ..\saturn-arc\
copy compare_solvers.py ..\saturn-arc\

# Navigate there
cd ..\saturn-arc

# Activate environment
arc-solver-env\Scripts\Activate.ps1

# Run comparison
python compare_solvers.py ARC-AGI-2\ARC-AGI-2 5
```

## Recommended: Option 3

This keeps everything in one place where the environment is already configured.

## After Setup:

```bash
# Test V2 works
python arc-booklets-solver-v2-stepwise.py ARC-AGI-2\ARC-AGI-2\data\training\00576224.json

# Run comparison
python compare_solvers.py ARC-AGI-2\ARC-AGI-2 5

# View in Streamlit
streamlit run streamlit_app.py
```

