# Flat File Structure for Visualizations

## Change Summary

Moved from hierarchical structure with folders to completely flat structure for guaranteed ordering.

---

## Old Structure (Hierarchical)

```
training_1/
├── 01_input.png
├── 02_hypotheticals/
│   ├── h_01.png
│   ├── h_02.png
│   └── h_03.png
├── 02_transform.png
├── 03_hypotheticals/
│   ├── h_01.png
│   └── h_02.png
├── 03_transform.png
├── XX_model_output.png
└── YY_actual_output.png
```

**Problems:**
- Folder ordering depends on file system
- `os.listdir()` returns arbitrary order
- Windows Explorer may show files before folders
- Complex logic needed to maintain correct order

---

## New Structure (Flat)

```
training_1/
├── 01_input.png
├── 02_01_hypothetical.png
├── 02_02_hypothetical.png
├── 02_03_hypothetical.png
├── 02_04_transform.png
├── 03_01_hypothetical.png
├── 03_02_hypothetical.png
├── 03_03_transform.png
├── XX_model_output.png
└── YY_actual_output.png
```

**Benefits:**
- ✅ Alphabetical sort = chronological order
- ✅ Works with any file listing method
- ✅ Simple: `sorted(os.listdir())` is sufficient
- ✅ Unambiguous ordering everywhere
- ✅ Easy to parse step and sequence numbers

---

## Naming Convention

### Format
```
{step:02d}_{sequence:02d}_{type}.png
```

### Examples
```
01_input.png           → Step 1 input (special case, no sequence)
02_01_hypothetical.png → Step 2, hypothetical #1
02_02_hypothetical.png → Step 2, hypothetical #2
02_03_hypothetical.png → Step 2, hypothetical #3
02_04_transform.png    → Step 2, chosen transform (comes after hypotheticals)
03_01_hypothetical.png → Step 3, hypothetical #1
03_02_transform.png    → Step 3, chosen transform
XX_model_output.png    → Model's final output (always XX prefix)
YY_actual_output.png   → Ground truth output (always YY prefix)
```

---

## Code Changes

### 1. Hypothetical Saving
```python
# Old:
hyp_dir = os.path.join(training_dir, f"{next_step:02d}_hypotheticals")
filename = f"h_{hyp_num:02d}.png"
save_path = os.path.join(hyp_dir, filename)

# New:
existing_hyps = [f for f in os.listdir(training_dir) 
                 if f.startswith(f"{next_step:02d}_") 
                 and f.endswith("_hypothetical.png")]
hyp_num = len(existing_hyps) + 1
filename = f"{next_step:02d}_{hyp_num:02d}_hypothetical.png"
save_path = os.path.join(training_dir, filename)
```

### 2. Transform Saving
```python
# Old:
filename = f"{step_num:02d}_transform.png"

# New:
existing_hyps = [f for f in os.listdir(training_dir) 
                 if f.startswith(f"{step_num:02d}_") 
                 and f.endswith("_hypothetical.png")]
sequence_num = len(existing_hyps) + 1
filename = f"{step_num:02d}_{sequence_num:02d}_transform.png"
```

### 3. Listing Visualizations
```python
# Old:
# Complex logic with folder detection, sorting, manual ordering

# New:
regular_files = sorted([
    f for f in all_files
    if f.endswith('.png')
    and not f.startswith('XX_')
    and not f.startswith('YY_')
])
# Natural sort handles ordering automatically
```

### 4. Finding Hypotheticals
```python
# Old:
hyp_dir = os.path.join(training_dir, f"{step_number:02d}_hypotheticals")
hyp_files = sorted([f for f in os.listdir(hyp_dir) if f.endswith('.png')])

# New:
all_files = os.listdir(training_dir)
hyp_files = sorted([
    f for f in all_files 
    if f.startswith(f"{step_number:02d}_") 
    and f.endswith('_hypothetical.png')
])
```

---

## Parsing Filenames

### Extract Step Number
```python
step_num = int(filename.split('_')[0])
# "02_01_hypothetical.png" → 2
```

### Extract Sequence Number
```python
sequence_num = int(filename.split('_')[1])
# "02_03_hypothetical.png" → 3
```

### Extract Type
```python
file_type = filename.split('_')[2].replace('.png', '')
# "02_03_hypothetical.png" → "hypothetical"
# "02_04_transform.png" → "transform"
```

### Check If Hypothetical
```python
is_hypothetical = filename.endswith('_hypothetical.png')
```

### Check If Transform
```python
is_transform = filename.endswith('_transform.png')
```

---

## Training Data Extraction

Now much simpler to process:

```python
def extract_step_data(training_dir, step_num):
    """Extract all files for a specific step"""
    all_files = sorted(os.listdir(training_dir))
    
    step_files = [
        f for f in all_files 
        if f.startswith(f"{step_num:02d}_")
    ]
    
    hypotheticals = [f for f in step_files if f.endswith('_hypothetical.png')]
    transform = next((f for f in step_files if f.endswith('_transform.png')), None)
    
    return {
        'step': step_num,
        'hypotheticals': hypotheticals,
        'transform': transform
    }
```

---

## Advantages for ML Training

### 1. Easy Chronological Parsing
Files naturally appear in order they were created:
```python
files = sorted(glob.glob('training_1/*.png'))
# Automatically chronological order
```

### 2. Simple Step Grouping
```python
from itertools import groupby

files = sorted(os.listdir('training_1'))
for step, group in groupby(files, key=lambda f: f.split('_')[0]):
    step_files = list(group)
    # Process all files for this step together
```

### 3. No Directory Traversal
```python
# Old: Need to check if item is directory, then list contents
# New: Simple flat list
all_visualizations = sorted(glob.glob('training_1/*.png'))
```

---

## Migration

Old visualizations with hierarchical structure will need to be converted:

```python
def migrate_to_flat_structure(training_dir):
    """Migrate from old hierarchical to new flat structure"""
    
    # Find all hypothetical directories
    hyp_dirs = [d for d in os.listdir(training_dir) 
                if d.endswith('_hypotheticals')]
    
    for hyp_dir in hyp_dirs:
        step_num = int(hyp_dir.split('_')[0])
        hyp_path = os.path.join(training_dir, hyp_dir)
        
        # Rename each hypothetical
        hyp_files = sorted(os.listdir(hyp_path))
        for idx, filename in enumerate(hyp_files, 1):
            old_path = os.path.join(hyp_path, filename)
            new_filename = f"{step_num:02d}_{idx:02d}_hypothetical.png"
            new_path = os.path.join(training_dir, new_filename)
            shutil.move(old_path, new_path)
        
        # Remove empty directory
        os.rmdir(hyp_path)
    
    # Rename transform files
    transform_files = [f for f in os.listdir(training_dir) 
                       if f.endswith('_transform.png') 
                       and not '_' in f.split('_')[0]]  # Old format
    
    for transform_file in transform_files:
        step_num = int(transform_file.split('_')[0])
        
        # Count hypotheticals for this step
        hyp_count = len([f for f in os.listdir(training_dir)
                        if f.startswith(f"{step_num:02d}_") 
                        and f.endswith('_hypothetical.png')])
        
        # Rename transform
        old_path = os.path.join(training_dir, transform_file)
        new_filename = f"{step_num:02d}_{hyp_count+1:02d}_transform.png"
        new_path = os.path.join(training_dir, new_filename)
        shutil.move(old_path, new_path)
```

---

## Summary

The flat structure eliminates all ordering ambiguity:
- ✅ No folder ordering issues
- ✅ Works with simple `sorted()`
- ✅ Chronological = Alphabetical
- ✅ Easy to parse and process
- ✅ Simpler code, fewer bugs
- ✅ Perfect for training data extraction
