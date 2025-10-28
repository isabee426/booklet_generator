# View Your Booklets

## 🚀 Quick Start

```bash
# Launch comprehensive viewer
streamlit run streamlit_comprehensive_viewer.py
```

## 📁 What You'll See

### **Organized by Approach:**
- **Batch Visual**: All runs with timestamps, can compare multiple runs
- **Single Example**: Individual booklets
- **Iterative Refiner**: Step evolution across examples
- **Ensemble**: Multi-booklet synthesis

### **For Each Run:**
- Timestamp (so you can see latest vs older runs)
- Success rates (training and test)
- Number of refinements
- All example booklets

### **For Each Example Booklet:**
- Step-by-step visual progression
- Model output vs Expected output
- Which steps succeeded/failed
- Full step descriptions

## 📊 Booklet Storage Structure

```
batch_visual_booklets/
  ├─ 00d62c1b_batch_20251028_140523/  ← Run 1 (with timestamp)
  │   ├─ batch_results.json
  │   ├─ example_1_booklet/
  │   ├─ example_2_booklet/
  │   └─ README.txt
  │
  └─ 00d62c1b_batch_20251028_153012/  ← Run 2 (newer timestamp)
      ├─ batch_results.json
      ├─ example_1_booklet/
      └─ ...

sample_booklets/
  ├─ task1_booklet/
  └─ task2_booklet/

refined_booklets/
  └─ task_refined/
```

## 🎯 Benefits

✅ **Never lose results** - Each run is timestamped
✅ **Compare runs** - See how changes affected results
✅ **Organized by approach** - Easy to find what you're looking for
✅ **All in one place** - Comprehensive view of all booklets

