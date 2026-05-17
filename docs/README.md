# 🧩 ARC Puzzle Creator & AI Step Generator
# PAPER CAN BE FOUND HERE: [PAPER](https://drive.google.com/file/d/1vAygiFlweI_tZg8ULc2_QoSXdrMOoNsa/view?pli=1)
**Create visual traces for ARC-AGI puzzles with an interactive web interface + AI-powered step-by-step generation**

Based on [arcprize.org/play](https://arcprize.org/play) with enhanced features for creating step-by-step transformation traces, plus AI generation and testing capabilities.

---

## 🤖 **NEW: AI Step Generator** (Recommended)

Generate and test step-by-step solutions automatically using GPT-5-mini!

### **Super Quick Start (PowerShell)**

```powershell
# Complete workflow: train → test → view
.\quick_commands.ps1 full 05f2a901
```

**Or step by step:**
```powershell
.\quick_commands.ps1 train 05f2a901    # Train on examples
.\quick_commands.ps1 test 05f2a901     # Solve test
.\quick_commands.ps1 view              # View results
```

📖 **See:** [`POWERSHELL_QUICK_START.md`](POWERSHELL_QUICK_START.md) for full guide  
📖 **See:** [`COMPLETE_WORKFLOW_GUIDE.md`](COMPLETE_WORKFLOW_GUIDE.md) for detailed workflow

---

## ⚡ Quick Start (Manual Puzzle Creator)

### Launch the App (Easiest Way)

**Windows:**
```bash
# Double-click this file:
START_APP.bat

# Or run:
python flask_app/app.py
```

**Mac/Linux:**
```bash
chmod +x scripts/run_flask_app.sh
./scripts/run_flask_app.sh
```

### Access
```
http://localhost:5000
```

## 📁 Organized Structure

```
booklets_ARCAGI/
│
├── 🌐 flask_app/          ← Interactive Web App ⭐
├── 🔧 batch_tools/        ← Batch processing (existing)
├── 📺 streamlit_viewers/  ← Streamlit viewers (existing)
├── 🛠️ utils/              ← Utilities (existing)
├── 📚 docs/               ← All documentation (30+ files)
├── 🚀 scripts/            ← Launch scripts
├── 📦 legacy/             ← Archived old versions
├── 💾 visual_traces/      ← YOUR SAVED WORK ⭐
└── 📂 Data folders        ← Existing data
    ├── batch_visual_booklets/
    ├── sample_booklets/
    └── ...
```

**See:** `docs/TOOLS_GUIDE.md` for what tool to use when
**See:** `docs/FILE_ORGANIZATION.md` for complete structure
**See:** `docs/PROJECT_STRUCTURE.md` for project overview

## ✨ Features

### Core Features
- 🎨 **Click-to-paint grid editor** with fill tools
- 📋 **Universal step templates** across examples
- 🤖 **AI rewrite** for clearer descriptions
- 👁️ **View any step** by clicking in sidebar
- ✏️ **Edit any step** anytime
- 🔧 **Manage steps** - reorder, insert, delete
- 📚 **Booklet viewer** - see all completed work

### Drawing Tools
- **Paint** - Click/drag to paint cells
- **Fill Area** - Drag to select, auto-fills
- **Select** - Drag to select, then fill button

### Step Management
- Sequential numbering (step_01, step_02...)
- Auto-renumbering when reorganizing
- Universal templates with context
- Edit any previous step

## 🤖 AI Booklet Generation (NEW!) ⭐

**Generate booklets automatically using meta-instructions!**

### ⚡ Quick Start (5 minutes)

```bash
# 1. Install
pip install -r requirements_ai.txt

# 2. Set API key
export OPENAI_API_KEY="sk-your-key"

# 3. Generate!
python scripts/meta_instruction_generator.py --puzzle 05f2a901
```

**See:** `docs/QUICK_START_AI_GENERATION.md` for complete guide

### 📊 What It Does

- ✅ **Visual-first reasoning** - No algorithms, pure visual thinking
- ✅ **Meta-instructions** - Cross-puzzle principles guide generation
- ✅ **Iterative refinement** - Auto-retries if validation fails
- ✅ **Research-ready** - Tracks all metrics for your paper
- ✅ **3-level system** - Meta-instructions → Universal templates → Execution

### 🎯 Features

**Generate:**
```bash
python scripts/meta_instruction_generator.py --puzzle [ID]
```

**Validate:**
```bash
python scripts/parallel_validator.py --result [generated_file]
```

**Batch test:**
```bash
python scripts/test_meta_instructions.py --visual-traces visual_traces
```

### 📚 Documentation

- **Quick Start:** `docs/QUICK_START_AI_GENERATION.md` (5-min setup)
- **Full Plan:** `docs/CUSTOM_TRAINING_PLAN_VISUAL_REASONING.md` (research approach)
- **Survey:** `docs/TRAINING_PLAN_SURVEY.md` (customize your approach)

## 📖 Documentation

### Getting Started
- `docs/FLASK_QUICK_START.md` - 30-second start
- `docs/FLASK_APP_README.md` - Complete guide
- `docs/SIMPLIFIED_README.md` - Core concepts

### Features
- `docs/UNIVERSAL_STEPS_README.md` - Template system
- `docs/AI_REWRITE_GUIDE.md` - AI description improvement
- `docs/STEP_MANAGER_GUIDE.md` - Reordering steps
- `docs/FILL_TOOL_GUIDE.md` - Drawing tools
- `docs/EDIT_STEPS_GUIDE.md` - Editing capabilities
- `docs/VIEWER_GUIDE.md` - Completed booklets viewer

### Advanced
- `docs/AI_CONTEXT_GUIDE.md` - AI with memory
- `docs/UNIVERSAL_BEHAVIOR.md` - How templates work
- `docs/STEP_VIEWER_GUIDE.md` - Viewing all steps

## 🎯 Typical Workflow

```
1. Launch: START_APP.bat or python flask_app/app.py

2. Select puzzle and example

3. Create steps:
   - Draw grid
   - Write description
   - Click 🤖 AI Rewrite (optional)
   - Save → Creates universal template

4. Switch to next example:
   - Form auto-fills with templates! ✨
   - Just draw grids
   - Save

5. Manage steps:
   - Click 🔧 Manage Steps
   - Reorder/delete/insert as needed
   - Apply changes

6. View completed:
   - Click 📚 View Completed Booklets
   - See all your work!
```

## 💾 File Structure

### Created Files
```
visual_traces/{puzzle_id}/
  ├── universal_steps.json      ← Shared templates
  ├── training_01/
  │   ├── step_01/
  │   │   ├── step.json         ← Full data
  │   │   ├── grid.png          ← Grid image
  │   │   └── metadata.txt      ← Human-readable
  │   └── step_02/
  └── testing_01/
      └── step_01/
```

## 🚀 Installation

```bash
# Install dependencies
pip install -r flask_app/requirements_flask.txt

# Optional: Set OpenAI API key for AI rewrite
$env:OPENAI_API_KEY="sk-your-key"

# Run
python flask_app/app.py
```

## 🆚 Why Flask?

**Better than Streamlit:**
- ✅ Click-to-paint (no button lag)
- ✅ Faster, more responsive
- ✅ True web app feel
- ✅ Matches arcprize.org design
- ✅ Better mobile support

## 📚 Key Concepts

### Universal Steps
- Templates shared across all examples
- Auto-fill form on subsequent examples
- Always uses most recent version
- Speeds up workflow dramatically

### Step Management
- Reorder with ▲ ▼ buttons
- Insert placeholders with ➕
- Delete with 🗑️
- Auto-renumbering

### AI Features
- Rewrite descriptions (2-3 sentences)
- Context-aware (sees previous steps)
- Optional but helpful

## 🎓 Learn More

**Read in order:**
1. `docs/FLASK_QUICK_START.md` (2 min)
2. `docs/SIMPLIFIED_README.md` (5 min)
3. `docs/FLASK_APP_README.md` (20 min)
4. Try the app!
5. Read feature guides as needed

## 🔧 Utilities

### Migration Script
```bash
# Update universal steps to most recent
python scripts/migrate_universal_steps.py
```

## ✅ Status

**Production Ready!**
- ✅ Flask web application
- ✅ Complete feature set
- ✅ Comprehensive documentation
- ✅ Organized file structure
- ✅ No linter errors
- ✅ Tested and working

## 🎊 Summary

**What You Get:**
- Complete web app matching arcprize.org
- Universal step templates
- AI-powered description improvement
- Full step management
- Completed booklet viewer
- Professional, organized codebase

**Launch Now:**
```bash
START_APP.bat
```

🧩✨ Happy puzzle solving!

