# Prompt Sections Checklist for `visual_step_generator.py`

## Current Prompt Structure (Lines 734-1124)

### ✅ **SECTION 1: Header & Core Philosophy** (Lines 734-752)
- [ ] Step number header
- [ ] "CRITICAL: STEP-BY-STEP OBJECT-ORIENTED VISUALIZATION" warning
- [ ] 7-point instructional booklet requirements
- [ ] FORBIDDEN vs CORRECT examples (COPY vs HIGHLIGHT distinction)

**Size:** ~18 lines
**Repetition:** Mentions COPY vs HIGHLIGHT distinction (repeated later)

---

### ✅ **SECTION 2: Grid Size Rules** (Lines 753-774)
- [ ] IF INPUT > OUTPUT rules
- [ ] IF INPUT = OUTPUT rules  
- [ ] IF INPUT < OUTPUT rules
- [ ] RULE SUMMARY

**Size:** ~22 lines
**Repetition:** Grid size mentioned again in "CRITICAL REQUIREMENTS" section

---

### ✅ **SECTION 3: When to Use Each Action** (Lines 776-787)
- [ ] Decision flowchart for actions
- [ ] COPY before HIGHLIGHT rule

**Size:** ~12 lines
**Repetition:** This is repeated in "DECISION TREE" section (lines 1013-1031)

---

### ✅ **SECTION 4: Data Context** (Lines 789-805)
- [ ] PREVIOUS STEPS list
- [ ] IMAGES descriptions
- [ ] INPUT grid (full grid data)
- [ ] EXPECTED OUTPUT grid (full grid data)
- [ ] CURRENT STATE grid (full grid data)
- [ ] Valid colors list
- [ ] DIFFERENCES (current vs expected)

**Size:** ~17 lines + 3 full grids (can be large!)
**Repetition:** None - this is essential context

---

### ✅ **SECTION 5: Action Functions Detailed** (Lines 807-935)
- [ ] KEY DISTINCTION reminder (COPY vs HIGHLIGHT)
- [ ] 1. COPY - detailed with examples
- [ ] 2. MOVE - brief
- [ ] 3. EXPAND - brief
- [ ] 4. FILL - detailed with examples
- [ ] 5. MODIFY - brief
- [ ] 5a. CROP/RESIZE - brief
- [ ] 6. HIGHLIGHT_SUBPROCESS - VERY DETAILED (3-step process)
- [ ] 7. NO-OP - brief

**Size:** ~129 lines
**Repetition:** HIGHLIGHT rules repeated in "HIGHLIGHT SUBPROCESS RULES" section (lines 937-963)

---

### ✅ **SECTION 6: HIGHLIGHT Subprocess Rules (DUPLICATE)** (Lines 937-963)
- [ ] MANDATORY FOR PARTIAL TRANSFORMATIONS
- [ ] When to use HIGHLIGHT
- [ ] When NOT to use HIGHLIGHT
- [ ] Process steps

**Size:** ~27 lines
**Repetition:** This duplicates Section 5's HIGHLIGHT details!

---

### ✅ **SECTION 7: HIGHLIGHT Examples** (Lines 965-968)
- [ ] Example descriptions

**Size:** ~4 lines
**Repetition:** Examples already in Section 5

---

### ✅ **SECTION 8: Critical Requirements** (Lines 969-975)
- [ ] ALWAYS track dimensions/colors/positions
- [ ] Description and Grid must match
- [ ] Grid size reminder
- [ ] Valid colors reminder
- [ ] Break into multiple steps

**Size:** ~7 lines
**Repetition:** Grid size and colors mentioned elsewhere

---

### ✅ **SECTION 9: COPY Examples** (Lines 976-986)
- [ ] BAD vs GOOD examples
- [ ] ONE object per step reminder

**Size:** ~11 lines
**Repetition:** COPY examples already in Section 5

---

### ✅ **SECTION 10: Other Action Examples** (Lines 988-992)
- [ ] MOVE, EXPAND, MODIFY examples

**Size:** ~5 lines
**Repetition:** Examples already in Section 5

---

### ✅ **SECTION 11: HIGHLIGHT Subprocess Examples (DUPLICATE)** (Lines 993-1011)
- [ ] Scenario 1: Color 6 stays, color 2 transforms
- [ ] Scenario 2: Top half stays, bottom transforms
- [ ] KEY RULES reminder

**Size:** ~19 lines
**Repetition:** This duplicates Section 5's HIGHLIGHT examples!

---

### ✅ **SECTION 12: Decision Tree (DUPLICATE)** (Lines 1013-1054)
- [ ] Q1-Q4 decision flowchart
- [ ] CRITICAL SEQUENCE reminder
- [ ] MANDATORY RULE reminder
- [ ] Examples of decision tree

**Size:** ~42 lines
**Repetition:** This duplicates Section 3's "When to Use Each Action"!

---

### ✅ **SECTION 13: Before Generating Checklist** (Lines 1056-1065)
- [ ] 5-point self-check questions

**Size:** ~10 lines
**Repetition:** Some overlap with earlier sections

---

### ✅ **SECTION 14: HIGHLIGHT Step 1 Checklist** (Lines 1066-1075)
- [ ] 6-step process for finding anchors

**Size:** ~10 lines
**Repetition:** This duplicates HIGHLIGHT details from Section 5!

---

### ✅ **SECTION 15: Critical Distinctions** (Lines 1076-1079)
- [ ] COPY vs HIGHLIGHT reminder

**Size:** ~4 lines
**Repetition:** This is the 3rd+ time mentioning COPY vs HIGHLIGHT!

---

### ✅ **SECTION 16: Mandatory Output Format** (Lines 1081-1117)
- [ ] Format requirements (Visual Analysis, Description, GRID)
- [ ] Examples in concise format
- [ ] SEQUENCE reminder

**Size:** ~37 lines
**Repetition:** Format mentioned earlier, examples duplicate Section 5

---

### ✅ **SECTION 17: Verify Checklist** (Lines 1119-1124)
- [ ] 4-point verification checklist

**Size:** ~6 lines
**Repetition:** Overlaps with Section 13

---

## Summary Statistics

**Total Lines:** ~390 lines of prompt text
**Total Sections:** 17 sections
**Major Repetitions:**
1. HIGHLIGHT subprocess explained 3+ times (Sections 5, 6, 11, 14)
2. COPY vs HIGHLIGHT distinction mentioned 4+ times (Sections 1, 5, 6, 15)
3. Decision tree/flowchart duplicated (Sections 3 and 12)
4. Examples scattered across multiple sections (5, 9, 10, 11, 16)
5. Grid size rules mentioned 3+ times (Sections 2, 8, 16)

## Recommendations for Reduction

### Keep (Essential):
- ✅ Section 1: Core philosophy (condensed)
- ✅ Section 2: Grid size rules (essential)
- ✅ Section 4: Data context (INPUT/OUTPUT/CURRENT grids - essential)
- ✅ Section 5: Action functions (but consolidate HIGHLIGHT into one place)
- ✅ Section 16: Output format (essential)

### Consolidate/Remove:
- ❌ Section 6: Remove (duplicate of Section 5's HIGHLIGHT)
- ❌ Section 7: Remove (examples already in Section 5)
- ❌ Section 9: Remove (examples already in Section 5)
- ❌ Section 10: Remove (examples already in Section 5)
- ❌ Section 11: Remove (duplicate HIGHLIGHT examples)
- ❌ Section 12: Remove (duplicate of Section 3)
- ❌ Section 13: Condense to 2-3 key points
- ❌ Section 14: Remove (duplicate of Section 5)
- ❌ Section 15: Remove (already covered)
- ❌ Section 17: Remove (redundant)

### Potential Size Reduction:
- Current: ~390 lines
- After consolidation: ~200-250 lines (35-40% reduction)
- Keep all essential info, remove repetition

