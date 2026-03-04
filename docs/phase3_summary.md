# Phase 3 Summary: Grade 3 Performance Investigation

**Date:** $(date)  
**Status:** IN PROGRESS - REQUIRES REVISED APPROACH  
**Target:** Grade 3 (Severe DR) Recall ≥60%

---

## Executive Summary

Phase 3 aimed to improve Grade 3 (Severe Diabetic Retinopathy) recall from 25% to 60%+. After running Config B (APTOS+DDR combined dataset), Grade 3 recall reached only **38%**, falling short of the target. Analysis reveals the core issue is **adjacent-class confusion** between Grade 2 (Moderate) and Grade 3 (Severe), not simply a data imbalance problem.

---

## Experiments Completed

### Baseline (5 epochs, CE Loss)
| Metric | Value |
|--------|-------|
| Grade 3 Recall | 41% |
| Overall QWK | 0.82 |
| Notes | Quick baseline to establish starting point |

### Config B (APTOS + DDR Combined)
| Metric | Value |
|--------|-------|
| Dataset | APTOS (3,662) + DDR stratified sample (1,600) |
| Training Samples | 3,878 |
| Epochs | 30 |
| Training Time | 66.6 minutes |
| Best Val QWK | 0.8916 (epoch 27) |
| **Test Accuracy** | **78.5%** |
| **Test QWK** | **0.8745** |
| **Grade 3 Recall** | **37.5% (12/32)** ❌ |
| Grade 3 Precision | 36.4% |

---

## Root Cause Analysis

### Confusion Matrix (Test Set)
```
             No DR | Mild | Moderate | Severe | Prolif
No DR:        289  |   9  |     0    |    0   |    0
Mild:          10  |  32  |    19    |    0   |    0
Moderate:       1  |  26  |   120    |   13   |    5
Severe:         0  |   0  |    15    |   12   |    5
Prolif:         0  |   5  |    17    |    5   |   22
```

### Grade 3 Misclassification Breakdown
| True Grade 3 Predicted As | Count | Percentage |
|---------------------------|-------|------------|
| **Grade 2 (Moderate)** | **15** | **46.9%** |
| Grade 4 (Proliferative) | 5 | 15.6% |
| Correct (Grade 3) | 12 | 37.5% |

### Key Insights

1. **Adjacent Class Confusion is the Primary Issue**
   - 46.9% of Grade 3 cases are misclassified as Grade 2 (Moderate)
   - This is NOT a data scarcity problem - it's a boundary problem

2. **Model Confidence is Very Low for Grade 3**
   - Mean confidence: 0.287 (compared to >0.5 for other classes)
   - Median confidence: 0.195
   - The model is uncertain about Grade 3 predictions

3. **Why More Data Didn't Help**
   - Adding DDR data improved overall QWK (0.82 → 0.8745)
   - But Grade 3 recall DECREASED (41% → 38%)
   - More "normal" data diluted the minority class signal

4. **Clinical Interpretation**
   - Grade 2→3 boundary is clinically subtle (microaneurysms count, hemorrhage extent)
   - Grade 3→4 boundary is more distinct (neovascularization)
   - Model struggles with the subtle Moderate/Severe distinction

---

## What Won't Work

Based on analysis, these approaches are unlikely to help:

| Approach | Why It Won't Work |
|----------|-------------------|
| More data | Already tried - diluted minority class |
| Stronger class weights | Already using weights - model still confused |
| Simple oversampling | Repeats same confusing boundary cases |
| Focal loss alone | Already using - not addressing root cause |

---

## Recommended Next Steps

### Option A: Class-Balanced Sampling (Config C)
- Force equal class representation each batch
- Ensures model sees enough Grade 3 during training
- **Risk:** May hurt Grade 0/2 performance

### Option B: Two-Stage Classifier
1. **Stage 1:** Binary "Needs Review" (≥Grade 2) vs "OK" (Grade 0-1)
2. **Stage 2:** Severity grading among positive cases
- **Advantage:** Separates easy/hard decisions

### Option C: Ordinal Regression Approach
- Replace softmax with ordinal thresholds
- Respects Grade 2 < Grade 3 < Grade 4 relationship
- Use CORAL or soft ordinal labels

### Option D: Attention/Feature Enhancement
- Add attention mechanism to focus on Grade 3 discriminative features
- Grad-CAM analysis to understand what model looks at
- Targeted augmentation for Grade 3 images

### Option E: Stronger Backbone + Larger Images
- EfficientNet-B3 or B4 (more capacity)
- 384×384 or 512×512 images (more detail)
- **Cost:** Longer training time

---

## Files Created This Phase

```
scripts/
├── train_config_b.py           # Combined dataset training
├── create_combined_splits.py    # APTOS+DDR split creation
├── preprocess_ddr.py           # DDR preprocessing
└── analyze_grade3.py           # Misclassification analysis

splits/combined/
├── train.csv (3,878 samples)
├── val.csv (779 samples)
└── test.csv (605 samples)

results/
├── config_b_results.json       # Training metrics
└── grade3_analysis.json        # Confusion matrix analysis

models/
└── efficientnet_b0_combined.pth  # Best model checkpoint
```

---

## Decision Point

**Question for stakeholder:** Which approach should we try next?

1. **Config C (Class-Balanced)** - Quick to implement, tests if sampling helps
2. **Two-Stage Classifier** - More complex, might be overkill
3. **Ordinal Regression** - Elegant but requires loss function changes
4. **Stronger Backbone** - More compute, may help marginally
5. **Accept 38% and document** - If timeline is critical

**Recommendation:** Try **Config C (Class-Balanced Sampling)** first. It's the simplest test of whether the model CAN learn Grade 3 features with sufficient exposure. If it fails, move to ordinal regression.

---

## Commits Made

1. `feat(phase3): Config B - APTOS+DDR combined training with analysis`
   - Hash: cb277af6
   - Contains all scripts, splits, and analysis results

---

## Timeline Impact

| Item | Original Estimate | Actual |
|------|-------------------|--------|
| Config B Training | 1 hour | 66 min ✅ |
| Analysis | 30 min | 20 min ✅ |
| Grade 3 Target Met | Expected | ❌ Not Met |

**Revised estimate for Phase 3 completion:** 2-3 more training runs needed (2-3 hours)

---

*Last updated: Phase 3 Config B completion*
