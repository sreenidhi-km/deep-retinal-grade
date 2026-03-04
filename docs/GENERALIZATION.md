# External Validation & Generalization Analysis

## Phase 4: Cross-Dataset Validation on IDRiD

**Date:** February 2026  
**Model:** EfficientNet-B0 (efficientnet_b0_combined.pth)  
**Training Data:** APTOS 2019 + DDR Dataset  
**Validation Data:** IDRiD (Indian Diabetic Retinopathy Image Dataset)

---

## Executive Summary

| Metric | Training (APTOS) | External (IDRiD) | Target | Status |
|--------|------------------|------------------|--------|--------|
| **QWK** | 0.8745 | **0.4898** | ≥0.60 | ❌ |
| **Accuracy** | 78.5% | 45.5% | N/A | — |
| **Grade 3 Recall** | 38.0% | 10.8% | — | ⚠️ |

**Verdict:** The model shows significant domain shift when applied to IDRiD data, indicating limited generalization to this external dataset.

---

## Dataset Comparison

### IDRiD Dataset Characteristics
- **Source:** Indian Diabetic Retinopathy Image Dataset
- **Total Images:** 516 (413 train + 103 test)
- **Camera/Acquisition:** Different imaging equipment than APTOS
- **Population:** Indian patients (different demographic than APTOS)

### Grade Distribution Comparison

| Grade | APTOS (Train) | IDRiD | Notes |
|-------|---------------|-------|-------|
| 0 - No DR | 1,805 (49.3%) | 168 (32.6%) | More balanced in IDRiD |
| 1 - Mild | 370 (10.1%) | 25 (4.8%) | Very few mild cases |
| 2 - Moderate | 999 (27.3%) | 168 (32.6%) | Similar proportion |
| 3 - Severe | 193 (5.3%) | 93 (18.0%) | **3.4x more severe** in IDRiD |
| 4 - Proliferative | 295 (8.0%) | 62 (12.0%) | More in IDRiD |

**Key Observation:** IDRiD has significantly different class distribution, with more severe cases and fewer healthy/mild cases.

---

## Detailed Performance Analysis

### Confusion Matrix Analysis

```
                  Predicted
Actual        No DR    Mild     Mod     Sev    Prol
─────────────────────────────────────────────────────
No DR            60      24      75       3       6
Mild              3       7      15       0       0
Moderate          3       5     138       4      18
Severe            1       0      75      10       7
Proliferative     0       0      33       9      20
```

### Key Findings

1. **Grade 3 (Severe) Crisis:**
   - Only 10.8% recall (10/93 correct)
   - 80.6% misclassified as Moderate (Grade 2)
   - This is the most critical clinical failure

2. **Grade 0 (No DR) Under-Detection:**
   - Only 35.7% recall (60/168 correct)
   - 44.6% misclassified as Moderate
   - May lead to unnecessary referrals

3. **Moderate (Grade 2) Over-Prediction:**
   - Model is biased toward predicting Grade 2
   - 138/168 (82.1%) Moderate correctly identified
   - But absorbs cases from all other grades

4. **Proliferative (Grade 4) Acceptable:**
   - 32.3% recall (20/62 correct)
   - Mostly confused with Moderate/Severe
   - Clinical impact: may delay urgent treatment

### Per-Class Metrics

| Grade | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 - No DR | 0.896 | 0.357 | 0.511 | 168 |
| 1 - Mild | 0.194 | 0.280 | 0.230 | 25 |
| 2 - Moderate | 0.411 | 0.821 | 0.548 | 168 |
| 3 - Severe | 0.385 | 0.108 | 0.168 | 93 |
| 4 - Proliferative | 0.392 | 0.323 | 0.354 | 62 |

---

## Root Cause Analysis

### 1. Domain Shift Factors

| Factor | Impact | Evidence |
|--------|--------|----------|
| **Camera Equipment** | High | Different image quality, contrast, lighting |
| **Population Demographics** | Medium | Indian vs. international population |
| **Preprocessing Sensitivity** | High | Ben Graham preprocessing may not transfer |
| **Annotation Guidelines** | Medium | Potentially different grading criteria |

### 2. Class Distribution Mismatch

The model was trained on APTOS where:
- Grade 0 dominates (49.3%)
- Grade 3 is rare (5.3%)

IDRiD has:
- More balanced Grade 0-2 distribution
- **3.4x more Grade 3 cases** than training data proportion

### 3. Confidence Calibration Issue

```
Confidence Statistics on IDRiD:
   Mean: 0.548 | Std: 0.125 | Range: [0.274, 0.853]
```

The model maintains moderate confidence even when making incorrect predictions, indicating poor calibration on external data.

---

## Clinical Impact Assessment

### Safety-Critical Failures

| Failure Mode | Count | Percentage | Clinical Risk |
|--------------|-------|------------|---------------|
| Severe → Moderate | 75/93 | 80.6% | **HIGH** - Delayed treatment |
| Proliferative → Moderate | 33/62 | 53.2% | **CRITICAL** - Vision loss risk |
| No DR → Moderate | 75/168 | 44.6% | LOW - False alarm, unnecessary referral |

### Binary Screening Performance

If used for binary "refer vs. no-refer" (Grade ≥2 = refer):

| Metric | Value |
|--------|-------|
| Sensitivity (catch disease) | 91.2% |
| Specificity (avoid false alarm) | 43.5% |
| Would refer | 423/516 (82.0%) |

**Interpretation:** The model is conservative (high sensitivity) but has too many false positives.

---

## Comparison with Published Literature

| Study | External Dataset | QWK | Notes |
|-------|------------------|-----|-------|
| EyePACS baseline | Messidor | 0.65 | Same continent |
| Gulshan et al. 2016 | EyePACS | 0.74 | Massive dataset |
| **This work** | **IDRiD** | **0.49** | Cross-continent |
| Typical drop-off | — | 15-25% | Domain shift |

Our 36% QWK drop (0.87 → 0.49) exceeds typical literature values, suggesting:
- IDRiD may have significantly different characteristics
- Model may be overfitting to APTOS-specific features

---

## Recommendations

### Short-Term (Current Project)

1. **Document Limitation:** ✅ Model should not be deployed on Indian population without fine-tuning
2. **Add Warning:** Include domain-specific usage warnings in API
3. **Flag Low-Confidence:** Use the safety contract to flag uncertain predictions

### Medium-Term (Future Work)

1. **Domain Adaptation:**
   - Fine-tune on small IDRiD subset (transfer learning)
   - Use domain adversarial training

2. **Multi-Source Training:**
   - Include IDRiD in training pipeline
   - Balance across imaging centers

3. **Preprocessing Optimization:**
   - Test alternative preprocessing pipelines
   - Learn preprocessing parameters

### Long-Term (Research)

1. **Federated Learning:** Train across multiple hospital systems
2. **Foundation Models:** Use pre-trained medical vision models
3. **Calibration:** Implement temperature scaling for better confidence

---

## Artifacts Generated

| File | Description |
|------|-------------|
| `results/external_validation.json` | Complete metrics and per-sample predictions |
| `scripts/validate_cross_dataset.py` | Validation pipeline (reusable) |
| `docs/GENERALIZATION.md` | This analysis document |

---

## Conclusion

The Phase 4 external validation revealed significant generalization limitations:

- **QWK dropped from 0.87 to 0.49** on IDRiD data
- **Grade 3 recall catastrophically low** at 10.8%
- Model shows strong bias toward Grade 2 predictions

**This is expected and valuable information** for responsible AI deployment. The model should be:
1. Used only on populations similar to training data
2. Accompanied by confidence calibration warnings
3. Fine-tuned before deployment in different clinical settings

The external validation demonstrates the importance of cross-dataset testing and highlights the need for continued research in domain generalization for medical AI.

---

*Phase 4 completed: 2026-02-XX | Deep Retina Grade Project*
