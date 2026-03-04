# Deep Retina Grade - Project Report

## Executive Summary

**Deep Retina Grade** is an AI-powered diabetic retinopathy (DR) grading system designed for clinical deployment. The project demonstrates end-to-end medical AI development: from data preprocessing through model training, explainability, fairness auditing, and external validation.

The system achieves **QWK of 0.8745** on the APTOS test set using EfficientNet-B0 with advanced training techniques (FocalLoss, Mixup augmentation, Test-Time Augmentation). However, external validation on the IDRiD dataset revealed significant **domain shift** (QWK dropped to 0.49), highlighting the critical importance of cross-dataset testing for medical AI systems.

This project serves as a comprehensive case study in responsible AI development, demonstrating both the capabilities and limitations of deep learning for medical image analysis. The codebase includes production-ready components: a tested API with safety contracts, uncertainty quantification, triple-method explainability (GradCAM, Integrated Gradients, LIME), and fairness auditing.

---

## Project Timeline

| Phase | Description | Duration | Status |
|-------|-------------|----------|--------|
| Phase 1 | Testing Infrastructure | ~2 hours | ✅ Complete |
| Phase 2 | API Safety Contract | ~1 hour | ✅ Complete |
| Phase 3 | Performance Gap Fix | ~3 hours | ✅ Complete |
| Phase 4 | External Validation | ~1.5 hours | ✅ Complete |
| Phase 5 | Documentation Polish | ~1 hour | ✅ Complete |
| **Total** | — | **~8.5 hours** | **✅ Done** |

---

## Phase 1: Testing Infrastructure

**Objective:** Create comprehensive pytest test suite for CI/CD readiness.

### Deliverables
- `tests/test_preprocessing.py` - Image preprocessing validation
- `tests/test_model.py` - Model architecture and inference tests
- `tests/test_api.py` - FastAPI endpoint testing
- `tests/test_safety_contract.py` - Safety flag verification
- `tests/conftest.py` - Shared fixtures

### Results
| Metric | Value |
|--------|-------|
| Total Tests | 69 |
| Passed | 69 |
| Failed | 0 |
| Coverage | Core modules tested |

### Key Test Categories
1. **Preprocessing Tests:** Input validation, output shape, normalization
2. **Model Tests:** Forward pass, output dimensions, gradient flow
3. **API Tests:** All endpoints, error handling, response formats
4. **Safety Tests:** Low confidence flags, borderline detection, Grade 3+ alerts

---

## Phase 2: API Safety Contract

**Objective:** Implement clinical safety flags in API responses.

### Safety Flags Implemented

| Flag | Trigger Condition | Clinical Purpose |
|------|-------------------|------------------|
| `low_confidence` | Confidence < 70% | Uncertain predictions need review |
| `borderline_case` | Adjacent grades close | Near decision boundary |
| `high_grade_detected` | Grade ≥ 3 | Severe/Proliferative needs urgent referral |
| `high_uncertainty` | MC Dropout std > 0.15 | Model disagreement |

### API Response Structure
```json
{
  "predicted_grade": 3,
  "grade_name": "Severe NPDR",
  "confidence": 0.65,
  "uncertainty": 0.18,
  "needs_human_review": true,
  "safety_flags": ["low_confidence", "high_grade_detected", "high_uncertainty"],
  "recommendation": "URGENT referral to ophthalmologist within 2-4 weeks"
}
```

---

## Phase 3: Performance Gap Fix

**Objective:** Improve Grade 3 (Severe DR) recall from 25% to ≥60%.

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Dataset | APTOS (3,662) + DDR (1,600) = 3,878 train images |
| Model | EfficientNet-B0 (timm) |
| Loss | FocalLoss (γ=2) + Label Smoothing (0.1) |
| Augmentation | Mixup (α=0.2) + Standard augmentations |
| Optimizer | AdamW (lr=1e-4, weight_decay=0.01) |
| Scheduler | CosineAnnealingLR |
| Epochs | 30 |
| Training Time | 66.6 minutes (M1 MacBook Air) |

### Results

| Metric | Baseline | Phase 3 | Target |
|--------|----------|---------|--------|
| **QWK** | 0.811 | **0.8745** ✅ | ≥0.85 |
| **Accuracy** | 65.1% | **78.5%** | — |
| **Grade 3 Recall** | 25% | **38%** ⚠️ | ≥60% |
| **Grade 3 Precision** | — | 36.4% | — |

### Root Cause Analysis
Grade 3 recall remained at 38% due to **adjacent-class confusion**:
- 46.9% of Grade 3 samples misclassified as Grade 2 (Moderate)
- The boundary between Moderate and Severe NPDR is clinically ambiguous
- Larger models and higher resolution needed for fine-grained discrimination

### Recommendations for Production
1. Use EfficientNet-B3+ (larger capacity)
2. Increase input resolution to 384×384 or 512×512
3. Implement ordinal regression with CORAL loss
4. Train on multi-center data for robustness

---

## Phase 4: External Validation

**Objective:** Validate generalization on IDRiD dataset (QWK ≥0.60).

### Dataset Comparison

| Dataset | Source | Images | Camera | Population |
|---------|--------|--------|--------|------------|
| APTOS 2019 | Kaggle | 3,662 | Various | International |
| IDRiD | Indian | 516 | Different | Indian |

### Results

| Metric | APTOS (Internal) | IDRiD (External) | Drop |
|--------|------------------|------------------|------|
| **QWK** | 0.8745 | **0.4898** | -36% |
| **Accuracy** | 78.5% | 45.5% | -33% |
| **Grade 3 Recall** | 38% | 10.8% | -27% |

### Key Findings

1. **Significant Domain Shift:** 36% QWK drop indicates model learned APTOS-specific features
2. **Grade 3 Catastrophe:** 80.6% of Grade 3 misclassified as Moderate on IDRiD
3. **Camera Equipment Impact:** Different imaging hardware affects preprocessing
4. **Population Difference:** Indian demographics differ from training distribution

### Clinical Implications
- **Do NOT deploy** this model on Indian population without fine-tuning
- External validation is **essential** for medical AI
- Domain adaptation or multi-center training required for generalization

---

## Phase 5: Documentation Polish

**Objective:** Complete all documentation for submission-ready state.

### Documentation Inventory

| Document | Purpose | Status |
|----------|---------|--------|
| [README.md](../README.md) | Project overview and quick start | ✅ Updated |
| [docs/USAGE.md](USAGE.md) | Installation and usage guide | ✅ Created |
| [docs/PROJECT_REPORT.md](PROJECT_REPORT.md) | This document | ✅ Created |
| [docs/GENERALIZATION.md](GENERALIZATION.md) | External validation analysis | ✅ Complete |
| [docs/phase3_summary.md](phase3_summary.md) | Grade 3 investigation | ✅ Complete |
| [docs/TESTING.md](TESTING.md) | Test suite documentation | ✅ Complete |
| [docs/LOCKED_PLAN.md](LOCKED_PLAN.md) | Original implementation plan | ✅ Complete |
| [ARCHITECTURE.md](../ARCHITECTURE.md) | Technical architecture | ✅ Existing |

---

## Final Metrics Summary

### Model Performance
| Metric | Value |
|--------|-------|
| Internal QWK | 0.8745 |
| External QWK | 0.4898 |
| Accuracy | 78.5% (internal) |
| Fairness (80% Rule) | ✅ PASS |

### Codebase Quality
| Metric | Value |
|--------|-------|
| Test Cases | 69 passed |
| API Endpoints | 8 |
| Safety Flags | 4 |
| XAI Methods | 3 (GradCAM, IG, LIME) |

### Project Artifacts
| Artifact | Location |
|----------|----------|
| Best Model | `models/efficientnet_b0_combined.pth` |
| Test Metrics | `results/test_metrics.json` |
| External Validation | `results/external_validation.json` |
| Fairness Audit | `results/fairness_audit.json` |

---

## Limitations & Future Work

### Current Limitations

1. **Domain Generalization:** Model does not transfer well to external datasets
2. **Grade 3 Detection:** 38% recall is insufficient for clinical use
3. **Compute Constraints:** Trained on consumer hardware (M1 MacBook Air)
4. **Single-Center Training:** Primarily APTOS data

### Recommended Future Work

| Priority | Task | Expected Impact |
|----------|------|-----------------|
| High | Multi-center training data | Better generalization |
| High | Larger model (EfficientNet-B3+) | +10-15% Grade 3 recall |
| High | Higher resolution (384×384) | Finer lesion detection |
| Medium | Ordinal regression (CORAL) | Better adjacent-class separation |
| Medium | Domain adaptation techniques | IDRiD performance |
| Low | Foundation model fine-tuning | State-of-the-art accuracy |

---

## Conclusion

Deep Retina Grade demonstrates the complete lifecycle of medical AI development:

✅ **Strengths:**
- Production-ready API with safety contracts
- Comprehensive testing (69 tests)
- Triple explainability (GradCAM, IG, LIME)
- Fairness auditing passing 80% rule
- Honest external validation revealing limitations

⚠️ **Honest Limitations:**
- Grade 3 recall below clinical threshold
- Significant domain shift on external data
- Not ready for deployment without fine-tuning

The project serves as a template for responsible medical AI development, emphasizing the importance of external validation, uncertainty quantification, and transparent documentation of limitations.

---

**Project Completed:** February 2026  
**Total Development Time:** ~8.5 hours  
**Status:** ✅ Submission-Ready

---

*Deep Retina Grade Project Team*
