# LOCKED_PLAN.md — Deep Retina Grade Completion Plan

**Created:** January 31, 2026  
**Status:** AWAITING APPROVAL  
**Mode:** Phase 0 Complete — READ-ONLY exploration finished

---

## Executive Summary

### Project State
| Component | Status | Notes |
|-----------|--------|-------|
| Data Pipeline | ✅ Complete | APTOS loaded, stratified splits exist |
| Model Training | ✅ Complete | EfficientNet-B0, QWK=0.864, Acc=79.8% |
| XAI | ✅ Complete | GradCAM, IG, LIME all implemented |
| Backend API | 🟡 Partial | Missing decision flags, TTA not wired, no /health validation |
| Frontend | ✅ Complete | React + Vite, drag-drop, results display |
| Tests | ❌ Missing | Zero automated tests |
| Documentation | 🟡 Partial | README/ARCH exist, no deployment guide |

### Critical Gap: Grade 3 Recall = 25%
The model correctly identifies only 8 of 32 severe DR cases (Grade 3). This is clinically dangerous.

---

## Dataset Inventory

### 1. APTOS 2019 (Primary - Already in use)
- **Path:** `aptos2019-blindness-detection/`
- **Images:** 3,662 (train: 3,662 with labels in `train.csv`)
- **Labels CSV:** `train.csv` (columns: `id_code`, `diagnosis`)
- **Label Mapping:** Direct 0-4 DR grades (native format)
- **Image Location:** `train_images/{id_code}.png`

### 2. DDR Dataset (NEW - Available)
- **Path:** `DDR Dataset/`
- **Images:** ~12,522 (per `DR_grading.csv` line count)
- **Labels CSV:** `DR_grading.csv` (columns: `id_code`, `diagnosis`)
- **Label Mapping:** Direct 0-4 DR grades (same as APTOS)
- **Image Location:** `DR_grading/DR_grading/{id_code}`
- **Notes:** Largest dataset available. Same label format as APTOS.

### 3. IDRiD Disease Grading (NEW - Available)
- **Path:** `B. Disease Grading/`
- **Images:** 516 total (413 train + 103 test per CSV line counts)
- **Labels CSV:** 
  - `2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv`
  - `2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv`
- **Label Mapping:** Column `Retinopathy grade` contains 0-4 grades (native format)
- **Image Location:** 
  - Train: `1. Original Images/a. Training Set/{Image name}.jpg`
  - Test: `1. Original Images/b. Testing Set/{Image name}.jpg`
- **Notes:** Contains macular edema labels too (optional use)

### Total Available: ~16,700 images across 3 datasets

---

## Phase Plan

### PHASE 1 — P0: TESTS + RELIABILITY
**Goal:** Make the project regression-safe with automated tests.

#### Files to Create:
| File | Purpose |
|------|---------|
| `tests/__init__.py` | Package marker |
| `tests/conftest.py` | Pytest fixtures (dummy images, model loading) |
| `tests/test_preprocessing.py` | Ben Graham + CLAHE shape/dtype/range tests |
| `tests/test_model_loading.py` | Model loads or fails gracefully |
| `tests/test_api_contract.py` | /health=200, /predict returns required keys |
| `tests/test_xai_endpoints.py` | XAI endpoints respond correctly |
| `docs/TESTING.md` | How to run tests |

#### Files to Modify:
| File | Change |
|------|--------|
| `requirements.txt` | Add `pytest>=7.4.0`, `pytest-cov>=4.1.0`, `httpx>=0.25.0` (for FastAPI testing) |

#### Acceptance Criteria:
- ✅ `pytest tests/ -q` runs with ≥10 passing tests
- ✅ Tests cover preprocessing, model loading, API endpoints
- ✅ Tests skip gracefully if model weights missing (with clear message)

#### Estimated Runtime: ~5 minutes to run full test suite

---

### PHASE 2 — P0: API SAFETY CONTRACT
**Goal:** Backend returns clinically meaningful, demo-safe responses.

#### Current /predict Response:
```json
{
  "grade": 2,
  "grade_name": "Moderate",
  "confidence": 0.85,
  "probabilities": {...},
  "recommendation": "...",
  "referral_urgency": "..."
}
```

#### Required /predict Response (NEW):
```json
{
  "grade": 2,
  "grade_name": "Moderate",
  "confidence": 0.85,
  "uncertainty": 0.12,
  "quality_score": 0.95,
  "is_ood": false,
  "decision": "OK",
  "explanation_text": "Moderate DR detected with high confidence...",
  "probabilities": {...},
  "recommendation": "...",
  "referral_urgency": "..."
}
```

#### Decision Logic:
```
if quality_score < 0.4:        decision = "RETAKE"
elif is_ood:                   decision = "OOD"
elif uncertainty > 0.25:       decision = "REVIEW"
else:                          decision = "OK"
```

#### Files to Modify:
| File | Change |
|------|--------|
| `app/main.py` | Add quality_score, uncertainty, is_ood, decision, explanation_text to /predict |
| `app/main.py` | Add /health validation (model loaded + imports work) |
| `app/main.py` | Add warm-up dummy inference on startup |
| `app/main.py` | Add DEMO_MODE env var for TTA/MC Dropout sample counts |
| `src/preprocessing/preprocess.py` | Add `compute_quality_score()` method (Laplacian variance, contrast) |

#### Files to Create:
| File | Purpose |
|------|---------|
| `assets/sample/sample_fundus.jpg` | Sample image for testing (can be any valid fundus image) |

#### Environment Variables:
| Variable | Default | Description |
|----------|---------|-------------|
| `DEMO_MODE` | `true` | Fast inference for demos |
| `OOD_THRESHOLD` | `0.5` | Threshold for OOD detection |

#### DEMO_MODE Settings:
| Setting | DEMO_MODE=true | DEMO_MODE=false |
|---------|----------------|-----------------|
| TTA Count | 2 | 5 |
| MC Dropout Samples | 5 | 20 |
| IG Steps | 16 | 50 |

#### Acceptance Criteria:
- ✅ GET /health returns 200 with model_loaded=true
- ✅ POST /predict returns all required fields
- ✅ Low quality image returns decision="RETAKE"
- ✅ High uncertainty image returns decision="REVIEW"
- ✅ First request completes in <3s (warm-up done)

#### Estimated Runtime: ~2 hours implementation

---

### PHASE 3 — P0/P1: MODEL QUALITY UPGRADE
**Goal:** Improve Grade 3 recall to ≥70% while maintaining QWK ≥0.80.

#### Current Performance:
| Metric | Value | Target |
|--------|-------|--------|
| QWK | 0.864 | ≥0.80 ✅ |
| Accuracy | 79.8% | Improve |
| Grade 3 Recall | 25% | ≥70% ❌ |

#### Strategy:
1. **Multi-dataset training:** Combine APTOS + DDR (+ optionally IDRiD)
2. **Class weighting:** Heavy weight on Grade 3/4
3. **Focal loss tuning:** Higher gamma for hard examples
4. **Optional ordinal head:** Compare with standard classification

#### Files to Create:
| File | Purpose |
|------|---------|
| `src/data/unified_dataset.py` | Unified loader for APTOS/DDR/IDRiD |
| `scripts/build_data_manifest.py` | Generate `results/data_manifest.json` |
| `scripts/train_multi_dataset.py` | Training with combined datasets |

#### Files to Modify:
| File | Change |
|------|--------|
| `src/training/losses.py` | Add configurable Grade 3/4 boost factor |

#### Experiment Configurations (max 4):
| Config | Loss | Grade 3/4 Weight | Dataset |
|--------|------|------------------|---------|
| baseline_multi | Focal | 2.0 | APTOS+DDR |
| high_severe | Focal | 4.0 | APTOS+DDR |
| ordinal_multi | Ordinal | 3.0 | APTOS+DDR |
| all_data | Focal | 3.0 | APTOS+DDR+IDRiD |

#### Output Artifacts (per run):
- `results/runs/{run_id}/config.yaml`
- `results/runs/{run_id}/metrics.json`
- `results/runs/{run_id}/confusion_matrix.png`
- `models/best_{run_id}.pth` (if better than current)

#### Estimated Runtime per Config:
- APTOS only: ~20 min (already done)
- APTOS+DDR (~16K images): ~60-90 min on M1 CPU
- Full dataset (~17K images): ~90-120 min on M1 CPU

#### ⚠️ CONFIRMATION REQUIRED:
Training runs on combined datasets will take **60-120 minutes each** on M1 CPU.
I will STOP and ask for confirmation before starting any training run.

#### Acceptance Criteria:
- ✅ Grade 3 Recall ≥70% on test set
- ✅ QWK ≥0.80 maintained
- ✅ All experiment configs documented in `results/runs/`
- ✅ Best model saved with clear naming

---

### PHASE 4 — P1: CROSS-DATASET VALIDATION
**Goal:** Demonstrate generalization across datasets (no Messidor-2).

#### Validation Matrix:
| Train On | Test On | Purpose |
|----------|---------|---------|
| APTOS | DDR | Domain shift test |
| APTOS | IDRiD | External validation |
| APTOS+DDR | IDRiD | Generalization test |

#### Files to Create:
| File | Purpose |
|------|---------|
| `scripts/validate_cross_dataset.py` | Cross-dataset evaluation script |
| `docs/GENERALIZATION.md` | Results summary and limitations |

#### Output:
- `results/external_validation.json`
- Per-dataset metrics (QWK, accuracy, per-class recall)

#### Estimated Runtime: ~30 min (inference only, no training)

#### Acceptance Criteria:
- ✅ QWK ≥0.60 on external datasets (lower due to domain shift is expected)
- ✅ Results documented in `docs/GENERALIZATION.md`
- ✅ Limitations clearly stated

---

### PHASE 5 — P2: DOCS + DEPLOYMENT POLISH
**Goal:** Production-ready documentation and deployment.

#### Files to Create:
| File | Purpose |
|------|---------|
| `docs/DEPLOYMENT.md` | Docker deployment instructions |
| `docs/API_EXAMPLES.md` | curl examples for all endpoints |

#### Files to Modify:
| File | Change |
|------|--------|
| `docker-compose.yml` | Add healthcheck using /health endpoint |

#### Acceptance Criteria:
- ✅ Fresh clone can be deployed in <15 minutes following guide
- ✅ All API endpoints have curl examples
- ✅ docker-compose healthcheck passes

---

## Priority Summary

| Priority | Phase | Task | Est. Time |
|----------|-------|------|-----------|
| **P0** | 1 | Tests + Reliability | 1-2 hours |
| **P0** | 2 | API Safety Contract | 2-3 hours |
| **P0** | 3 | Model Quality (Grade 3) | 3-5 hours (incl. training) |
| **P1** | 4 | Cross-Dataset Validation | 1-2 hours |
| **P2** | 5 | Docs + Deployment | 1 hour |

**Total Estimated Time:** 8-13 hours

---

## Git Commit Plan

| Phase | Commit Message |
|-------|----------------|
| 1 | `feat(tests): add pytest suite for preprocessing, model, API` |
| 2 | `feat(api): add decision flags, quality score, warm-up to /predict` |
| 3 | `feat(training): multi-dataset training with improved Grade 3 recall` |
| 4 | `feat(validation): cross-dataset validation without Messidor-2` |
| 5 | `docs: add deployment guide and API examples` |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Training too slow on M1 | Limit to 3-4 configs, use early stopping |
| Grade 3 recall not reaching 70% | Document trade-offs, show improvement vs baseline |
| DDR/IDRiD label quality issues | Validate label distributions before training |
| OOD detection poorly calibrated | Use configurable threshold via env var |

---

## Approval Required

**Please confirm:**
1. ✅ This plan is acceptable
2. ✅ I may proceed with Phase 1 (Tests)
3. ✅ I will STOP before any training run >10 minutes for confirmation

**Reply with "APPROVED" to begin Phase 1.**
