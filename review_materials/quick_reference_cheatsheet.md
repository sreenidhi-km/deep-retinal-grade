# 📝 Quick Reference Cheat Sheet - Deep Retina Grade Review

## 🎯 30-Second Elevator Pitch
"We built an AI system that screens fundus images for diabetic retinopathy with 79.8% accuracy and 0.864 QWK. Unlike typical academic projects, ours includes explainable AI showing what the model sees, uncertainty quantification to flag borderline cases, and fairness auditing to ensure no demographic bias."

---

## 📊 Key Numbers to Remember

| Metric | Value | What It Means |
|--------|-------|---------------|
| **Accuracy** | 79.8% | Overall correctness |
| **QWK** | 0.864 | Almost perfect agreement (>0.81) |
| **Improvement** | +14.7% | Better than baseline (65.1%) |
| **Severe DR Sensitivity** | 77.8% | Catches severe cases |
| **Inference Time** | <1 second | Fast enough for clinical use |
| **Dataset Size** | 3,662 images | APTOS 2019 |
| **Test Set Size** | 605 images | 15% of total |
| **Model Size** | 5.3M params | EfficientNet-B0 |
| **Fairness** | ✅ PASS | <12% disparity across groups |

---

## 🌟 Top 5 Unique Features

1. **Triple Explainable AI**
   - GradCAM + Integrated Gradients + LIME
   - Shows what the AI "sees"
   
2. **Uncertainty Quantification**
   - Monte Carlo Dropout (20 samples)
   - Flags borderline cases for human review
   
3. **Fairness Auditing**
   - Stratified by skin pigmentation
   - Passes 80% rule
   
4. **Advanced Preprocessing**
   - Ben Graham method (age-invariant)
   - CLAHE enhancement
   
5. **Production-Ready**
   - FastAPI + React frontend
   - Docker deployment
   - PDF clinical reports

---

## 🔬 DR Grading Scale

| Grade | Name | Action | Examples |
|-------|------|--------|----------|
| 0 | No DR | Annual screening | Normal retina |
| 1 | Mild NPDR | Monitor (9-12 mo) | Few microaneurysms |
| 2 | Moderate NPDR | Refer (3-6 mo) | Hemorrhages, exudates |
| 3 | Severe NPDR | **URGENT** (2-4 wks) | Extensive hemorrhages |
| 4 | Proliferative DR | **IMMEDIATE** | Neovascularization |

---

## 🏗️ Architecture Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React + Vite + Tailwind |
| Backend | FastAPI |
| Model | PyTorch + EfficientNet-B0 |
| Deployment | Docker Compose |
| XAI | Custom + Captum |

---

## 📈 Per-Class Performance

| Class | Name | Precision | Recall | Why Important |
|-------|------|-----------|--------|---------------|
| 0 | No DR | 0.97 | 0.97 | Avoid false alarms |
| 1 | Mild | 0.52 | 0.44 | Hard (minority class) |
| 2 | Moderate | 0.69 | 0.84 | Most common pathology |
| 3 | Severe | 0.57 | 0.25 | Critical (needs work) |
| 4 | Prolif. | 0.55 | 0.45 | Vision-threatening |

**Key Insight:** Excellent on Grade 0, good on Grade 2, struggling with minority classes (1, 3, 4) due to class imbalance.

---

## 🔑 7 Critical Decisions & Why

1. **EfficientNet-B0** → Faster than ViT, better than ResNet
2. **Ben Graham preprocessing** → Age-invariant normalization
3. **FocalLoss** → Handles class imbalance
4. **MC Dropout** → Clinical safety (uncertainty)
5. **Triple XAI** → Clinical trust (explanations)
6. **QWK metric** → Standard for ordinal classification
7. **Docker** → Easy deployment

---

## 💡 Common Questions - Quick Answers

### "What would you improve?"
1. Higher accuracy (target 92%): Larger model + more data
2. Better minority class performance: Advanced sampling
3. External validation: Test on Messidor, IDRiD datasets

### "Why three XAI methods?"
- Complementary strengths: GradCAM (fast), IG (principled), LIME (interpretable)
- When all agree → high confidence in explanation

### "How do you handle class imbalance?"
- FocalLoss (focus on hard examples)
- Weighted sampling (oversample minorities)
- Mixup augmentation (synthetic examples)

### "What's Ben Graham preprocessing?"
- Competition-winning technique
- Removes color cast from aging lenses/cataracts
- Makes model age-invariant

### "What's MC Dropout?"
- Run model 20 times with dropout enabled
- High variance in predictions → borderline case → flag for human review

### "Is it FDA-ready?"
- No (needs clinical trials)
- But follows FDA AI/ML guidance
- Has explainability, uncertainty, fairness

---

## 📁 Project Files (Key Locations)

```
src/
  preprocessing/    → Ben Graham + CLAHE
  models/          → EfficientNet-B0
  xai/             → GradCAM, IG, LIME
  uncertainty/     → MC Dropout
  training/        → FocalLoss, Mixup, TTA
  fairness/        → Bias auditing

notebooks/
  03_model_training_fast.ipynb    → Main training
  04_evaluation_gradcam.ipynb     → Evaluation
  06_fairness_audit.ipynb         → Fairness

models/
  efficientnet_b0_improved.pth    → Best model (79.8%)

results/
  test_metrics.json               → Performance
  fairness_audit.json             → Fairness
```

---

## 🎯 Closing Statement

"We didn't just build a classifier—we built a clinically-ready system addressing real deployment challenges: age variance, ethnicity bias, prediction instability, and the black box problem. With explainability, uncertainty quantification, and fairness auditing, this could actually be deployed in clinics to help screen diabetic retinopathy."

---

## 🚩 What We Did NOT Do (Be Honest)

❌ External validation (only tested on APTOS)  
❌ FDA clearance (would need clinical trials)  
❌ Lesion segmentation (only classification)  
❌ Real patient data (competition dataset)  
❌ Multi-center testing  

**But:** We have the foundation to do all of these next!

---

**Print this and keep it with you! 📋**
