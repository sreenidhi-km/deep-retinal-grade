# 🔬 Deep Retina Grade

### AI-Powered Diabetic Retinopathy Grading System with Explainable AI

[![CI](https://github.com/Shivasai132678/deep-retina-grade/actions/workflows/ci.yml/badge.svg)](https://github.com/Shivasai132678/deep-retina-grade/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-82%20passed-brightgreen.svg)](tests/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Version](https://img.shields.io/badge/version-2.0.0-informational.svg)]()

<p align="center">
  <img src="artifacts/architecture_diagram.png" alt="Deep Retina Grade Architecture" width="600">
</p>

---

## 🎯 Overview

Deep Retina Grade is a production-ready deep learning system for automated diabetic retinopathy (DR) screening. Built with clinical deployment in mind, it addresses real-world challenges that cause AI models to fail in practice:

| Problem | Our Solution | Impact |
|---------|-------------|--------|
| **Age Variance** | Ben Graham + CLAHE preprocessing | Works equally on 20 and 80-year-old patients |
| **Ethnicity Bias** | Pigmentation-stratified fairness audit | <12% disparity across skin tones |
| **Prediction Instability** | Test-Time Augmentation (TTA) | Stable grades instead of flickering predictions |
| **Black Box Problem** | Triple XAI (GradCAM + IG + LIME) | Doctors see exactly what the AI sees |
| **Overconfident Errors** | MC Dropout Uncertainty | Flags borderline cases for human review |

---

## 📊 Performance Summary

### Internal Validation (APTOS Test Set)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Quadratic Weighted Kappa** | **0.8745** | ≥0.85 | ✅ |
| **Overall Accuracy** | **78.5%** | — | — |
| **Severe DR Recall (Grade 3)** | **38%** | ≥60% | ⚠️ |
| **Fairness (80% Rule)** | PASS | Pass | ✅ |

### ⚠️ External Validation (IDRiD Dataset)

| Metric | APTOS (Train) | IDRiD (External) | Drop |
|--------|---------------|------------------|------|
| **QWK** | 0.8745 | **0.4898** | -36% |
| **Accuracy** | 78.5% | 45.5% | -33% |
| **Grade 3 Recall** | 38% | 10.8% | -27% |

> **⚠️ CRITICAL:** The model shows significant **domain shift** on external data. IDRiD uses different camera equipment and demographics (Indian population). See [docs/GENERALIZATION.md](docs/GENERALIZATION.md) for full analysis. **Do not deploy on populations different from training data without fine-tuning.**

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone and setup
git clone https://github.com/Shivasai132678/deep-retina-grade.git
cd deep-retina-grade
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Create .env config (one-time setup)
python setup_env.py
```

### 2. Run Inference on a Single Image

```python
import torch
from PIL import Image
from src.models.efficientnet import RetinaModel
from src.preprocessing.preprocess import preprocess_image

# Load model
model = RetinaModel(num_classes=5, pretrained=False)
checkpoint = torch.load("models/efficientnet_b0_combined.pth", map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Preprocess and predict
image = preprocess_image("path/to/fundus_image.jpg", size=224)
image_tensor = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float()

with torch.no_grad():
    logits = model(image_tensor)
    predicted_grade = logits.argmax(dim=1).item()
    confidence = torch.softmax(logits, dim=1).max().item()

grade_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
print(f"Grade: {predicted_grade} ({grade_names[predicted_grade]})")
print(f"Confidence: {confidence:.1%}")
```

### 3. Run the API Server

```bash
cd app && uvicorn main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### 4. Run with Docker

```bash
docker-compose up --build
# API: http://localhost:8000 | Frontend: http://localhost:5173
```

For detailed usage instructions, see [docs/USAGE.md](docs/USAGE.md).

---

## 📁 Project Structure

```
deep-retina-grade/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Multi-service deployment
│
├── src/                         # Core source code
│   ├── models/                  # EfficientNet-B0 + Ensemble predictor
│   ├── preprocessing/           # Ben Graham + CLAHE + Quality assessment
│   ├── training/                # FocalLoss, CORAL, Mixup, TTA, Calibration
│   ├── xai/                     # GradCAM, Integrated Gradients, LIME
│   ├── uncertainty/             # Monte Carlo Dropout
│   └── fairness/                # Bias auditing
│
├── app/                         # FastAPI backend + React frontend
│   ├── main.py                  # API endpoints with safety contract
│   ├── middleware.py             # Rate limiting, logging, security headers
│   └── frontend/                # Vite + React + TailwindCSS
│
├── models/                      # Trained weights
│   └── efficientnet_b0_combined.pth  # Best model (QWK 0.8745)
│
├── scripts/                     # Training and validation scripts
│   ├── train_improved.py        # Phase 3 training
│   └── validate_cross_dataset.py # Phase 4 external validation
│
├── tests/                       # pytest suite (82 tests)
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_api.py
│   └── test_safety_contract.py
│
├── notebooks/                   # Jupyter notebooks (exploration)
│   ├── 01_data_preparation.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation_gradcam.ipynb
│   ├── 05_xai_advanced.ipynb
│   └── 06_fairness_audit.ipynb
│
├── results/                     # Metrics and evaluation results
│   ├── test_metrics.json
│   ├── external_validation.json # Phase 4 IDRiD results
│   └── fairness_audit.json
│
└── docs/                        # Documentation
    ├── USAGE.md                 # Installation & usage guide
    ├── PROJECT_REPORT.md        # Complete project summary
    ├── GENERALIZATION.md        # External validation analysis
    ├── phase3_summary.md        # Grade 3 improvement analysis
    └── LOCKED_PLAN.md           # Original implementation plan
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Grade with safety decision flags (OK/REVIEW/RETAKE/OOD) |
| `/predict-with-tta` | POST | TTA-enhanced prediction (+2-3% accuracy) |
| `/predict-ensemble` | POST | Multi-model ensemble prediction |
| `/explain` | POST | Get GradCAM explanation |
| `/predict-with-uncertainty` | POST | Prediction + MC Dropout uncertainty |
| `/generate-report` | POST | Generate PDF clinical report |
| `/metrics` | GET | Model performance metrics |

### Safety Decision Flags

Every `/predict` response includes a `decision` field:

| Flag | Meaning | Action |
|------|---------|--------|
| `OK` | High confidence, good quality | Safe for clinical guidance |
| `REVIEW` | Borderline confidence or severe finding | Clinician must review |
| `RETAKE` | Poor image quality | Request new image |
| `OOD` | Out-of-distribution | Image may not be a fundus photo |

### Example: Predict with Uncertainty

```python
import requests

url = "http://localhost:8000/predict-with-uncertainty"
files = {"file": open("fundus_image.png", "rb")}
response = requests.post(url, files=files)

result = response.json()
print(f"Grade: {result['predicted_grade']} ({result['grade_name']})")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Uncertainty: {result['uncertainty']:.3f}")
print(f"Borderline: {result['is_borderline']}")
```

---

## 🔬 Explainability (XAI)

The system provides three complementary explanation methods:

### 1. GradCAM (Gradient-weighted Class Activation Mapping)
- **What:** Regional heatmap from final convolutional layer
- **Good for:** Fast, intuitive localization
- **Shows:** "Which regions influenced the prediction"

### 2. Integrated Gradients
- **What:** Pixel-level attribution via path integration
- **Good for:** Theoretically grounded, fine-grained details
- **Shows:** "Which pixels matter most"

### 3. LIME (Local Interpretable Model-agnostic Explanations)
- **What:** Superpixel importance via perturbation
- **Good for:** Human-interpretable regions
- **Shows:** "Which segments drive the decision"

When all three methods agree on the same region → **High confidence** in the explanation.

---

## ⚖️ Fairness Audit

We stratify evaluation by skin pigmentation (proxy: LAB luminance):

| Metric | Light | Medium | Dark | Disparity |
|--------|-------|--------|------|-----------|
| Accuracy | 72.8% | 60.8% | 61.5% | 12.0% |
| QWK | 0.830 | 0.792 | 0.800 | 3.8% |
| Sensitivity | 63.2% | 70.8% | 75.3% | 12.0% |

**80% Rule Status:** ✅ PASS (ratio > 0.8 for all metrics)

---

## 📋 Clinical Recommendations

| Grade | Name | Recommendation | Follow-up |
|-------|------|----------------|-----------|
| 0 | No DR | Continue annual screening | 12 months |
| 1 | Mild NPDR | Optimize diabetes control | 9-12 months |
| 2 | Moderate NPDR | Refer to ophthalmologist | 3-6 months |
| 3 | Severe NPDR | **URGENT** referral | 2-4 weeks |
| 4 | Proliferative DR | **IMMEDIATE** specialist referral | Now |

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📚 References

### Key Papers
1. Sundararajan et al., "Axiomatic Attribution for Deep Networks" (ICML 2017) - Integrated Gradients
2. Ribeiro et al., "Why Should I Trust You?" (KDD 2016) - LIME
3. Selvaraju et al., "Grad-CAM" (ICCV 2017) - GradCAM
4. Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016) - MC Dropout

### Dataset
- APTOS 2019 Blindness Detection Challenge (Kaggle)
- 3,662 high-resolution fundus images with 5-class grading

---

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Limitations & Disclaimer

### Known Limitations

1. **Domain Shift:** Model performance drops significantly on external data (QWK 0.87→0.49 on IDRiD). Fine-tuning required for new populations.

2. **Grade 3 Under-Detection:** Severe DR recall is 38% (target was 60%). 46.9% of Grade 3 cases are misclassified as Moderate due to adjacent-class confusion.

3. **Compute Constraints:** Trained on MacBook Air M1 (8GB RAM). Production deployment should use:
   - Larger models (EfficientNet-B3+)
   - Higher resolution (384×384)
   - Ordinal regression with CORAL loss

4. **Single Dataset Training:** Primarily trained on APTOS 2019 data (Asian population, specific camera equipment).

### Clinical Disclaimer

**This system is intended as a clinical decision support tool only.** It should not be used as the sole basis for diagnosis or treatment decisions. Final clinical decisions must be made by qualified healthcare providers. The system has been developed following FDA guidance on AI/ML medical devices but has not been FDA-cleared.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [docs/USAGE.md](docs/USAGE.md) | Installation and usage guide |
| [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) | Complete project summary |
| [docs/GENERALIZATION.md](docs/GENERALIZATION.md) | External validation analysis |
| [docs/phase3_summary.md](docs/phase3_summary.md) | Grade 3 improvement analysis |
| [docs/TESTING.md](docs/TESTING.md) | Test suite documentation |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Technical architecture |

---

## � Citation

If you use this project in your research, please cite:

```bibtex
@software{deep_retina_grade_2026,
  author = {Vemula, Shivasai},
  title = {Deep Retina Grade: AI-Powered Diabetic Retinopathy Grading with Explainable AI},
  year = {2026},
  url = {https://github.com/Shivasai132678/deep-retina-grade},
  version = {1.0.0}
}
```

---

## 👥 Author

**Shivasai Vemula**  
GitHub: [@Shivasai132678](https://github.com/Shivasai132678)

---

## 🙏 Acknowledgments

### Datasets
- **APTOS 2019 Blindness Detection Challenge** - Primary training dataset (Kaggle)
- **IDRiD (Indian Diabetic Retinopathy Image Dataset)** - External validation dataset
- **DDR Dataset** - Supplementary training data

### Libraries & Frameworks
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [timm](https://github.com/huggingface/pytorch-image-models) - EfficientNet implementation
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
- [Captum](https://captum.ai/) - Model interpretability
- [LIME](https://github.com/marcotcr/lime) - Local explanations

### Research References
- Gulshan et al., "Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy" (JAMA 2016)
- Ben Graham, "Kaggle Diabetic Retinopathy Detection Competition" - Preprocessing technique
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (ICCV 2017)
- Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016) - MC Dropout
- Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017) - Temperature Scaling
- Cao et al., "Rank Consistent Ordinal Regression (CORAL)" (Pattern Recognition Letters, 2020)

---

## 🔧 Configuration

All thresholds are configurable via `.env` (see [.env.example](.env.example)):

| Variable | Default | Description |
|----------|---------|-------------|
| `DR_DEMO_MODE` | `false` | Skip quality/uncertainty checks |
| `DR_QUALITY_THRESHOLD` | `0.4` | Min quality before RETAKE flag |
| `DR_UNCERTAINTY_THRESHOLD` | `0.15` | Max uncertainty before REVIEW flag |
| `DR_OOD_ENTROPY_THRESHOLD` | `1.5` | Max entropy before OOD flag |
| `DR_MC_SAMPLES` | `5` | MC Dropout forward passes |
| `DR_RATE_LIMIT_MAX` | `60` | Max requests per window per IP |
| `DR_ENSEMBLE_ENABLED` | `false` | Enable multi-checkpoint ensemble |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Built with ❤️ for improving diabetic retinopathy screening worldwide</b>
  <br>
  <sub>v2.0.0 | EfficientNet-B0 + GradCAM + MC Dropout + TTA + CORAL</sub>
</p>

