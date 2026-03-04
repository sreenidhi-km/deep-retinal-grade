# Deep Retina Grade - Usage Guide

Complete installation and usage instructions for the Deep Retina Grade diabetic retinopathy detection system.

---

## Table of Contents

1. [Installation](#installation)
2. [Preprocessing Images](#preprocessing-images)
3. [Running Inference](#running-inference)
4. [Using the API](#using-the-api)
5. [Running Tests](#running-tests)
6. [Docker Deployment](#docker-deployment)

---

## Installation

### Prerequisites

- Python 3.11 or higher
- 8GB RAM minimum (16GB recommended)
- macOS with Apple Silicon (MPS) or NVIDIA GPU (CUDA)

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/deep-retina-grade.git
cd deep-retina-grade

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
```

### Verify Model Weights

```bash
# Check that the trained model exists
ls -la models/efficientnet_b0_combined.pth
# Should show: ~17MB file
```

---

## Preprocessing Images

### Using the Preprocessing Module

The system uses Ben Graham's preprocessing + CLAHE enhancement for optimal results.

```python
from src.preprocessing.preprocess import preprocess_image, ben_graham_preprocess
import numpy as np
from PIL import Image

# Method 1: Full preprocessing pipeline
preprocessed = preprocess_image(
    image_path="path/to/fundus_image.jpg",
    size=224,  # Must match training size
    apply_clahe=True
)
# Returns: numpy array of shape (224, 224, 3), normalized to [0, 1]

# Method 2: Ben Graham preprocessing only
image = np.array(Image.open("path/to/fundus_image.jpg"))
processed = ben_graham_preprocess(image, sigmaX=10)
```

### Preprocessing New Dataset

```python
import os
from pathlib import Path
from tqdm import tqdm
from src.preprocessing.preprocess import preprocess_image
import numpy as np

# Preprocess and cache a directory of images
input_dir = Path("path/to/raw_images")
output_dir = Path("cache/preprocessed_224")
output_dir.mkdir(parents=True, exist_ok=True)

for img_path in tqdm(list(input_dir.glob("*.jpg"))):
    processed = preprocess_image(str(img_path), size=224)
    np.save(output_dir / f"{img_path.stem}.npy", processed)
```

---

## Running Inference

### Single Image Inference

```python
import torch
import numpy as np
from PIL import Image
from src.models.efficientnet import RetinaModel
from src.preprocessing.preprocess import preprocess_image

# Configuration
MODEL_PATH = "models/efficientnet_b0_combined.pth"
IMAGE_PATH = "path/to/fundus_image.jpg"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Grade mapping
GRADE_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR", 
    3: "Severe NPDR",
    4: "Proliferative DR"
}

# Load model
model = RetinaModel(num_classes=5, pretrained=False)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# Preprocess image
image = preprocess_image(IMAGE_PATH, size=224)
image_tensor = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float()
image_tensor = image_tensor.to(DEVICE)

# Run inference
with torch.no_grad():
    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)
    predicted_grade = probabilities.argmax(dim=1).item()
    confidence = probabilities.max().item()

# Output
print(f"Predicted Grade: {predicted_grade} ({GRADE_NAMES[predicted_grade]})")
print(f"Confidence: {confidence:.1%}")
print(f"\nAll probabilities:")
for i, prob in enumerate(probabilities[0].cpu().numpy()):
    print(f"  Grade {i} ({GRADE_NAMES[i]}): {prob:.1%}")
```

### Batch Inference

```python
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm

class FundusDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load preprocessed .npy file
        img = np.load(self.image_paths[idx])
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img, self.image_paths[idx].stem

# Setup
image_paths = list(Path("cache/preprocessed_224").glob("*.npy"))
dataset = FundusDataset(image_paths)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

# Run batch inference
all_predictions = []
all_confidences = []
all_image_ids = []

model.eval()
with torch.no_grad():
    for images, image_ids in tqdm(dataloader, desc="Inference"):
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        confs = probs.max(dim=1).values
        
        all_predictions.extend(preds.cpu().numpy())
        all_confidences.extend(confs.cpu().numpy())
        all_image_ids.extend(image_ids)

# Results DataFrame
import pandas as pd
results = pd.DataFrame({
    'image_id': all_image_ids,
    'predicted_grade': all_predictions,
    'confidence': all_confidences
})
results.to_csv("predictions.csv", index=False)
```

### Inference with Uncertainty (MC Dropout)

```python
from src.uncertainty.mc_dropout import MCDropoutPredictor

# Initialize predictor
mc_predictor = MCDropoutPredictor(model, n_samples=20, device=DEVICE)

# Get prediction with uncertainty
result = mc_predictor.predict_with_uncertainty(image_tensor)

print(f"Predicted Grade: {result['predicted_grade']} ({GRADE_NAMES[result['predicted_grade']]})")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Uncertainty (std): {result['uncertainty']:.4f}")
print(f"Is Borderline: {result['is_borderline']}")
```

---

## Using the API

### Starting the Server

```bash
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Basic prediction |
| `/predict-with-tta` | POST | TTA-enhanced prediction |
| `/predict-with-uncertainty` | POST | Prediction + MC Dropout |
| `/explain` | POST | GradCAM explanation |
| `/generate-report` | POST | Generate PDF report |

### Example: Python Client

```python
import requests

BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Predict with uncertainty
with open("fundus_image.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/predict-with-uncertainty",
        files={"file": f}
    )

result = response.json()
print(f"Grade: {result['predicted_grade']} ({result['grade_name']})")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Uncertainty: {result['uncertainty']:.4f}")
print(f"Needs Review: {result['needs_human_review']}")

# Safety flags
if result.get('safety_flags'):
    print("\n⚠️ Safety Flags:")
    for flag in result['safety_flags']:
        print(f"  - {flag}")
```

### Example: cURL

```bash
# Basic prediction
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@fundus_image.jpg"

# Prediction with uncertainty
curl -X POST "http://localhost:8000/predict-with-uncertainty" \
  -H "accept: application/json" \
  -F "file=@fundus_image.jpg"
```

---

## Running Tests

### Run Full Test Suite

```bash
# From project root
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # View coverage report
```

### Run Specific Test Categories

```bash
# Preprocessing tests
pytest tests/test_preprocessing.py -v

# Model tests
pytest tests/test_model.py -v

# API tests
pytest tests/test_api.py -v

# Safety contract tests
pytest tests/test_safety_contract.py -v
```

### Test Output Example

```
tests/test_preprocessing.py::test_preprocess_image_shape PASSED
tests/test_preprocessing.py::test_ben_graham_output_range PASSED
tests/test_model.py::test_model_output_shape PASSED
tests/test_model.py::test_model_forward_pass PASSED
tests/test_api.py::test_health_endpoint PASSED
tests/test_safety_contract.py::test_low_confidence_flag PASSED
...
========================= 69 passed in 45.23s =========================
```

---

## Docker Deployment

### Build and Run

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Access Points

- **API Backend:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Frontend:** http://localhost:5173

### Docker Compose Configuration

The `docker-compose.yml` includes:
- `backend`: FastAPI server with model
- `frontend`: React + Vite web interface

---

## Troubleshooting

### Common Issues

**1. Model file not found**
```
FileNotFoundError: models/efficientnet_b0_combined.pth
```
Solution: Ensure model weights are in the `models/` directory.

**2. MPS memory error (Apple Silicon)**
```
RuntimeError: MPS backend out of memory
```
Solution: Reduce batch size or use CPU: `DEVICE = "cpu"`

**3. CUDA out of memory**
```
RuntimeError: CUDA out of memory
```
Solution: Reduce batch size or use `torch.cuda.empty_cache()`

**4. Import errors**
```
ModuleNotFoundError: No module named 'src'
```
Solution: Run from project root or add to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Performance Tips

1. **Use cached preprocessed images** for repeated inference
2. **Batch processing** is 10x faster than single-image inference
3. **GPU/MPS acceleration** provides 5-20x speedup over CPU
4. **Reduce MC Dropout samples** (n=10 vs n=20) for faster uncertainty estimation

---

*For more information, see the [main README](../README.md) and [API documentation](http://localhost:8000/docs).*
