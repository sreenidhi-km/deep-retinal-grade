# Testing Guide for Deep Retina Grade

## Overview

This project uses **pytest** for automated testing. The test suite covers:

- **Preprocessing**: Ben Graham preprocessing, CLAHE enhancement
- **Model Loading**: Model architecture, checkpoint loading, inference
- **API Contract**: Health check, prediction endpoints, response schema
- **XAI**: GradCAM, Integrated Gradients, LIME explainers

## Quick Start

### Install Test Dependencies

```bash
pip install pytest pytest-cov httpx
```

### Run All Tests

```bash
# From project root
pytest tests/ -v
```

### Run with Coverage Report

```bash
pytest tests/ --cov=src --cov=app --cov-report=term-missing
```

### Run Specific Test Files

```bash
# Preprocessing tests only
pytest tests/test_preprocessing.py -v

# Model tests only
pytest tests/test_model_loading.py -v

# API tests only
pytest tests/test_api_contract.py -v

# XAI tests only
pytest tests/test_xai_endpoints.py -v
```

## Test Structure

```
tests/
├── __init__.py              # Package marker
├── conftest.py              # Shared fixtures
├── test_preprocessing.py    # Preprocessing pipeline tests
├── test_model_loading.py    # Model loading and inference tests
├── test_api_contract.py     # FastAPI endpoint tests
└── test_xai_endpoints.py    # XAI functionality tests
```

## Key Fixtures (conftest.py)

| Fixture | Description |
|---------|-------------|
| `dummy_rgb_image` | 224×224 RGB numpy array simulating fundus image |
| `dummy_pil_image` | PIL Image version of dummy image |
| `dummy_image_bytes` | JPEG-encoded bytes for API testing |
| `dummy_tensor` | Normalized PyTorch tensor (1, 3, 224, 224) |
| `model_path` | Path to best model checkpoint |
| `device` | CPU device (M1 Mac compatible) |

## Test Categories

### Preprocessing Tests (`test_preprocessing.py`)

- Output shape validation (224×224×3)
- Output dtype (uint8) and value range [0, 255]
- Handling of edge cases (black, white, small, large images)
- CLAHE contrast enhancement
- Determinism verification

### Model Tests (`test_model_loading.py`)

- Model architecture creation
- Output dimensions (batch, 5)
- Checkpoint loading
- Inference determinism
- NaN/Inf output detection

### API Tests (`test_api_contract.py`)

- `/health` returns 200 with status
- `/predict` returns grade, confidence, probabilities
- Grade in range [0-4]
- Confidence in range [0-1]
- Probabilities sum to 1.0
- Error handling for invalid inputs

### XAI Tests (`test_xai_endpoints.py`)

- GradCAM heatmap generation
- Integrated Gradients attributions
- LIME explanations
- Shape and value validation

## Running Tests in CI/CD

```yaml
# GitHub Actions example
- name: Run Tests
  run: |
    pip install pytest pytest-cov httpx
    pytest tests/ --cov=src --cov=app --cov-report=xml
```

## Common Issues

### Model Not Found

If model tests are skipped with "Model file not found":
```bash
# Ensure model exists
ls models/efficientnet_b0_best.pth
```

### Import Errors

Ensure you're running from project root:
```bash
cd deep-retina-grade
pytest tests/ -v
```

### FastAPI Tests Skipped

Install missing dependencies:
```bash
pip install fastapi httpx python-multipart
```

## Minimum Coverage Targets

| Component | Target |
|-----------|--------|
| Preprocessing | 80% |
| Model | 70% |
| API | 70% |
| XAI | 60% |
| **Overall** | **75%** |

## Adding New Tests

1. Create test file in `tests/` directory
2. Import fixtures from `conftest.py`
3. Use `pytest.mark.skipif` for conditional tests
4. Follow naming convention: `test_*.py`

Example:
```python
import pytest

def test_new_feature(dummy_rgb_image):
    """Test description."""
    result = some_function(dummy_rgb_image)
    assert result is not None
```
