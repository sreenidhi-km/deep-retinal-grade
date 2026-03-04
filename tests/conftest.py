"""
Pytest fixtures for Deep Retina Grade test suite.
"""

import pytest
import numpy as np
import torch
import sys
import os
from pathlib import Path
from io import BytesIO
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def device():
    """Return the compute device (CPU for M1 Mac compatibility)."""
    return torch.device("cpu")


@pytest.fixture
def dummy_rgb_image():
    """Create a dummy RGB image as numpy array (224x224x3)."""
    # Create a realistic-ish fundus-like image with circular mask
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Create circular fundus-like region
    center = (112, 112)
    radius = 100
    y, x = np.ogrid[:224, :224]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    
    # Fill with reddish-brown color (typical fundus)
    img[mask, 0] = 180  # R
    img[mask, 1] = 100  # G
    img[mask, 2] = 80   # B
    
    # Add some noise
    noise = np.random.randint(0, 20, (224, 224, 3), dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise - 10, 0, 255).astype(np.uint8)
    
    return img


@pytest.fixture
def dummy_pil_image(dummy_rgb_image):
    """Create a dummy PIL Image."""
    return Image.fromarray(dummy_rgb_image)


@pytest.fixture
def dummy_image_bytes(dummy_pil_image):
    """Create dummy image as bytes (JPEG format)."""
    buffer = BytesIO()
    dummy_pil_image.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def dummy_tensor(device):
    """Create a dummy normalized image tensor (1, 3, 224, 224)."""
    tensor = torch.randn(1, 3, 224, 224)
    return tensor.to(device)


@pytest.fixture
def black_image():
    """Create a completely black image (for edge case testing)."""
    return np.zeros((224, 224, 3), dtype=np.uint8)


@pytest.fixture
def white_image():
    """Create a completely white image (for edge case testing)."""
    return np.ones((224, 224, 3), dtype=np.uint8) * 255


@pytest.fixture
def small_image():
    """Create a small image (for resize testing)."""
    return np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)


@pytest.fixture
def large_image():
    """Create a large image (for resize testing)."""
    return np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def model_path(project_root):
    """Return path to best model checkpoint."""
    path = project_root / "models" / "efficientnet_b0_best.pth"
    return path


@pytest.fixture(scope="session")
def model_exists(model_path):
    """Check if model file exists."""
    return model_path.exists()


@pytest.fixture
def sample_probabilities():
    """Sample probability distribution for 5 DR grades."""
    probs = np.array([0.6, 0.2, 0.1, 0.05, 0.05])
    return probs


@pytest.fixture
def sample_predictions():
    """Sample prediction data matching API contract."""
    return {
        "grade": 0,
        "confidence": 0.85,
        "probabilities": [0.85, 0.10, 0.03, 0.01, 0.01],
        "grade_name": "No DR"
    }
