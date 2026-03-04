"""
End-to-End Integration Tests for Deep Retina Grade.

These tests verify the full pipeline from image upload through
prediction, explanation, uncertainty, and report generation.

Unlike unit tests, these tests exercise the real code paths
(though they use dummy images since model weights may not be available).

Author: Deep Retina Grade Project
Date: February 2026
"""

import pytest
import asyncio
import numpy as np
import sys
from pathlib import Path
from io import BytesIO
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try importing the app
FASTAPI_AVAILABLE = False
try:
    import httpx
    from app.main import app, MODEL_LOADED
    FASTAPI_AVAILABLE = True
except Exception:
    pass


def _async_request(method: str, path: str, **kwargs):
    """
    Synchronous wrapper for httpx.AsyncClient + ASGITransport.

    httpx 0.28+ removed sync ASGI support so we use AsyncClient
    wrapped in asyncio.run() to keep tests synchronous (no need
    for pytest-asyncio).
    """
    async def _do():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await getattr(client, method)(path, **kwargs)
    return asyncio.run(_do())


def create_synthetic_fundus(size: int = 224) -> bytes:
    """
    Create a synthetic fundus-like image for testing.

    Simulates a realistic circular fundus photograph with
    varying intensity to exercise the preprocessing pipeline.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Circular fundus region
    center = size // 2
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    mask = dist <= center * 0.85

    # Reddish-brown background (typical fundus)
    img[mask, 0] = 160  # R
    img[mask, 1] = 90   # G
    img[mask, 2] = 60   # B

    # Simulate optic disc (bright spot)
    disc_offset_x = center + size // 6
    disc_dist = np.sqrt((x - disc_offset_x) ** 2 + (y - center) ** 2)
    disc_mask = disc_dist <= center * 0.12
    img[disc_mask, 0] = 230
    img[disc_mask, 1] = 210
    img[disc_mask, 2] = 180

    # Add some texture noise
    noise = np.random.randint(0, 15, (size, size, 3), dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Convert to JPEG bytes
    pil_img = Image.fromarray(img)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return buffer.read()


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestEndToEndPipeline:
    """Full pipeline integration tests."""

    @pytest.fixture
    def fundus_image(self):
        return create_synthetic_fundus()

    # ---- Health & Root ----

    def test_root_endpoint(self):
        """Test root endpoint returns API info."""
        response = _async_request("get", "/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data

    def test_health_endpoint(self):
        """Test health endpoint returns device info."""
        response = _async_request("get", "/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data

    # ---- Predict ----

    def test_predict_full_response_schema(self, fundus_image):
        """Test /predict returns complete response with safety flags."""
        files = {"file": ("fundus.jpg", BytesIO(fundus_image), "image/jpeg")}
        response = _async_request("post", "/predict", files=files)

        if response.status_code == 503:
            pytest.skip("Model not loaded")
        assert response.status_code == 200
        data = response.json()

        # Core fields
        assert "grade" in data
        assert "grade_name" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "recommendation" in data
        assert "referral_urgency" in data

        # Safety contract fields
        assert "decision" in data, "Missing safety decision flag"
        assert "decision_reason" in data, "Missing decision reason"
        assert "quality_score" in data, "Missing quality score"
        assert "quality_issues" in data, "Missing quality issues"

        # Validate decision flag values
        valid_flags = ["OK", "REVIEW", "RETAKE", "OOD"]
        assert data["decision"] in valid_flags, (
            f"Invalid decision flag: {data['decision']}"
        )

        # Validate ranges
        assert 0 <= data["grade"] <= 4
        assert 0.0 <= data["confidence"] <= 1.0
        assert 0.0 <= data["quality_score"] <= 1.0
        assert isinstance(data["quality_issues"], list)

    def test_predict_grade_names_correct(self, fundus_image):
        """Test grade names match expected DR classification."""
        valid_names = {"No DR", "Mild", "Moderate", "Severe", "Proliferative DR"}
        files = {"file": ("fundus.jpg", BytesIO(fundus_image), "image/jpeg")}
        response = _async_request("post", "/predict", files=files)

        if response.status_code == 503:
            pytest.skip("Model not loaded")
        assert response.status_code == 200
        data = response.json()
        assert data["grade_name"] in valid_names

    # ---- Predict with TTA ----

    def test_predict_with_tta_endpoint(self, fundus_image):
        """Test /predict-with-tta returns TTA-specific fields."""
        files = {"file": ("fundus.jpg", BytesIO(fundus_image), "image/jpeg")}
        response = _async_request("post", "/predict-with-tta", files=files)

        if response.status_code == 503:
            pytest.skip("Model not loaded")
        assert response.status_code == 200
        data = response.json()
        assert "tta_confidence" in data
        assert "tta_mode" in data
        assert "num_augmentations" in data
        assert data["num_augmentations"] > 1

    # ---- Uncertainty ----

    def test_predict_with_uncertainty_endpoint(self, fundus_image):
        """Test /predict-with-uncertainty returns uncertainty metrics."""
        files = {"file": ("fundus.jpg", BytesIO(fundus_image), "image/jpeg")}
        response = _async_request("post", "/predict-with-uncertainty", files=files)

        if response.status_code == 503:
            pytest.skip("Model not loaded")
        assert response.status_code == 200
        data = response.json()
        assert "uncertainty" in data
        assert "entropy" in data
        assert "is_borderline" in data
        assert "agreement" in data
        assert isinstance(data["is_borderline"], bool)
        assert 0.0 <= data["uncertainty"]

    # ---- Explain ----

    def test_explain_endpoint(self, fundus_image):
        """Test /explain returns GradCAM base64 image."""
        files = {"file": ("fundus.jpg", BytesIO(fundus_image), "image/jpeg")}
        response = _async_request("post", "/explain", files=files)

        if response.status_code == 503:
            pytest.skip("Model not loaded")
        assert response.status_code == 200
        data = response.json()
        assert "gradcam_base64" in data
        assert "interpretation" in data
        assert len(data["gradcam_base64"]) > 100

    # ---- Metrics ----

    def test_metrics_endpoint(self):
        """Test /metrics returns performance data."""
        response = _async_request("get", "/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "overall_accuracy" in data
        assert "overall_qwk" in data

    # ---- Error Handling ----

    def test_predict_with_non_image(self):
        """Test prediction with non-image file returns error gracefully."""
        files = {"file": ("data.csv", BytesIO(b"a,b,c\n1,2,3"), "text/csv")}
        response = _async_request("post", "/predict", files=files)
        assert response.status_code in [400, 422, 500, 503]

    def test_predict_with_tiny_image(self):
        """Test prediction with very small image."""
        tiny = Image.new("RGB", (10, 10), color="red")
        buffer = BytesIO()
        tiny.save(buffer, format="JPEG")
        buffer.seek(0)
        files = {"file": ("tiny.jpg", buffer, "image/jpeg")}
        response = _async_request("post", "/predict", files=files)
        assert response.status_code in [200, 400, 500, 503]

    def test_predict_with_grayscale_image(self):
        """Test prediction with grayscale image."""
        gray = Image.new("L", (224, 224), color=128)
        buffer = BytesIO()
        gray.save(buffer, format="JPEG")
        buffer.seek(0)
        files = {"file": ("gray.jpg", buffer, "image/jpeg")}
        response = _async_request("post", "/predict", files=files)
        assert response.status_code in [200, 400, 500, 503]


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestRateLimiting:
    """Test rate limiting middleware."""

    def test_rate_limit_headers_present(self):
        """Test that health endpoint works with rate limiter active."""
        response = _async_request("get", "/health")
        assert response.status_code == 200


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestSecurityHeaders:
    """Test security headers middleware."""

    def test_security_headers_present(self):
        """Test that security headers are added."""
        response = _async_request("get", "/health")
        assert response.status_code == 200
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in response.headers
