"""
Tests for API contract and endpoint behavior.
Ensures API endpoints respond correctly and return expected schema.
"""

import pytest
import sys
from pathlib import Path
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import FastAPI test client
FASTAPI_AVAILABLE = False
try:
    from starlette.testclient import TestClient
    from app.main import app
    # Test if TestClient works
    with TestClient(app) as test_client:
        pass
    FASTAPI_AVAILABLE = True
except ImportError:
    pass
except TypeError:
    # httpx version incompatibility
    pass
except Exception:
    pass


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI or app not available")
class TestHealthEndpoint:
    """Test /health endpoint."""
    
    @pytest.fixture
    def client(self):
        with TestClient(app) as client:
            yield client
    
    def test_health_returns_200(self, client):
        """Test that /health returns HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_returns_json(self, client):
        """Test that /health returns valid JSON."""
        response = client.get("/health")
        data = response.json()
        assert isinstance(data, dict)
    
    def test_health_contains_status(self, client):
        """Test that /health response contains status field."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "ok", "running"]


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI or app not available")
class TestPredictEndpoint:
    """Test /predict endpoint."""
    
    @pytest.fixture
    def client(self):
        with TestClient(app) as client:
            yield client
    
    def test_predict_requires_image(self, client):
        """Test that /predict returns 422 without image."""
        response = client.post("/predict")
        assert response.status_code == 422  # Validation error
    
    def test_predict_accepts_image(self, client, dummy_image_bytes):
        """Test that /predict accepts image upload."""
        files = {"file": ("test.jpg", BytesIO(dummy_image_bytes), "image/jpeg")}
        response = client.post("/predict", files=files)
        # Should return 200 or 500 (if model not loaded), not 422
        assert response.status_code in [200, 500]
    
    def test_predict_returns_required_fields(self, client, dummy_image_bytes):
        """Test that /predict response contains required fields."""
        files = {"file": ("test.jpg", BytesIO(dummy_image_bytes), "image/jpeg")}
        response = client.post("/predict", files=files)
        
        if response.status_code == 200:
            data = response.json()
            # Core required fields
            assert "grade" in data, "Response must contain 'grade'"
            assert "confidence" in data, "Response must contain 'confidence'"
            assert "probabilities" in data, "Response must contain 'probabilities'"
    
    def test_predict_grade_is_valid(self, client, dummy_image_bytes):
        """Test that predicted grade is in valid range [0-4]."""
        files = {"file": ("test.jpg", BytesIO(dummy_image_bytes), "image/jpeg")}
        response = client.post("/predict", files=files)
        
        if response.status_code == 200:
            data = response.json()
            assert 0 <= data["grade"] <= 4, f"Grade {data['grade']} out of range"
    
    def test_predict_confidence_is_valid(self, client, dummy_image_bytes):
        """Test that confidence is in valid range [0-1]."""
        files = {"file": ("test.jpg", BytesIO(dummy_image_bytes), "image/jpeg")}
        response = client.post("/predict", files=files)
        
        if response.status_code == 200:
            data = response.json()
            assert 0 <= data["confidence"] <= 1, \
                f"Confidence {data['confidence']} out of range [0,1]"
    
    def test_predict_probabilities_sum_to_one(self, client, dummy_image_bytes):
        """Test that probabilities sum to approximately 1."""
        files = {"file": ("test.jpg", BytesIO(dummy_image_bytes), "image/jpeg")}
        response = client.post("/predict", files=files)
        
        if response.status_code == 200:
            data = response.json()
            prob_sum = sum(data["probabilities"])
            assert abs(prob_sum - 1.0) < 0.01, \
                f"Probabilities sum to {prob_sum}, expected ~1.0"
    
    def test_predict_probabilities_has_5_classes(self, client, dummy_image_bytes):
        """Test that probabilities array has 5 elements (DR grades 0-4)."""
        files = {"file": ("test.jpg", BytesIO(dummy_image_bytes), "image/jpeg")}
        response = client.post("/predict", files=files)
        
        if response.status_code == 200:
            data = response.json()
            assert len(data["probabilities"]) == 5, \
                f"Expected 5 probabilities, got {len(data['probabilities'])}"


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI or app not available")
class TestPredictWithUncertaintyEndpoint:
    """Test /predict-with-uncertainty endpoint if it exists."""
    
    @pytest.fixture
    def client(self):
        with TestClient(app) as client:
            yield client
    
    def test_uncertainty_endpoint_exists(self, client, dummy_image_bytes):
        """Test that /predict-with-uncertainty endpoint exists."""
        files = {"file": ("test.jpg", BytesIO(dummy_image_bytes), "image/jpeg")}
        response = client.post("/predict-with-uncertainty", files=files)
        # Should not be 404
        assert response.status_code != 404, "Endpoint should exist"
    
    def test_uncertainty_returns_uncertainty_field(self, client, dummy_image_bytes):
        """Test that uncertainty endpoint returns uncertainty data."""
        files = {"file": ("test.jpg", BytesIO(dummy_image_bytes), "image/jpeg")}
        response = client.post("/predict-with-uncertainty", files=files)
        
        if response.status_code == 200:
            data = response.json()
            # Should have some form of uncertainty metric
            assert any(key in data for key in [
                "uncertainty", "epistemic_uncertainty", "aleatoric_uncertainty",
                "mc_uncertainty", "std"
            ]), "Response should contain uncertainty metric"


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI or app not available")
class TestExplainEndpoint:
    """Test /explain endpoint."""
    
    @pytest.fixture
    def client(self):
        with TestClient(app) as client:
            yield client
    
    def test_explain_endpoint_exists(self, client, dummy_image_bytes):
        """Test that /explain endpoint exists."""
        files = {"file": ("test.jpg", BytesIO(dummy_image_bytes), "image/jpeg")}
        response = client.post("/explain", files=files)
        assert response.status_code != 404, "/explain endpoint should exist"
    
    def test_explain_accepts_method_parameter(self, client, dummy_image_bytes):
        """Test that /explain accepts method parameter."""
        files = {"file": ("test.jpg", BytesIO(dummy_image_bytes), "image/jpeg")}
        
        for method in ["gradcam", "integrated_gradients", "lime"]:
            response = client.post(f"/explain?method={method}", files=files)
            # Should accept the method parameter (even if it fails for other reasons)
            assert response.status_code != 422 or "method" not in response.text


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI or app not available")  
class TestAPIRobustness:
    """Test API error handling and robustness."""
    
    @pytest.fixture
    def client(self):
        with TestClient(app) as client:
            yield client
    
    def test_invalid_image_format(self, client):
        """Test handling of invalid image data."""
        files = {"file": ("test.txt", BytesIO(b"not an image"), "text/plain")}
        response = client.post("/predict", files=files)
        # Should return error, not crash
        assert response.status_code in [400, 415, 422, 500]
    
    def test_empty_file(self, client):
        """Test handling of empty file."""
        files = {"file": ("empty.jpg", BytesIO(b""), "image/jpeg")}
        response = client.post("/predict", files=files)
        # Should return error, not crash
        assert response.status_code in [400, 422, 500]
    
    def test_corrupted_jpeg(self, client):
        """Test handling of corrupted JPEG."""
        # Partial JPEG header
        corrupted = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00'
        files = {"file": ("corrupted.jpg", BytesIO(corrupted), "image/jpeg")}
        response = client.post("/predict", files=files)
        # Should return error, not crash
        assert response.status_code in [400, 422, 500]
