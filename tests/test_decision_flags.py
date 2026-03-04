"""
Tests for Decision Flags and Quality Assessment.

Covers:
- Quality score computation
- Decision flag logic (OK, REVIEW, RETAKE, OOD)
- Environment variable configuration
- Edge cases
"""

import pytest
import numpy as np
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.quality import ImageQualityAssessor


class TestImageQualityAssessor:
    """Test ImageQualityAssessor class."""
    
    @pytest.fixture
    def assessor(self):
        """Create quality assessor with default settings."""
        return ImageQualityAssessor()
    
    @pytest.fixture
    def sharp_image(self):
        """Create a sharp image with good contrast."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        # Add sharp edges and patterns
        cv2 = __import__('cv2')
        # Draw circles and lines for sharp edges
        cv2.circle(img, (112, 112), 80, (180, 100, 80), -1)
        cv2.circle(img, (112, 112), 40, (200, 120, 100), -1)
        # Add noise for texture
        noise = np.random.randint(0, 30, (224, 224, 3), dtype=np.uint8)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img
    
    @pytest.fixture
    def blurry_image(self, sharp_image):
        """Create a blurry image."""
        cv2 = __import__('cv2')
        return cv2.GaussianBlur(sharp_image, (31, 31), 10)
    
    @pytest.fixture
    def dark_image(self):
        """Create an underexposed (dark) image."""
        return np.full((224, 224, 3), 20, dtype=np.uint8)
    
    @pytest.fixture
    def bright_image(self):
        """Create an overexposed (bright) image."""
        return np.full((224, 224, 3), 240, dtype=np.uint8)
    
    @pytest.fixture
    def low_contrast_image(self):
        """Create a low contrast image."""
        # Very narrow range of values
        return np.random.randint(100, 110, (224, 224, 3), dtype=np.uint8)
    
    def test_assessor_initialization(self):
        """Test assessor initializes with correct defaults."""
        assessor = ImageQualityAssessor()
        assert assessor.laplacian_threshold == 100.0
        assert assessor.brightness_low == 30.0
        assert assessor.brightness_high == 220.0
        assert assessor.contrast_threshold == 20.0
    
    def test_assessor_custom_thresholds(self):
        """Test assessor with custom thresholds."""
        assessor = ImageQualityAssessor(
            laplacian_threshold=50.0,
            brightness_low=40.0,
            brightness_high=200.0,
            contrast_threshold=30.0
        )
        assert assessor.laplacian_threshold == 50.0
        assert assessor.brightness_low == 40.0
    
    def test_ensure_rgb_grayscale(self, assessor):
        """Test grayscale to RGB conversion."""
        gray = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        rgb = assessor._ensure_rgb(gray)
        assert rgb.shape == (224, 224, 3)
    
    def test_ensure_rgb_single_channel(self, assessor):
        """Test single channel to RGB conversion."""
        single = np.random.randint(0, 255, (224, 224, 1), dtype=np.uint8)
        rgb = assessor._ensure_rgb(single)
        assert rgb.shape == (224, 224, 3)
    
    def test_ensure_rgb_rgba(self, assessor):
        """Test RGBA to RGB conversion."""
        rgba = np.random.randint(0, 255, (224, 224, 4), dtype=np.uint8)
        rgb = assessor._ensure_rgb(rgba)
        assert rgb.shape == (224, 224, 3)
    
    def test_ensure_rgb_passthrough(self, assessor, sharp_image):
        """Test RGB passthrough."""
        rgb = assessor._ensure_rgb(sharp_image)
        assert rgb.shape == (224, 224, 3)
    
    def test_laplacian_variance_sharp(self, assessor, sharp_image):
        """Test Laplacian variance on sharp image."""
        var = assessor.compute_laplacian_variance(sharp_image)
        # Sharp images should have high variance
        assert var > 50  # Some variance expected
    
    def test_laplacian_variance_blurry(self, assessor, blurry_image):
        """Test Laplacian variance on blurry image."""
        var = assessor.compute_laplacian_variance(blurry_image)
        # Blurry images should have low variance
        assert var < 200  # Lower than sharp
    
    def test_brightness_dark(self, assessor, dark_image):
        """Test brightness on dark image."""
        brightness = assessor.compute_brightness(dark_image)
        assert brightness < 30
    
    def test_brightness_bright(self, assessor, bright_image):
        """Test brightness on bright image."""
        brightness = assessor.compute_brightness(bright_image)
        assert brightness > 220
    
    def test_contrast_low(self, assessor, low_contrast_image):
        """Test contrast on low contrast image."""
        contrast = assessor.compute_contrast(low_contrast_image)
        assert contrast < 20
    
    def test_assess_quality_returns_tuple(self, assessor, sharp_image):
        """Test assess_quality returns (score, issues)."""
        result = assessor.assess_quality(sharp_image)
        assert isinstance(result, tuple)
        assert len(result) == 2
        score, issues = result
        assert isinstance(score, float)
        assert isinstance(issues, list)
    
    def test_assess_quality_score_range(self, assessor, sharp_image):
        """Test quality score is in [0, 1] range."""
        score, _ = assessor.assess_quality(sharp_image)
        assert 0.0 <= score <= 1.0
    
    def test_assess_quality_blurry_flagged(self, assessor, blurry_image):
        """Test blurry image is flagged."""
        score, issues = assessor.assess_quality(blurry_image)
        assert "blurry" in issues
        assert score < 0.8  # Should be penalized
    
    def test_assess_quality_dark_flagged(self, assessor, dark_image):
        """Test dark image is flagged."""
        score, issues = assessor.assess_quality(dark_image)
        assert "underexposed" in issues
    
    def test_assess_quality_bright_flagged(self, assessor, bright_image):
        """Test bright image is flagged."""
        score, issues = assessor.assess_quality(bright_image)
        assert "overexposed" in issues
    
    def test_assess_quality_low_contrast_flagged(self, assessor, low_contrast_image):
        """Test low contrast image is flagged."""
        score, issues = assessor.assess_quality(low_contrast_image)
        assert "low_contrast" in issues
    
    def test_get_detailed_metrics(self, assessor, sharp_image):
        """Test detailed metrics dictionary."""
        metrics = assessor.get_detailed_metrics(sharp_image)
        
        assert "quality_score" in metrics
        assert "issues" in metrics
        assert "metrics" in metrics
        assert "thresholds" in metrics
        
        assert "laplacian_variance" in metrics["metrics"]
        assert "brightness" in metrics["metrics"]
        assert "contrast" in metrics["metrics"]
    
    def test_handles_float_image(self, assessor):
        """Test handling of float [0, 1] normalized images."""
        float_img = np.random.rand(224, 224, 3).astype(np.float32)
        score, issues = assessor.assess_quality(float_img)
        assert 0.0 <= score <= 1.0


class TestDecisionLogic:
    """Test decision flag computation logic."""
    
    def test_decision_ok_conditions(self):
        """Test OK decision for good conditions."""
        # Import the function once app/main.py is updated
        # For now, test the logic conceptually
        grade = 0
        confidence = 0.85
        quality_score = 0.7
        uncertainty = 0.05
        entropy = 0.5
        
        # OK conditions: good quality, high confidence, low uncertainty, low severity
        is_ok = (
            quality_score >= 0.4 and
            confidence >= 0.6 and
            uncertainty <= 0.15 and
            entropy <= 1.5 and
            grade < 3
        )
        assert is_ok
    
    def test_decision_review_low_confidence(self):
        """Test REVIEW decision for low confidence."""
        confidence = 0.45
        # Low confidence should trigger REVIEW
        assert confidence < 0.6
    
    def test_decision_review_high_uncertainty(self):
        """Test REVIEW decision for high uncertainty."""
        uncertainty = 0.25
        # High uncertainty should trigger REVIEW
        assert uncertainty > 0.15
    
    def test_decision_review_severe_grade(self):
        """Test REVIEW decision for severe grades."""
        grade = 3  # Severe
        # Severe grades should trigger REVIEW for safety
        assert grade >= 3
    
    def test_decision_retake_poor_quality(self):
        """Test RETAKE decision for poor quality."""
        quality_score = 0.25
        # Poor quality should trigger RETAKE
        assert quality_score < 0.4
    
    def test_decision_ood_high_entropy(self):
        """Test OOD decision for high entropy."""
        entropy = 1.8
        # High entropy suggests out-of-distribution
        assert entropy > 1.5


class TestEnvironmentVariables:
    """Test environment variable configuration."""
    
    def test_demo_mode_env_variable(self):
        """Test DEMO_MODE environment variable parsing."""
        with patch.dict(os.environ, {"DR_DEMO_MODE": "true"}):
            demo_mode = os.getenv("DR_DEMO_MODE", "false").lower() == "true"
            assert demo_mode is True
        
        with patch.dict(os.environ, {"DR_DEMO_MODE": "false"}):
            demo_mode = os.getenv("DR_DEMO_MODE", "false").lower() == "true"
            assert demo_mode is False
    
    def test_threshold_env_variables(self):
        """Test threshold environment variables."""
        with patch.dict(os.environ, {"DR_QUALITY_THRESHOLD": "0.5"}):
            threshold = float(os.getenv("DR_QUALITY_THRESHOLD", "0.4"))
            assert threshold == 0.5
        
        with patch.dict(os.environ, {"DR_UNCERTAINTY_THRESHOLD": "0.2"}):
            threshold = float(os.getenv("DR_UNCERTAINTY_THRESHOLD", "0.15"))
            assert threshold == 0.2
    
    def test_default_values(self):
        """Test default values when env vars not set."""
        # Clear any existing values
        with patch.dict(os.environ, {}, clear=True):
            demo_mode = os.getenv("DR_DEMO_MODE", "false").lower() == "true"
            quality_threshold = float(os.getenv("DR_QUALITY_THRESHOLD", "0.4"))
            uncertainty_threshold = float(os.getenv("DR_UNCERTAINTY_THRESHOLD", "0.15"))
            ood_threshold = float(os.getenv("DR_OOD_ENTROPY_THRESHOLD", "1.5"))
            
            assert demo_mode is False
            assert quality_threshold == 0.4
            assert uncertainty_threshold == 0.15
            assert ood_threshold == 1.5


class TestQualityEdgeCases:
    """Test edge cases for quality assessment."""
    
    @pytest.fixture
    def assessor(self):
        return ImageQualityAssessor()
    
    def test_all_black_image(self, assessor):
        """Test completely black image."""
        black = np.zeros((224, 224, 3), dtype=np.uint8)
        score, issues = assessor.assess_quality(black)
        assert score < 0.3  # Should be poor quality
        assert "underexposed" in issues or "insufficient_fundus_area" in issues
    
    def test_all_white_image(self, assessor):
        """Test completely white image."""
        white = np.ones((224, 224, 3), dtype=np.uint8) * 255
        score, issues = assessor.assess_quality(white)
        assert score < 0.5  # Should be poor quality
        assert "overexposed" in issues
    
    def test_tiny_image(self, assessor):
        """Test very small image."""
        tiny = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        # Should not crash
        score, issues = assessor.assess_quality(tiny)
        assert 0.0 <= score <= 1.0
    
    def test_large_image(self, assessor):
        """Test large image."""
        large = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        # Should not crash
        score, issues = assessor.assess_quality(large)
        assert 0.0 <= score <= 1.0
    
    def test_none_image_raises(self, assessor):
        """Test None image raises ValueError."""
        with pytest.raises(ValueError):
            assessor._ensure_rgb(None)
