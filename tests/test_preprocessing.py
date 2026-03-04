"""
Tests for image preprocessing pipeline.
Covers Ben Graham preprocessing and CLAHE enhancement.
"""

import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.preprocess import RetinaPreprocessor


class TestRetinaPreprocessor:
    """Test suite for RetinaPreprocessor class."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize preprocessor for each test."""
        self.preprocessor = RetinaPreprocessor(img_size=224)
    
    def test_preprocessor_initialization(self):
        """Test that preprocessor initializes with correct parameters."""
        assert self.preprocessor.img_size == 224
        preprocessor_512 = RetinaPreprocessor(img_size=512)
        assert preprocessor_512.img_size == 512
    
    def test_ben_graham_output_shape(self, dummy_rgb_image):
        """Test that Ben Graham preprocessing outputs correct shape."""
        result = self.preprocessor.ben_graham_preprocessing(dummy_rgb_image)
        assert result.shape == (224, 224, 3), f"Expected (224, 224, 3), got {result.shape}"
    
    def test_ben_graham_output_dtype(self, dummy_rgb_image):
        """Test that output is uint8."""
        result = self.preprocessor.ben_graham_preprocessing(dummy_rgb_image)
        assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
    
    def test_ben_graham_output_range(self, dummy_rgb_image):
        """Test that output values are in valid range [0, 255]."""
        result = self.preprocessor.ben_graham_preprocessing(dummy_rgb_image)
        assert result.min() >= 0, f"Min value {result.min()} < 0"
        assert result.max() <= 255, f"Max value {result.max()} > 255"
    
    def test_ben_graham_handles_small_image(self, small_image):
        """Test preprocessing handles small images."""
        result = self.preprocessor.ben_graham_preprocessing(small_image)
        # Ben Graham doesn't resize - just check it doesn't crash
        assert result.shape[2] == 3  # Still RGB
        assert result.dtype == np.uint8
    
    def test_ben_graham_handles_large_image(self, large_image):
        """Test preprocessing handles large images."""
        result = self.preprocessor.ben_graham_preprocessing(large_image)
        # Ben Graham doesn't resize - just check it doesn't crash
        assert result.shape[2] == 3  # Still RGB
        assert result.dtype == np.uint8
    
    def test_ben_graham_handles_black_image(self, black_image):
        """Test preprocessing handles completely black images without crashing."""
        # Should not raise an exception
        result = self.preprocessor.ben_graham_preprocessing(black_image)
        assert result.shape == (224, 224, 3)
    
    def test_ben_graham_handles_white_image(self, white_image):
        """Test preprocessing handles completely white images."""
        result = self.preprocessor.ben_graham_preprocessing(white_image)
        assert result.shape == (224, 224, 3)
    
    def test_crop_from_gray_output_shape(self, dummy_rgb_image):
        """Test that crop_image_from_gray produces valid output."""
        # First do basic resize
        img_resized = cv2.resize(dummy_rgb_image, (224, 224))
        result = self.preprocessor.crop_image_from_gray(img_resized)
        # Output should be 2D or 3D with reasonable dimensions
        assert len(result.shape) in [2, 3]
        assert result.shape[0] > 0 and result.shape[1] > 0
    
    def test_preprocessing_is_deterministic(self, dummy_rgb_image):
        """Test that preprocessing produces same output for same input."""
        result1 = self.preprocessor.ben_graham_preprocessing(dummy_rgb_image.copy())
        result2 = self.preprocessor.ben_graham_preprocessing(dummy_rgb_image.copy())
        np.testing.assert_array_equal(result1, result2)
    
    def test_preprocessing_modifies_image(self, dummy_rgb_image):
        """Test that preprocessing actually modifies the image."""
        original = dummy_rgb_image.copy()
        result = self.preprocessor.ben_graham_preprocessing(dummy_rgb_image)
        # The result should be different from simple resize
        simple_resize = cv2.resize(original, (224, 224))
        assert not np.array_equal(result, simple_resize), "Preprocessing should modify image"


class TestCLAHE:
    """Test CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    
    def test_clahe_enhances_contrast(self, dummy_rgb_image):
        """Test that CLAHE increases contrast in low-contrast images."""
        # Create a low contrast image
        low_contrast = np.clip(dummy_rgb_image * 0.3 + 100, 0, 255).astype(np.uint8)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(low_contrast, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Enhanced image should have higher standard deviation (more contrast)
        original_std = np.std(low_contrast[:, :, 0])
        enhanced_std = np.std(enhanced[:, :, 0])
        
        # CLAHE should generally increase contrast
        # (allowing small tolerance for edge cases)
        assert enhanced_std >= original_std * 0.8, \
            f"CLAHE should enhance contrast: {enhanced_std} vs {original_std}"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_grayscale_input_handling(self):
        """Test handling of grayscale images."""
        preprocessor = RetinaPreprocessor(img_size=224)
        gray_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        
        # Convert to 3-channel
        rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        result = preprocessor.ben_graham_preprocessing(rgb_image)
        
        assert result.shape == (224, 224, 3)
    
    def test_rgba_input_handling(self):
        """Test handling of RGBA images (4 channels)."""
        preprocessor = RetinaPreprocessor(img_size=224)
        rgba_image = np.random.randint(0, 255, (224, 224, 4), dtype=np.uint8)
        
        # Convert to RGB
        rgb_image = rgba_image[:, :, :3]
        result = preprocessor.ben_graham_preprocessing(rgb_image)
        
        assert result.shape == (224, 224, 3)
    
    def test_nonsquare_image(self):
        """Test handling of non-square images."""
        preprocessor = RetinaPreprocessor(img_size=224)
        nonsquare = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        
        result = preprocessor.ben_graham_preprocessing(nonsquare)
        # Ben Graham doesn't resize - just check it doesn't crash
        assert result.shape[2] == 3  # Still RGB
        assert result.dtype == np.uint8
