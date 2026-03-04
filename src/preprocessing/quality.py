"""
Image Quality Assessment for Fundus Images

Lightweight quality assessment using:
- Laplacian variance (focus/blur detection)
- Brightness analysis (exposure issues)
- Contrast analysis (low contrast detection)

This is optimized for M1 Mac CPU inference.

Author: Deep Retina Grade Project
Date: January 2026
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any


class ImageQualityAssessor:
    """
    Lightweight image quality assessment for fundus images.
    
    Uses simple, fast metrics that work well on CPU:
    - Laplacian variance for blur detection
    - Mean intensity for brightness
    - Standard deviation for contrast
    
    Usage:
        assessor = ImageQualityAssessor()
        score, issues = assessor.assess_quality(image)
    """
    
    def __init__(
        self,
        laplacian_threshold: float = 100.0,
        brightness_low: float = 30.0,
        brightness_high: float = 220.0,
        contrast_threshold: float = 20.0
    ):
        """
        Initialize quality assessor with configurable thresholds.
        
        Args:
            laplacian_threshold: Min Laplacian variance for sharp image
            brightness_low: Min mean brightness (underexposed below)
            brightness_high: Max mean brightness (overexposed above)
            contrast_threshold: Min std dev for good contrast
        """
        self.laplacian_threshold = laplacian_threshold
        self.brightness_low = brightness_low
        self.brightness_high = brightness_high
        self.contrast_threshold = contrast_threshold
    
    def _ensure_rgb(self, img: np.ndarray) -> np.ndarray:
        """
        Ensure image is in RGB format (H, W, 3).
        
        Handles:
        - Grayscale (H, W) -> RGB
        - Single channel (H, W, 1) -> RGB
        - RGBA (H, W, 4) -> RGB
        - Already RGB -> pass through
        
        Args:
            img: Input image in any format
            
        Returns:
            RGB image (H, W, 3)
        """
        if img is None:
            raise ValueError("Input image is None")
        
        if len(img.shape) == 2:
            # Grayscale (H, W) -> RGB
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3:
            if img.shape[2] == 1:
                # Single channel (H, W, 1) -> RGB
                return cv2.cvtColor(img.squeeze(axis=2), cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                # RGBA -> RGB
                return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif img.shape[2] == 3:
                # Already RGB
                return img
        
        raise ValueError(f"Unsupported image shape: {img.shape}")
    
    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert image to grayscale for analysis."""
        if len(img.shape) == 2:
            return img
        elif len(img.shape) == 3:
            if img.shape[2] == 1:
                return img.squeeze(axis=2)
            else:
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img
    
    def compute_laplacian_variance(self, img: np.ndarray) -> float:
        """
        Compute Laplacian variance as a measure of image sharpness.
        
        Higher values = sharper/more focused image.
        Lower values = blurry image.
        
        Args:
            img: Input image (any format)
            
        Returns:
            Laplacian variance (higher = sharper)
        """
        gray = self._to_grayscale(img)
        
        # Ensure uint8 for Laplacian
        if gray.dtype != np.uint8:
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return float(variance)
    
    def compute_brightness(self, img: np.ndarray) -> float:
        """
        Compute mean brightness of image.
        
        Args:
            img: Input image (any format)
            
        Returns:
            Mean brightness (0-255 scale)
        """
        gray = self._to_grayscale(img)
        
        # Normalize to 0-255 range
        if gray.max() <= 1.0 and gray.dtype in [np.float32, np.float64]:
            gray = gray * 255
        
        return float(np.mean(gray))
    
    def compute_contrast(self, img: np.ndarray) -> float:
        """
        Compute contrast as standard deviation of pixel values.
        
        Args:
            img: Input image (any format)
            
        Returns:
            Contrast (std dev, 0-127 typical range)
        """
        gray = self._to_grayscale(img)
        
        # Normalize to 0-255 range
        if gray.max() <= 1.0 and gray.dtype in [np.float32, np.float64]:
            gray = gray * 255
        
        return float(np.std(gray))
    
    def compute_fundus_coverage(self, img: np.ndarray) -> float:
        """
        Estimate how much of the image contains the fundus (vs black borders).
        
        Args:
            img: Input image
            
        Returns:
            Coverage ratio (0-1, higher = more fundus visible)
        """
        gray = self._to_grayscale(img)
        
        # Normalize to 0-255
        if gray.max() <= 1.0 and gray.dtype in [np.float32, np.float64]:
            gray = (gray * 255).astype(np.uint8)
        elif gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        
        # Count non-black pixels (threshold at 10)
        non_black = np.sum(gray > 10)
        total = gray.size
        
        return float(non_black / total)
    
    def assess_quality(self, img: np.ndarray) -> Tuple[float, List[str]]:
        """
        Assess overall image quality and identify issues.
        
        Args:
            img: Input image (any format - will be converted to RGB)
            
        Returns:
            Tuple of:
                - quality_score: 0.0 (poor) to 1.0 (excellent)
                - issues: List of identified quality issues
        """
        # Ensure RGB format
        img_rgb = self._ensure_rgb(img)
        
        # Compute metrics
        laplacian_var = self.compute_laplacian_variance(img_rgb)
        brightness = self.compute_brightness(img_rgb)
        contrast = self.compute_contrast(img_rgb)
        coverage = self.compute_fundus_coverage(img_rgb)
        
        # Identify issues
        issues = []
        scores = []
        
        # Blur check (Laplacian variance)
        if laplacian_var < self.laplacian_threshold:
            issues.append("blurry")
            blur_score = laplacian_var / self.laplacian_threshold
        else:
            blur_score = min(1.0, laplacian_var / (self.laplacian_threshold * 3))
        scores.append(blur_score)
        
        # Brightness check
        if brightness < self.brightness_low:
            issues.append("underexposed")
            brightness_score = brightness / self.brightness_low
        elif brightness > self.brightness_high:
            issues.append("overexposed")
            brightness_score = (255 - brightness) / (255 - self.brightness_high)
        else:
            # Optimal brightness around 100-150
            optimal = 125
            brightness_score = 1.0 - abs(brightness - optimal) / optimal
        scores.append(max(0, brightness_score))
        
        # Contrast check
        if contrast < self.contrast_threshold:
            issues.append("low_contrast")
            contrast_score = contrast / self.contrast_threshold
        else:
            contrast_score = min(1.0, contrast / (self.contrast_threshold * 3))
        scores.append(contrast_score)
        
        # Coverage check
        if coverage < 0.3:
            issues.append("insufficient_fundus_area")
            coverage_score = coverage / 0.3
        else:
            coverage_score = min(1.0, coverage)
        scores.append(coverage_score)
        
        # Compute overall score (weighted average)
        weights = [0.35, 0.20, 0.25, 0.20]  # blur, brightness, contrast, coverage
        quality_score = sum(s * w for s, w in zip(scores, weights))
        quality_score = max(0.0, min(1.0, quality_score))
        
        return quality_score, issues
    
    def get_detailed_metrics(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Get detailed quality metrics for debugging/logging.
        
        Args:
            img: Input image
            
        Returns:
            Dict with all computed metrics
        """
        img_rgb = self._ensure_rgb(img)
        
        laplacian_var = self.compute_laplacian_variance(img_rgb)
        brightness = self.compute_brightness(img_rgb)
        contrast = self.compute_contrast(img_rgb)
        coverage = self.compute_fundus_coverage(img_rgb)
        quality_score, issues = self.assess_quality(img_rgb)
        
        return {
            "quality_score": quality_score,
            "issues": issues,
            "metrics": {
                "laplacian_variance": laplacian_var,
                "brightness": brightness,
                "contrast": contrast,
                "fundus_coverage": coverage
            },
            "thresholds": {
                "laplacian_threshold": self.laplacian_threshold,
                "brightness_low": self.brightness_low,
                "brightness_high": self.brightness_high,
                "contrast_threshold": self.contrast_threshold
            }
        }
