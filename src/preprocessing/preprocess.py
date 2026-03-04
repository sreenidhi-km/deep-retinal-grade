"""
RetinaPreprocessor - Age-Invariant Fundus Image Preprocessing

This module implements Ben Graham's preprocessing pipeline for fundus images,
which won the Kaggle Diabetic Retinopathy Detection competition in 2015.

The preprocessing addresses:
- Age variance (yellowing lenses in elderly patients)
- Ethnicity variance (different fundus pigmentation levels)
- Camera/lighting differences between clinics

Author: Deep Retina Grade Project
Date: January 2026
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Dict, Optional


class RetinaPreprocessor:
    """
    Preprocessing pipeline for fundus images using Ben Graham's method + CLAHE.

    This addresses:
    - Age variance (yellowing lenses in elderly)
    - Ethnicity variance (different pigmentation levels)
    - Camera/lighting differences between clinics

    Reference: Ben Graham's winning solution for Kaggle DR Detection (2015)
    """

    def __init__(
        self, 
        img_size: int = 224, 
        ben_graham_sigma: float = 10, 
        clahe_clip: float = 2.0, 
        clahe_grid: tuple = (8, 8)
    ):
        """
        Initialize preprocessor with configurable parameters.

        Args:
            img_size: Target image size (default 224 for EfficientNet)
            ben_graham_sigma: Gaussian blur sigma for Ben Graham method
            clahe_clip: CLAHE clip limit (higher = more contrast enhancement)
            clahe_grid: CLAHE grid size (smaller = more local enhancement)
        """
        self.img_size = img_size
        self.ben_graham_sigma = ben_graham_sigma
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid

        # Initialize CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)

    def crop_image_from_gray(self, img: np.ndarray, tol: int = 7) -> np.ndarray:
        """
        Crop black borders from fundus image.

        Args:
            img: Input image (BGR or RGB)
            tol: Tolerance for black detection

        Returns:
            Cropped image containing only the fundus region
        """
        if img.ndim == 2:
            mask = img > tol
            return img[np.ix_(mask.any(1), mask.any(0))]
        elif img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] == 3 else img[:,:,0]
            mask = gray > tol

            if not mask.any():
                return img

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            return img[rmin:rmax+1, cmin:cmax+1]

        return img

    def ben_graham_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """
        Apply Ben Graham's Gaussian Difference method.

        Formula: output = img - gaussian_blur(img) + 128

        Args:
            img: Input image (RGB, uint8)

        Returns:
            Processed image with normalized colors
        """
        sigma = self.ben_graham_sigma * (img.shape[0] / 224.0)
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        result = cv2.addWeighted(img, 4, blurred, -4, 128)
        return result

    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            img: Input image (RGB, uint8)

        Returns:
            Contrast-enhanced image
        """
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return result

    def resize_with_aspect_ratio(self, img: np.ndarray) -> np.ndarray:
        """
        Resize image to target size while maintaining aspect ratio.

        Args:
            img: Input image

        Returns:
            Resized image of shape (img_size, img_size, 3)
        """
        h, w = img.shape[:2]
        scale = self.img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pad_h = (self.img_size - new_h) // 2
        pad_w = (self.img_size - new_w) // 2

        result = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        result[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

        return result

    def preprocess(
        self, 
        image_path: Union[str, Path], 
        apply_ben_graham: bool = True, 
        apply_clahe: bool = True,
        return_tensor: bool = False
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline.

        Steps:
        1. Load image
        2. Crop black borders
        3. Resize to target size
        4. Apply Ben Graham's method (optional)
        5. Apply CLAHE (optional)
        6. Normalize to [0, 1]

        Args:
            image_path: Path to input image
            apply_ben_graham: Whether to apply Ben Graham preprocessing
            apply_clahe: Whether to apply CLAHE
            return_tensor: If True, return in CHW format for PyTorch

        Returns:
            Preprocessed image as numpy array [0, 1] float32
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.crop_image_from_gray(img)
        img = self.resize_with_aspect_ratio(img)

        if apply_ben_graham:
            img = self.ben_graham_preprocessing(img)

        if apply_clahe:
            img = self.apply_clahe(img)

        img = img.astype(np.float32) / 255.0

        if return_tensor:
            # Convert HWC to CHW for PyTorch
            img = np.transpose(img, (2, 0, 1))

        return img

    def preprocess_array(
        self, 
        img: np.ndarray, 
        apply_ben_graham: bool = True, 
        apply_clahe: bool = True,
        return_tensor: bool = False
    ) -> np.ndarray:
        """
        Preprocess a numpy array image (for API use).

        Steps:
        1. Crop black borders
        2. Resize to target size
        3. Apply Ben Graham's method (optional)
        4. Apply CLAHE (optional)
        5. Normalize to [0, 1]

        Args:
            img: Input image as numpy array (RGB, uint8)
            apply_ben_graham: Whether to apply Ben Graham preprocessing
            apply_clahe: Whether to apply CLAHE
            return_tensor: If True, return in CHW format for PyTorch

        Returns:
            Preprocessed image as numpy array [0, 1] float32
        """
        if img is None:
            raise ValueError("Input image is None")

        # Ensure RGB format
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        img = self.crop_image_from_gray(img)
        img = self.resize_with_aspect_ratio(img)

        if apply_ben_graham:
            img = self.ben_graham_preprocessing(img)

        if apply_clahe:
            img = self.apply_clahe(img)

        img = img.astype(np.float32) / 255.0

        if return_tensor:
            # Convert HWC to CHW for PyTorch
            img = np.transpose(img, (2, 0, 1))

        return img

    def preprocess_for_visualization(self, image_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Generate intermediate stages for visualization.

        Returns:
            Dictionary with all preprocessing stages (uint8 for display)
        """
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        stages = {}
        stages['original'] = img.copy()

        img_cropped = self.crop_image_from_gray(img)
        stages['cropped'] = img_cropped.copy()

        img_resized = self.resize_with_aspect_ratio(img_cropped)
        stages['resized'] = img_resized.copy()

        img_graham = self.ben_graham_preprocessing(img_resized)
        stages['ben_graham'] = img_graham.copy()

        img_clahe = self.apply_clahe(img_graham)
        stages['ben_graham_clahe'] = img_clahe.copy()

        return stages


# Convenience function for quick preprocessing
def preprocess_fundus_image(
    image_path: Union[str, Path],
    img_size: int = 224,
    apply_ben_graham: bool = True,
    apply_clahe: bool = True
) -> np.ndarray:
    """
    Convenience function to preprocess a single fundus image.

    Args:
        image_path: Path to the fundus image
        img_size: Target size (default 224)
        apply_ben_graham: Whether to apply Ben Graham preprocessing
        apply_clahe: Whether to apply CLAHE

    Returns:
        Preprocessed image as numpy array [0, 1] float32
    """
    preprocessor = RetinaPreprocessor(img_size=img_size)
    return preprocessor.preprocess(image_path, apply_ben_graham, apply_clahe)