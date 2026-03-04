"""
GradCAM - Gradient-weighted Class Activation Mapping

Visual explanations for CNN-based predictions.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (ICCV 2017)

Author: Deep Retina Grade Project
Date: January 2026
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple


class GradCAM:
    """
    GradCAM implementation for visual explanations.

    This class computes gradient-weighted class activation maps,
    highlighting regions important for the model's prediction.

    Usage:
        gradcam = GradCAM(model, target_layer)
        heatmap, pred_class, confidence = gradcam.generate(input_tensor)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Initialize GradCAM.

        Args:
            model: The neural network model
            target_layer: The convolutional layer to compute GradCAM for
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """Store activations during forward pass."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Store gradients during backward pass."""
        self.gradients = grad_output[0].detach()

    def generate(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate GradCAM heatmap.

        Args:
            input_tensor: Input image tensor [1, 3, H, W]
            target_class: Target class index (if None, uses predicted class)

        Returns:
            heatmap: GradCAM heatmap [H, W] in range [0, 1]
            pred_class: Predicted class
            confidence: Prediction confidence
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = F.softmax(output, dim=1)[0, pred_class].item()

        # Use predicted class if target not specified
        if target_class is None:
            target_class = pred_class

        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()

        # Get weights (global average pooling of gradients)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Resize to input size
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, pred_class, confidence


def overlay_heatmap(
    image: np.ndarray, 
    heatmap: np.ndarray, 
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on image.

    Args:
        image: Original image [H, W, 3] in range [0, 1]
        heatmap: Heatmap [H, W] in range [0, 1]
        alpha: Blending factor (0 = only image, 1 = only heatmap)
        colormap: OpenCV colormap

    Returns:
        Blended image [H, W, 3] in range [0, 1]
    """
    # Convert heatmap to color
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), 
        colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

    # Blend
    overlaid = alpha * heatmap_colored + (1 - alpha) * image
    overlaid = np.clip(overlaid, 0, 1)

    return overlaid