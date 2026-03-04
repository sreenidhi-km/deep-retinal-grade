"""
Integrated Gradients - Pixel-level Attribution

Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks" (ICML 2017)

Author: Deep Retina Grade Project
Date: January 2026
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class IntegratedGradients:
    """
    Integrated Gradients for pixel-level attribution.

    This method computes attribution by integrating gradients
    along a path from a baseline to the input.
    """

    def __init__(self, model: torch.nn.Module):
        """Initialize with model."""
        self.model = model

    def generate(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None, 
        baseline: Optional[torch.Tensor] = None, 
        steps: int = 50
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Integrated Gradients attribution map.

        Args:
            input_tensor: Input image [1, 3, H, W]
            target_class: Target class (if None, uses predicted)
            baseline: Baseline image (if None, uses zeros)
            steps: Number of interpolation steps

        Returns:
            attribution: Attribution map [H, W]
            pred_class: Predicted class
            confidence: Prediction confidence
        """
        self.model.eval()

        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            pred_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, pred_class].item()

        if target_class is None:
            target_class = pred_class

        # Create baseline
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        # Generate interpolated inputs
        scaled_inputs = [
            baseline + (float(i) / steps) * (input_tensor - baseline)
            for i in range(steps + 1)
        ]
        scaled_inputs = torch.cat(scaled_inputs, dim=0)
        scaled_inputs.requires_grad_(True)

        # Compute gradients
        outputs = self.model(scaled_inputs)
        target_outputs = outputs[:, target_class]

        self.model.zero_grad()
        target_outputs.sum().backward()

        gradients = scaled_inputs.grad
        avg_gradients = gradients.mean(dim=0, keepdim=True)

        # Compute attribution
        attribution = (input_tensor - baseline) * avg_gradients
        attribution = attribution.abs().sum(dim=1).squeeze()

        # Normalize
        attribution = attribution.cpu().numpy()
        if attribution.max() > 0:
            attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())

        return attribution, pred_class, confidence