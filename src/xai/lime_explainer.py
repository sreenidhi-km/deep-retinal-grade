"""
LIME - Local Interpretable Model-agnostic Explanations

Reference: Ribeiro et al., "Why Should I Trust You?" (KDD 2016)

Author: Deep Retina Grade Project
Date: January 2026
"""

import torch
import torch.nn.functional as F
import numpy as np
from lime import lime_image
from typing import Optional, Tuple, Callable


class LIMEExplainer:
    """
    LIME for image classification explanations.

    Uses superpixel perturbation to explain predictions.
    """

    def __init__(
        self, 
        model: torch.nn.Module, 
        preprocess_fn: Callable,
        device: torch.device
    ):
        """
        Initialize LIME explainer.

        Args:
            model: PyTorch model
            preprocess_fn: Preprocessing function for images
            device: Torch device
        """
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.device = device
        self.explainer = lime_image.LimeImageExplainer()

    def predict_fn(self, images: np.ndarray) -> np.ndarray:
        """Batch prediction function for LIME."""
        self.model.eval()
        batch_probs = []

        with torch.no_grad():
            for img in images:
                preprocessed = self.preprocess_fn(image=img)['image']
                input_tensor = preprocessed.unsqueeze(0).to(self.device)
                output = self.model(input_tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]
                batch_probs.append(probs)

        return np.array(batch_probs)

    def explain(
        self, 
        image: np.ndarray, 
        target_class: Optional[int] = None, 
        num_samples: int = 500,
        num_features: int = 10
    ) -> Tuple:
        """
        Generate LIME explanation.

        Args:
            image: Input image [H, W, 3] in uint8
            target_class: Class to explain
            num_samples: Number of perturbed samples
            num_features: Number of superpixels to show

        Returns:
            explanation: LIME explanation object
            mask: Binary mask of important regions
            pred_class: Predicted class
            confidence: Prediction confidence
        """
        probs = self.predict_fn(np.array([image]))[0]
        pred_class = probs.argmax()
        confidence = probs[pred_class]

        if target_class is None:
            target_class = pred_class

        explanation = self.explainer.explain_instance(
            image,
            self.predict_fn,
            top_labels=5,
            hide_color=0,
            num_samples=num_samples
        )

        temp, mask = explanation.get_image_and_mask(
            target_class,
            positive_only=True,
            num_features=num_features,
            hide_rest=False
        )

        return explanation, mask, int(pred_class), float(confidence)