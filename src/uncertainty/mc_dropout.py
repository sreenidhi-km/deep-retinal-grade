"""
Monte Carlo Dropout - Uncertainty Quantification

Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016)

This module enables uncertainty estimation by running inference multiple times
with dropout enabled, then analyzing the variance in predictions.

Author: Deep Retina Grade Project
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


def enable_dropout(model: nn.Module) -> None:
    """
    Enable dropout layers during inference for MC Dropout.
    
    Args:
        model: PyTorch model with dropout layers
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def disable_dropout(model: nn.Module) -> None:
    """
    Disable dropout layers (return to eval mode).
    
    Args:
        model: PyTorch model
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()


class MCDropoutPredictor:
    """
    Monte Carlo Dropout for Uncertainty Quantification.
    
    Runs multiple forward passes with dropout enabled to estimate
    epistemic (model) uncertainty.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        n_samples: int = 20,
        uncertainty_threshold: float = 0.15
    ):
        """
        Initialize MC Dropout predictor.
        
        Args:
            model: PyTorch model with dropout layers
            n_samples: Number of Monte Carlo samples (forward passes)
            uncertainty_threshold: Flag predictions above this uncertainty
        """
        self.model = model
        self.n_samples = n_samples
        self.uncertainty_threshold = uncertainty_threshold
        self.device = next(model.parameters()).device
    
    def predict_with_uncertainty(
        self, 
        input_tensor: torch.Tensor
    ) -> Dict[str, any]:
        """
        Make prediction with uncertainty estimation.
        
        Args:
            input_tensor: Input image tensor [1, 3, H, W]
            
        Returns:
            Dictionary containing:
                - predicted_grade: Most likely grade (0-4)
                - confidence: Mean confidence in prediction
                - uncertainty: Standard deviation of predictions (epistemic)
                - entropy: Predictive entropy (total uncertainty)
                - is_borderline: Whether uncertainty exceeds threshold
                - all_predictions: All MC samples for analysis
                - grade_distribution: Count of each grade across samples
        """
        self.model.eval()
        
        # First, get deterministic prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            deterministic_probs = F.softmax(output, dim=1)[0].cpu().numpy()
            deterministic_grade = output.argmax(dim=1).item()
        
        # Enable dropout for MC sampling
        enable_dropout(self.model)
        
        all_probs = []
        all_grades = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                output = self.model(input_tensor)
                probs = F.softmax(output, dim=1)[0].cpu().numpy()
                grade = output.argmax(dim=1).item()
                
                all_probs.append(probs)
                all_grades.append(grade)
        
        # Disable dropout (return to normal eval mode)
        disable_dropout(self.model)
        
        # Convert to numpy arrays
        all_probs = np.array(all_probs)  # [n_samples, num_classes]
        all_grades = np.array(all_grades)  # [n_samples]
        
        # Calculate statistics
        mean_probs = all_probs.mean(axis=0)  # Mean probability per class
        std_probs = all_probs.std(axis=0)    # Std per class
        
        # Predicted grade (mode of samples)
        unique, counts = np.unique(all_grades, return_counts=True)
        predicted_grade = unique[counts.argmax()]
        
        # Confidence in predicted grade
        confidence = mean_probs[predicted_grade]
        
        # Epistemic uncertainty (std of predicted class probability)
        uncertainty = std_probs[predicted_grade]
        
        # Predictive entropy (total uncertainty)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
        
        # Grade distribution across samples
        grade_distribution = {int(g): int((all_grades == g).sum()) for g in range(5)}
        
        # Agreement score (how many samples agree with final prediction)
        agreement = (all_grades == predicted_grade).mean()
        
        # Flag borderline cases
        is_borderline = (
            uncertainty > self.uncertainty_threshold or
            agreement < 0.7 or  # Less than 70% agreement
            confidence < 0.5    # Low confidence
        )
        
        return {
            'predicted_grade': int(predicted_grade),
            'deterministic_grade': int(deterministic_grade),
            'confidence': float(confidence),
            'uncertainty': float(uncertainty),
            'entropy': float(entropy),
            'agreement': float(agreement),
            'is_borderline': bool(is_borderline),
            'grade_distribution': grade_distribution,
            'mean_probabilities': mean_probs.tolist(),
            'std_probabilities': std_probs.tolist(),
            'recommendation': self._get_uncertainty_recommendation(
                predicted_grade, uncertainty, agreement, confidence
            )
        }
    
    def _get_uncertainty_recommendation(
        self, 
        grade: int, 
        uncertainty: float, 
        agreement: float,
        confidence: float
    ) -> str:
        """Generate recommendation based on uncertainty analysis."""
        
        if uncertainty > self.uncertainty_threshold or agreement < 0.7:
            if grade >= 2:
                return (
                    "⚠️ HIGH UNCERTAINTY: Model shows significant disagreement. "
                    "Given potential severity, recommend manual review by ophthalmologist."
                )
            else:
                return (
                    "⚠️ BORDERLINE CASE: Prediction uncertainty is elevated. "
                    "Consider additional imaging or follow-up screening."
                )
        
        if confidence < 0.5:
            return (
                "⚠️ LOW CONFIDENCE: Model is not highly confident in this prediction. "
                "Clinical correlation recommended."
            )
        
        if grade == 0 and confidence > 0.8 and agreement > 0.9:
            return "✅ HIGH CONFIDENCE: Strong agreement across model samples. No DR detected."
        
        if grade >= 3:
            return (
                f"🔴 SEVERE FINDING (Grade {grade}): High confidence detection. "
                "Urgent referral recommended."
            )
        
        return "✅ Prediction is stable across model samples."


def analyze_uncertainty_distribution(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    n_samples: int = 20,
    device: torch.device = None
) -> Dict[str, any]:
    """
    Analyze uncertainty distribution across a dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with test images
        n_samples: MC samples per image
        device: Torch device
        
    Returns:
        Dictionary with uncertainty analysis results
    """
    if device is None:
        device = next(model.parameters()).device
    
    mc_predictor = MCDropoutPredictor(model, n_samples=n_samples)
    
    results = {
        'uncertainties': [],
        'agreements': [],
        'entropies': [],
        'borderline_count': 0,
        'total_count': 0,
        'per_grade_uncertainty': {i: [] for i in range(5)}
    }
    
    model.eval()
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        
        for i in range(images.size(0)):
            img = images[i:i+1]
            label = labels[i].item()
            
            pred = mc_predictor.predict_with_uncertainty(img)
            
            results['uncertainties'].append(pred['uncertainty'])
            results['agreements'].append(pred['agreement'])
            results['entropies'].append(pred['entropy'])
            results['per_grade_uncertainty'][label].append(pred['uncertainty'])
            
            if pred['is_borderline']:
                results['borderline_count'] += 1
            results['total_count'] += 1
    
    # Calculate summary statistics
    results['mean_uncertainty'] = np.mean(results['uncertainties'])
    results['std_uncertainty'] = np.std(results['uncertainties'])
    results['mean_agreement'] = np.mean(results['agreements'])
    results['borderline_rate'] = results['borderline_count'] / results['total_count']
    
    # Per-grade summary
    results['per_grade_mean_uncertainty'] = {
        grade: np.mean(vals) if vals else 0.0 
        for grade, vals in results['per_grade_uncertainty'].items()
    }
    
    return results
