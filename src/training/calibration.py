"""
Temperature Scaling - Post-hoc Calibration for Confidence Scores

Medical AI systems must have well-calibrated confidence scores.
Temperature scaling is a simple, effective post-hoc calibration method
that learns a single scalar temperature T to rescale logits:

    calibrated_probs = softmax(logits / T)

Reference: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)

Author: Deep Retina Grade Project
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path


class TemperatureScaler(nn.Module):
    """
    Temperature Scaling for post-hoc confidence calibration.
    
    After training the base model, learn a single temperature parameter T 
    on the validation set. During inference, divide logits by T before softmax.
    
    - T > 1: Softens probabilities (reduces overconfidence)
    - T < 1: Sharpens probabilities (increases confidence)
    - T = 1: No change (uncalibrated)
    
    Usage:
        scaler = TemperatureScaler()
        scaler.fit(model, val_loader, device)
        calibrated_probs = scaler.calibrate(logits)
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self._fitted = False
    
    @property
    def is_fitted(self) -> bool:
        return self._fitted
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw model logits [B, num_classes]
            
        Returns:
            Temperature-scaled logits
        """
        return logits / self.temperature.clamp(min=0.01)
    
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Calibrate logits and return probabilities.
        
        Args:
            logits: Raw model logits [B, num_classes]
            
        Returns:
            Calibrated probability distribution [B, num_classes]
        """
        scaled_logits = self.forward(logits)
        return F.softmax(scaled_logits, dim=-1)
    
    def fit(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        max_iter: int = 100,
        lr: float = 0.01
    ) -> Dict[str, float]:
        """
        Learn optimal temperature on validation set.
        
        Uses NLL loss (negative log-likelihood) to find the temperature
        that best calibrates the model's confidence.
        
        Args:
            model: Trained model (kept frozen)
            val_loader: Validation DataLoader
            device: Compute device
            max_iter: Maximum optimization iterations
            lr: Learning rate for temperature optimization
            
        Returns:
            Dictionary with calibration metrics
        """
        was_training = model.training
        model.eval()
        
        try:
            return self._calibrate_inner(model, val_loader, device, max_iter, lr)
        finally:
            if was_training:
                model.train()
    
    def _calibrate_inner(self, model, val_loader, device, max_iter, lr):
        """Internal calibration logic."""
        # Collect all logits and labels from validation set
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    images, labels = batch[0], batch[1]
                else:
                    images, labels = batch['image'], batch['label']
                    
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute pre-calibration metrics (on CPU)
        pre_nll = F.cross_entropy(all_logits, all_labels).item()
        pre_ece = self._compute_ece(all_logits, all_labels)
        
        # Move concatenated tensors to device for optimization
        all_logits = all_logits.to(device)
        all_labels = all_labels.to(device)
        
        # Optimize temperature
        self.to(device)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(all_logits)
            loss = F.cross_entropy(scaled_logits, all_labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        # Compute post-calibration metrics
        with torch.no_grad():
            post_nll = F.cross_entropy(self.forward(all_logits), all_labels).item()
            post_ece = self._compute_ece(self.forward(all_logits), all_labels)
        
        self._fitted = True
        
        return {
            'temperature': self.temperature.item(),
            'pre_calibration_nll': pre_nll,
            'post_calibration_nll': post_nll,
            'pre_calibration_ece': pre_ece,
            'post_calibration_ece': post_ece,
            'nll_improvement': pre_nll - post_nll,
            'ece_improvement': pre_ece - post_ece
        }
    
    def _compute_ece(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 15
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE measures the difference between predicted confidence and actual accuracy
        across multiple confidence bins.
        
        Args:
            logits: Model logits [N, C]
            labels: True labels [N]
            n_bins: Number of confidence bins
            
        Returns:
            ECE value (lower is better, 0 = perfectly calibrated)
        """
        probs = F.softmax(logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)
        accuracies = predictions.eq(labels)
        
        ece = torch.zeros(1, device=logits.device)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].float().mean()
                ece += torch.abs(avg_confidence - avg_accuracy) * prop_in_bin
        
        return ece.item()
    
    def save(self, path: str) -> None:
        """Save temperature parameter to file."""
        torch.save({
            'temperature': self.temperature.data,
            'fitted': self._fitted
        }, path)
    
    @classmethod
    def load(cls, path: str, device: torch.device = torch.device('cpu')) -> 'TemperatureScaler':
        """Load temperature parameter from file."""
        scaler = cls()
        checkpoint = torch.load(path, map_location=device)
        scaler.temperature.data = checkpoint['temperature']
        scaler._fitted = checkpoint.get('fitted', True)
        return scaler


def compute_reliability_diagram(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10
) -> Dict[str, np.ndarray]:
    """
    Compute data for a reliability diagram (calibration plot).
    
    Args:
        logits: Model logits [N, C]
        labels: True labels [N]
        n_bins: Number of bins
        
    Returns:
        Dictionary with bin edges, accuracies, confidences, and counts
    """
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels).float()
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = in_bin.sum().item()
        bin_counts.append(count)
        
        if count > 0:
            bin_accuracies.append(accuracies[in_bin].mean().item())
            bin_confidences.append(confidences[in_bin].mean().item())
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
    
    return {
        'bin_edges': bin_boundaries.numpy(),
        'bin_accuracies': np.array(bin_accuracies),
        'bin_confidences': np.array(bin_confidences),
        'bin_counts': np.array(bin_counts)
    }
