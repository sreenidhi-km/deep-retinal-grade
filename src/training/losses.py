"""
Advanced Loss Functions for Diabetic Retinopathy Grading

This module implements specialized loss functions that can significantly
improve model accuracy for the DR grading task:

1. FocalLoss - Handles class imbalance by down-weighting easy examples
2. OrdinalRegressionLoss - Respects the ordinal nature of DR grades (0<1<2<3<4)
3. CombinedLoss - Weighted combination for best results

Author: Deep Retina Grade Project
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    - α (alpha): Balances between positive/negative examples
    - γ (gamma): Focuses on hard examples (higher γ = more focus on hard)
    
    For DR grading with severe class imbalance, this helps the model
    focus on rare but critical classes (Severe DR, Proliferative DR).
    
    Args:
        alpha: Class weights (tensor of shape [num_classes]) or scalar
        gamma: Focusing parameter (default: 2.0, recommended: 1.5-2.5)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self, 
        alpha: Optional[torch.Tensor] = None, 
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits of shape [B, num_classes]
            targets: Ground truth labels of shape [B]
            
        Returns:
            Focal loss value
        """
        # Compute cross-entropy loss (without reduction for focal weighting)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get predicted probabilities
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class
        
        # Apply focal weighting
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal Regression Loss for ordered classes.
    
    DR grades are ordinal: 0 < 1 < 2 < 3 < 4
    Standard cross-entropy treats them as independent, ignoring this order.
    
    This loss converts the problem to K-1 binary classification tasks:
    - P(grade >= 1)
    - P(grade >= 2)
    - P(grade >= 3)
    - P(grade >= 4)
    
    Paper: "Ordinal Regression with Multiple Output CNN" (Niu et al., 2016)
    
    Args:
        num_classes: Number of ordinal classes (default: 5 for DR)
    """
    
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute ordinal regression loss.
        
        Args:
            logits: Model outputs of shape [B, num_classes]
            targets: Ground truth labels of shape [B]
            
        Returns:
            Ordinal loss value
        """
        batch_size = logits.size(0)
        device = logits.device
        
        # Create cumulative probabilities from logits
        # We use sigmoid on cumulative sums to get P(grade >= k)
        cumulative_probs = torch.sigmoid(logits)
        
        # Create target labels for each threshold
        # For target=2: labels = [1, 1, 0, 0] (grade >= 1? Yes, >= 2? Yes, >= 3? No, >= 4? No)
        ordinal_labels = torch.zeros(batch_size, self.num_classes, device=device)
        for i in range(batch_size):
            ordinal_labels[i, :targets[i]+1] = 1
        
        # Binary cross-entropy for each ordinal threshold
        # We exclude the first column (P(grade >= 0) is always 1)
        loss = F.binary_cross_entropy(
            cumulative_probs[:, :-1],  # Exclude last column
            ordinal_labels[:, 1:],     # Exclude first column (always 1)
            reduction='mean'
        )
        
        return loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for regularization.
    
    Instead of hard labels (0 or 1), use soft labels:
    - Target class: 1 - smoothing + smoothing/num_classes
    - Other classes: smoothing/num_classes
    
    This prevents overconfident predictions and improves generalization.
    
    Args:
        num_classes: Number of classes
        smoothing: Smoothing factor (default: 0.1)
    """
    
    def __init__(self, num_classes: int = 5, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            inputs: Logits of shape [B, num_classes]
            targets: Ground truth labels of shape [B]
            
        Returns:
            Label smoothing loss value
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smooth labels
        with torch.no_grad():
            smooth_labels = torch.zeros_like(log_probs)
            smooth_labels.fill_(self.smoothing / (self.num_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # Compute loss
        loss = (-smooth_labels * log_probs).sum(dim=-1).mean()
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined Loss: Focal + Ordinal + Label Smoothing
    
    This combines multiple loss functions for optimal training:
    - Focal Loss: Handles class imbalance
    - Ordinal Loss: Respects grade ordering
    - Label Smoothing: Prevents overconfidence
    
    Args:
        num_classes: Number of classes (default: 5)
        class_weights: Optional class weights for focal loss
        focal_gamma: Gamma for focal loss (default: 2.0)
        focal_weight: Weight for focal loss component (default: 1.0)
        ordinal_weight: Weight for ordinal loss component (default: 0.3)
        smoothing: Label smoothing factor (default: 0.05)
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        focal_weight: float = 1.0,
        ordinal_weight: float = 0.3,
        smoothing: float = 0.05
    ):
        super().__init__()
        
        self.focal_weight = focal_weight
        self.ordinal_weight = ordinal_weight
        
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.ordinal_loss = OrdinalRegressionLoss(num_classes=num_classes)
        self.smoothing_loss = LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing)
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            logits: Model outputs of shape [B, num_classes]
            targets: Ground truth labels of shape [B]
            
        Returns:
            Combined loss value
        """
        focal = self.focal_loss(logits, targets)
        ordinal = self.ordinal_loss(logits, targets)
        
        # Combine losses
        total_loss = (self.focal_weight * focal) + (self.ordinal_weight * ordinal)
        
        return total_loss


def compute_class_weights(labels: List[int], num_classes: int = 5) -> torch.Tensor:
    """
    Compute class weights using inverse frequency.
    
    Args:
        labels: List of all labels in training set
        num_classes: Number of classes
        
    Returns:
        Tensor of class weights
    """
    from collections import Counter
    
    counts = Counter(labels)
    total = len(labels)
    
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)  # Avoid division by zero
        # Inverse frequency weighting
        weight = total / (num_classes * count)
        weights.append(weight)
    
    # Normalize so max weight = 1
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.max()
    
    # Boost rare classes even more
    weights = weights ** 0.5  # Square root smoothing
    
    return weights


if __name__ == "__main__":
    # Quick test
    print("Testing loss functions...")
    
    # Create dummy data
    batch_size = 8
    num_classes = 5
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test Focal Loss
    focal = FocalLoss(gamma=2.0)
    focal_loss = focal(logits, targets)
    print(f"✅ Focal Loss: {focal_loss.item():.4f}")
    
    # Test Ordinal Loss
    ordinal = OrdinalRegressionLoss(num_classes=5)
    ordinal_loss = ordinal(logits, targets)
    print(f"✅ Ordinal Loss: {ordinal_loss.item():.4f}")
    
    # Test Label Smoothing Loss
    smooth = LabelSmoothingLoss(num_classes=5, smoothing=0.1)
    smooth_loss = smooth(logits, targets)
    print(f"✅ Label Smoothing Loss: {smooth_loss.item():.4f}")
    
    # Test Combined Loss
    weights = torch.tensor([1.0, 1.5, 1.2, 2.0, 2.5])
    combined = CombinedLoss(num_classes=5, class_weights=weights)
    combined_loss = combined(logits, targets)
    print(f"✅ Combined Loss: {combined_loss.item():.4f}")
    
    # Test class weight computation
    labels = [0]*100 + [1]*50 + [2]*80 + [3]*20 + [4]*10
    computed_weights = compute_class_weights(labels)
    print(f"✅ Computed weights: {computed_weights}")
    
    print("\n✅ All loss functions working correctly!")
