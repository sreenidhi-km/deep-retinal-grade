"""
CORAL Loss - Consistent Rank Logits for Ordinal Regression

CORAL improves upon standard ordinal regression by enforcing consistent 
rank predictions through shared weights with rank-specific bias terms.

Key advantage for DR grading:
- Standard cross-entropy treats Grade 2→4 misclassification the same as 2→1
- CORAL penalizes distant misclassifications more heavily
- Reduces adjacent-class confusion (Grade 2↔3 problem)

Reference: Cao et al., "Rank Consistent Ordinal Regression for Neural Networks 
with Application to Age Estimation" (Pattern Recognition Letters, 2020)

Author: Deep Retina Grade Project
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CORALLoss(nn.Module):
    """
    CORAL (Consistent Rank Logits) Loss for Ordinal Regression.
    
    Converts K-class ordinal classification into K-1 binary tasks:
    - Task k: P(Y > k) for k = 0, 1, ..., K-2
    
    Uses binary cross-entropy on each task with importance weighting
    for rare classes.
    
    Args:
        num_classes: Number of ordinal classes (5 for DR grades)
        importance_weights: Optional weights per rank threshold
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        importance_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_classes - 1
        
        if importance_weights is not None:
            if len(importance_weights) != self.num_tasks:
                raise ValueError(
                    f"Expected {self.num_tasks} importance_weights, got {len(importance_weights)}"
                )
            self.register_buffer('importance_weights', importance_weights)
        else:
            # Default: upweight the Grade 2→3 boundary (where most confusion occurs)
            default_weights = torch.ones(self.num_tasks)
            if self.num_tasks > 2:  # P(Y > 2) - the Grade 2 vs 3 boundary
                default_weights[2] = 2.0
            if self.num_tasks > 3:  # P(Y > 3) - the Grade 3 vs 4 boundary
                default_weights[3] = 1.5
            self.register_buffer('importance_weights', default_weights)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute CORAL loss.
        
        Args:
            logits: Model output of shape [B, num_tasks] (K-1 binary logits)
            targets: Ground truth labels of shape [B] (values 0 to K-1)
            
        Returns:
            Weighted binary cross-entropy loss
        """
        batch_size = logits.size(0)
        device = logits.device
        
        # Create ordinal labels: for target=2, labels = [1, 1, 0, 0]
        # (Y > 0? Yes. Y > 1? Yes. Y > 2? No. Y > 3? No.)
        ordinal_labels = torch.zeros(batch_size, self.num_tasks, device=device)
        for k in range(self.num_tasks):
            ordinal_labels[:, k] = (targets > k).float()
        
        # Binary cross-entropy for each rank threshold
        bce = F.binary_cross_entropy_with_logits(
            logits, ordinal_labels, reduction='none'
        )
        
        # Apply importance weights
        weighted_bce = bce * self.importance_weights.unsqueeze(0)
        
        return weighted_bce.mean()
    
    @staticmethod
    def logits_to_grade(logits: torch.Tensor) -> torch.Tensor:
        """
        Convert CORAL logits to predicted grade.
        
        Args:
            logits: CORAL logits [B, K-1]
            
        Returns:
            Predicted grades [B] (values 0 to K-1)
        """
        # Apply sigmoid to get P(Y > k)
        probs = torch.sigmoid(logits)
        # Grade = number of thresholds exceeded
        predicted = (probs > 0.5).sum(dim=1)
        return predicted
    
    @staticmethod
    def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
        """
        Convert CORAL logits to class probabilities.
        
        P(Y = k) = P(Y > k-1) - P(Y > k)
        
        Args:
            logits: CORAL logits [B, K-1]
            
        Returns:
            Class probabilities [B, K]
        """
        cumprobs = torch.sigmoid(logits)  # [B, K-1]
        
        # Add boundaries: P(Y > -1) = 1 and P(Y > K-1) = 0
        ones = torch.ones(cumprobs.size(0), 1, device=cumprobs.device)
        zeros = torch.zeros(cumprobs.size(0), 1, device=cumprobs.device)
        
        extended = torch.cat([ones, cumprobs, zeros], dim=1)  # [B, K+1]
        
        # P(Y = k) = P(Y > k-1) - P(Y > k)
        probs = extended[:, :-1] - extended[:, 1:]
        
        # Clamp to avoid numerical issues
        probs = probs.clamp(min=0)
        
        # Renormalize
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        
        return probs


class CORALHead(nn.Module):
    """
    CORAL classification head replacing standard linear head.
    
    Uses shared weights with rank-specific biases for consistent
    ordinal predictions.
    
    Args:
        in_features: Input feature dimension
        num_classes: Number of ordinal classes (5 for DR)
    """
    
    def __init__(self, in_features: int, num_classes: int = 5):
        super().__init__()
        self.num_tasks = num_classes - 1
        
        # Shared weights for all rank thresholds
        self.fc = nn.Linear(in_features, 1, bias=False)
        
        # Rank-specific biases (learned independently)
        self.biases = nn.Parameter(torch.zeros(self.num_tasks))
        
        # Initialize biases in decreasing order (prior: lower grades more common)
        # Linearly spaced from +1.0 (P(Y > 0), most samples) to
        # -0.5 (P(Y > K-2), fewest samples)
        with torch.no_grad():
            self.biases.copy_(
                torch.linspace(1.0, -0.5, self.num_tasks)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing K-1 rank logits.
        
        Args:
            x: Input features [B, in_features]
            
        Returns:
            CORAL logits [B, K-1]
        """
        # Shared representation
        shared = self.fc(x)  # [B, 1]
        
        # Add rank-specific biases
        logits = shared + self.biases.unsqueeze(0)  # [B, K-1]
        
        return logits


if __name__ == "__main__":
    """Quick test for CORAL loss."""
    print("Testing CORAL Loss...")
    
    batch_size = 8
    num_classes = 5
    num_tasks = num_classes - 1
    
    # Simulate CORAL logits (K-1 outputs instead of K)
    logits = torch.randn(batch_size, num_tasks)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test CORAL loss
    coral = CORALLoss(num_classes=5)
    loss = coral(logits, targets)
    print(f"✅ CORAL Loss: {loss.item():.4f}")
    
    # Test grade prediction
    grades = CORALLoss.logits_to_grade(logits)
    print(f"✅ Predicted grades: {grades.tolist()}")
    
    # Test probability conversion
    probs = CORALLoss.logits_to_probs(logits)
    print(f"✅ Probabilities sum: {probs.sum(dim=1).tolist()}")
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=0.01)
    
    # Test CORAL head
    head = CORALHead(in_features=1280, num_classes=5)
    features = torch.randn(batch_size, 1280)
    coral_logits = head(features)
    print(f"✅ CORAL head output shape: {coral_logits.shape}")
    assert coral_logits.shape == (batch_size, num_tasks)
    
    print("\n✅ All CORAL tests passed!")
