"""
RetinaModel - EfficientNet-B0 based model for Diabetic Retinopathy Grading

This module implements the classification model for DR grading using
EfficientNet-B0 as the backbone, pretrained on ImageNet.

Features:
- EfficientNet-B0 backbone (optimized for Apple M1 Mac)
- Custom classification head with dropout
- Support for feature extraction (GradCAM, etc.)
- Configurable for different backbone sizes

Author: Deep Retina Grade Project
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Dict, Any


class RetinaModel(nn.Module):
    """
    EfficientNet-B0 based model for Diabetic Retinopathy grading.
    
    Features:
    - Pretrained EfficientNet-B0 backbone
    - Custom classification head with dropout
    - Supports feature extraction for GradCAM
    
    Args:
        num_classes: Number of output classes (default 5 for DR grades 0-4)
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout rate in classification head
        backbone: Backbone model name (default 'efficientnet_b0')
    """
    
    def __init__(
        self, 
        num_classes: int = 5, 
        pretrained: bool = True, 
        dropout_rate: float = 0.3,
        backbone: str = 'efficientnet_b0'
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Load pretrained backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove original head
            global_pool=''  # We'll handle pooling ourselves
        )
        
        # Get feature dimension
        self.num_features = self.backbone.num_features
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.num_features, num_classes)
        )
        
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before pooling (for GradCAM).
        
        Args:
            x: Input tensor of shape [B, 3, H, W]
            
        Returns:
            Feature maps of shape [B, C, H', W']
        """
        return self.backbone(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [B, 3, H, W]
            
        Returns:
            Logits of shape [B, num_classes]
        """
        features = self.forward_features(x)
        pooled = self.global_pool(features)
        logits = self.head(pooled)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get pooled features (before classification head).
        
        Args:
            x: Input tensor of shape [B, 3, H, W]
            
        Returns:
            Pooled features of shape [B, num_features]
        """
        features = self.forward_features(x)
        pooled = self.global_pool(features)
        return pooled.flatten(1)
    
    def freeze_backbone(self):
        """Freeze backbone parameters (for transfer learning)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class RetinaModelWithUncertainty(RetinaModel):
    """
    RetinaModel with Monte Carlo Dropout for uncertainty estimation.
    
    This model keeps dropout active during inference to estimate
    epistemic uncertainty through multiple forward passes.
    """
    
    def __init__(
        self, 
        num_classes: int = 5, 
        pretrained: bool = True, 
        dropout_rate: float = 0.3,
        backbone: str = 'efficientnet_b0'
    ):
        super().__init__(num_classes, pretrained, dropout_rate, backbone)
    
    def mc_forward(
        self, 
        x: torch.Tensor, 
        n_samples: int = 20
    ) -> Dict[str, torch.Tensor]:
        """
        Monte Carlo forward pass for uncertainty estimation.
        
        Args:
            x: Input tensor of shape [B, 3, H, W]
            n_samples: Number of forward passes
            
        Returns:
            Dictionary with:
            - 'mean_logits': Mean logits [B, num_classes]
            - 'std_logits': Std of logits [B, num_classes]
            - 'mean_probs': Mean probabilities [B, num_classes]
            - 'uncertainty': Uncertainty score [B]
        """
        self.train()  # Enable dropout
        
        logits_list = []
        for _ in range(n_samples):
            with torch.no_grad():
                logits = self.forward(x)
                logits_list.append(logits)
        
        logits_stack = torch.stack(logits_list)  # [n_samples, B, num_classes]
        
        # Calculate statistics
        mean_logits = logits_stack.mean(dim=0)
        std_logits = logits_stack.std(dim=0)
        
        # Convert to probabilities
        probs_stack = F.softmax(logits_stack, dim=-1)
        mean_probs = probs_stack.mean(dim=0)
        
        # Uncertainty: std of predicted class
        pred_classes = mean_probs.argmax(dim=-1)
        uncertainty = std_logits.gather(1, pred_classes.unsqueeze(1)).squeeze(1)
        
        self.eval()  # Disable dropout
        
        return {
            'mean_logits': mean_logits,
            'std_logits': std_logits,
            'mean_probs': mean_probs,
            'uncertainty': uncertainty,
            'predictions': pred_classes
        }


def create_model(
    model_name: str = 'efficientnet_b0',
    num_classes: int = 5,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    with_uncertainty: bool = False
) -> nn.Module:
    """
    Factory function to create RetinaModel.
    
    Args:
        model_name: Backbone model name
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout rate
        with_uncertainty: Whether to use uncertainty-aware model
        
    Returns:
        RetinaModel instance
    """
    ModelClass = RetinaModelWithUncertainty if with_uncertainty else RetinaModel
    
    return ModelClass(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        backbone=model_name
    )


def load_model(
    checkpoint_path: str,
    device: torch.device = torch.device('cpu'),
    **kwargs
) -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        **kwargs: Additional arguments for create_model
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = create_model(**kwargs)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model
