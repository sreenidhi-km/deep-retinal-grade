"""
Test-Time Augmentation (TTA) for Improved Predictions

TTA improves prediction accuracy and stability by:
1. Running multiple augmented versions of the input through the model
2. Averaging the predictions for a more robust final prediction

Typically provides +2-3% accuracy improvement with no model changes.

Author: Deep Retina Grade Project
Date: January 2026
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TTAPredictor:
    """
    Test-Time Augmentation predictor.
    
    Applies multiple augmentations at inference time and averages predictions.
    
    Supported TTA modes:
    - 'flip': Horizontal and vertical flips (5 variants)
    - 'rotate': 4 rotation angles (4 variants)
    - 'full': All augmentations (8 variants)
    - 'light': Just horizontal flip (2 variants) - fastest
    
    Args:
        model: PyTorch model
        device: Device to run on
        mode: TTA mode ('flip', 'rotate', 'full', 'light')
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        device: torch.device,
        mode: str = 'flip'
    ):
        self.model = model
        self.device = device
        self.mode = mode
        self.transforms = self._build_transforms()
        
    def _build_transforms(self) -> List[A.Compose]:
        """Build TTA transform list based on mode."""
        
        base = A.Compose([
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
        
        if self.mode == 'light':
            return [
                base,
                A.Compose([
                    A.HorizontalFlip(p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]),
            ]
        
        elif self.mode == 'flip':
            return [
                base,
                A.Compose([
                    A.HorizontalFlip(p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]),
                A.Compose([
                    A.VerticalFlip(p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]),
                A.Compose([
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]),
            ]
        
        elif self.mode == 'rotate':
            return [
                base,
                A.Compose([
                    A.Rotate(limit=(90, 90), p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]),
                A.Compose([
                    A.Rotate(limit=(180, 180), p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]),
                A.Compose([
                    A.Rotate(limit=(270, 270), p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]),
            ]
        
        else:  # 'full'
            return [
                base,
                A.Compose([
                    A.HorizontalFlip(p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]),
                A.Compose([
                    A.VerticalFlip(p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]),
                A.Compose([
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]),
                A.Compose([
                    A.Rotate(limit=(90, 90), p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]),
                A.Compose([
                    A.Rotate(limit=(180, 180), p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]),
                A.Compose([
                    A.Rotate(limit=(270, 270), p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]),
                A.Compose([
                    A.Rotate(limit=(90, 90), p=1.0),
                    A.HorizontalFlip(p=1.0),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]),
            ]
    
    @torch.no_grad()
    def predict(
        self, 
        image: np.ndarray,
        return_all: bool = False
    ) -> Tuple[int, float, torch.Tensor]:
        """
        Predict with TTA.
        
        Args:
            image: Input image as numpy array [H, W, C], uint8
            return_all: If True, return all individual predictions
            
        Returns:
            Tuple of (predicted_class, confidence, probabilities)
            If return_all=True, also returns all_probs [n_tta, num_classes]
        """
        self.model.eval()
        all_probs = []
        
        for transform in self.transforms:
            # Apply transform
            augmented = transform(image=image)
            img_tensor = augmented['image'].unsqueeze(0).to(self.device)
            
            # Get prediction
            logits = self.model(img_tensor)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)
        
        # Stack and average
        all_probs = torch.cat(all_probs, dim=0)  # [n_tta, num_classes]
        mean_probs = all_probs.mean(dim=0)  # [num_classes]
        
        # Get prediction
        pred_class = mean_probs.argmax().item()
        confidence = mean_probs[pred_class].item()
        
        if return_all:
            return pred_class, confidence, mean_probs, all_probs
        return pred_class, confidence, mean_probs
    
    @torch.no_grad()
    def predict_batch(
        self, 
        images: List[np.ndarray]
    ) -> Tuple[List[int], List[float], torch.Tensor]:
        """
        Predict a batch of images with TTA.
        
        Args:
            images: List of images as numpy arrays [H, W, C], uint8
            
        Returns:
            Tuple of (predicted_classes, confidences, probabilities)
        """
        self.model.eval()
        
        all_batch_probs = []
        
        for image in images:
            _, _, mean_probs = self.predict(image)
            all_batch_probs.append(mean_probs)
        
        # Stack results
        all_probs = torch.stack(all_batch_probs)  # [B, num_classes]
        pred_classes = all_probs.argmax(dim=1).tolist()
        confidences = all_probs.max(dim=1).values.tolist()
        
        return pred_classes, confidences, all_probs
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self, 
        image: np.ndarray
    ) -> dict:
        """
        Predict with TTA and compute uncertainty metrics.
        
        Uses the variance across TTA predictions as a measure of uncertainty.
        
        Args:
            image: Input image as numpy array [H, W, C], uint8
            
        Returns:
            Dictionary with prediction and uncertainty metrics
        """
        pred_class, confidence, mean_probs, all_probs = self.predict(image, return_all=True)
        
        # Compute uncertainty metrics
        std_probs = all_probs.std(dim=0)  # Standard deviation across TTA
        uncertainty = std_probs[pred_class].item()
        
        # Agreement: how many TTA predictions agree with final
        tta_predictions = all_probs.argmax(dim=1)
        agreement = (tta_predictions == pred_class).float().mean().item()
        
        # Entropy of mean predictions
        entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum().item()
        
        return {
            'predicted_class': pred_class,
            'confidence': confidence,
            'probabilities': mean_probs.cpu().numpy(),
            'uncertainty': uncertainty,
            'agreement': agreement,
            'entropy': entropy,
            'n_tta_samples': len(self.transforms),
            'is_stable': agreement > 0.7 and uncertainty < 0.1
        }


if __name__ == "__main__":
    # Quick test with dummy model
    print("Testing TTA predictor...")
    
    import torch.nn as nn
    
    # Simple dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 5)
            
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    device = torch.device('cpu')
    model = DummyModel().to(device)
    model.eval()
    
    # Create dummy image
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test different modes
    for mode in ['light', 'flip', 'rotate', 'full']:
        tta = TTAPredictor(model, device, mode=mode)
        pred, conf, probs = tta.predict(img)
        print(f"✅ TTA mode '{mode}': {len(tta.transforms)} variants, pred={pred}, conf={conf:.3f}")
    
    # Test with uncertainty
    tta = TTAPredictor(model, device, mode='flip')
    result = tta.predict_with_uncertainty(img)
    print(f"\n✅ Uncertainty prediction:")
    print(f"   Predicted class: {result['predicted_class']}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Uncertainty: {result['uncertainty']:.3f}")
    print(f"   Agreement: {result['agreement']:.1%}")
    print(f"   Is stable: {result['is_stable']}")
    
    print("\n✅ All TTA tests passed!")
