"""
Advanced Augmentation Pipeline for Diabetic Retinopathy Grading

This module implements medical-imaging-specific augmentations that
significantly improve model generalization and accuracy.

Key features:
- Geometric augmentations (rotation, flip, scale)
- Color augmentations (brightness, contrast, saturation)
- Noise augmentations (simulates different camera qualities)
- Cutout/CoarseDropout (regularization)

Author: Deep Retina Grade Project
Date: January 2026
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# ImageNet normalization (standard for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224, level: str = 'strong') -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Three levels of augmentation:
    - 'light': Basic flips and rotations (fast training)
    - 'medium': + color jitter and noise
    - 'strong': + cutout and advanced transforms (best accuracy)
    
    Args:
        img_size: Target image size (default: 224)
        level: Augmentation intensity ('light', 'medium', 'strong')
        
    Returns:
        Albumentations composition
    """
    
    if level == 'light':
        return A.Compose([
            # Basic geometric
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            # Normalize and convert
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    
    elif level == 'medium':
        return A.Compose([
            # Geometric
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=45,
                border_mode=0,  # Constant border
                p=0.5
            ),
            
            # Color
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
            
            # Normalize and convert
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    
    else:  # 'strong'
        return A.Compose([
            # Geometric transforms (fundus-safe)
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)},
                scale=(0.85, 1.15),
                rotate=(-45, 45),
                border_mode=0,
                p=0.6
            ),
            
            # Sometimes apply slight distortion (simulates lens variations)
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.05),
                A.GridDistortion(num_steps=5, distort_limit=0.05),
            ], p=0.2),
            
            # Color transforms (important for fundus images)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=15,
                p=0.4
            ),
            
            # Simulate different camera qualities (std_range must be 0-1)
            A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.1)),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MotionBlur(blur_limit=3),
            ], p=0.25),
            
            # Simulate different image qualities
            A.OneOf([
                A.ImageCompression(quality_range=(70, 100)),
                A.Downscale(scale_range=(0.7, 0.95)),
            ], p=0.2),
            
            # Regularization via cutout (drops random patches)
            A.CoarseDropout(
                num_holes_range=(2, 8),
                hole_height_range=(int(img_size * 0.05), int(img_size * 0.1)),
                hole_width_range=(int(img_size * 0.05), int(img_size * 0.1)),
                fill=0,
                p=0.4
            ),
            
            # Normalize and convert
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


def get_val_transforms(img_size: int = 224) -> A.Compose:
    """
    Get validation/test augmentation pipeline (no augmentation, just normalize).
    
    Args:
        img_size: Target image size (default: 224)
        
    Returns:
        Albumentations composition
    """
    return A.Compose([
        # Just normalize - no augmentation for validation
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size: int = 224) -> list:
    """
    Get Test-Time Augmentation transforms.
    
    Returns a list of transforms to apply at test time.
    Predictions are averaged for more stable results.
    
    Args:
        img_size: Target image size (default: 224)
        
    Returns:
        List of Albumentations compositions
    """
    base_transform = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    
    transforms = [
        # Original (no augmentation)
        base_transform,
        
        # Horizontal flip
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
        
        # Vertical flip
        A.Compose([
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
        
        # 90 degree rotation
        A.Compose([
            A.Rotate(limit=(90, 90), p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
        
        # 180 degree rotation
        A.Compose([
            A.Rotate(limit=(180, 180), p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
    ]
    
    return transforms


class MixupAugmentation:
    """
    Mixup augmentation for training.
    
    Mixes pairs of images and labels:
    - x_mixed = λ * x_i + (1-λ) * x_j
    - y_mixed = λ * y_i + (1-λ) * y_j
    
    This improves generalization by creating interpolated training samples.
    
    Paper: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2017)
    
    Args:
        alpha: Mixup interpolation coefficient (default: 0.4)
            Higher = more mixing, Lower = closer to original
    """
    
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        
    def __call__(self, images, labels):
        """
        Apply mixup to a batch.
        
        Args:
            images: Tensor of shape [B, C, H, W]
            labels: Tensor of shape [B]
            
        Returns:
            mixed_images, labels_a, labels_b, lambda
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = np.random.permutation(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute mixup loss.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a, y_b: Original and shuffled labels
        lam: Mixup coefficient
        
    Returns:
        Mixed loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if __name__ == "__main__":
    # Quick test
    print("Testing augmentation pipelines...")
    
    # Create dummy image
    import torch
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test light transforms
    light = get_train_transforms(level='light')
    result = light(image=img)
    print(f"✅ Light transforms: {result['image'].shape}")
    
    # Test medium transforms
    medium = get_train_transforms(level='medium')
    result = medium(image=img)
    print(f"✅ Medium transforms: {result['image'].shape}")
    
    # Test strong transforms
    strong = get_train_transforms(level='strong')
    result = strong(image=img)
    print(f"✅ Strong transforms: {result['image'].shape}")
    
    # Test val transforms
    val = get_val_transforms()
    result = val(image=img)
    print(f"✅ Validation transforms: {result['image'].shape}")
    
    # Test TTA transforms
    tta = get_tta_transforms()
    print(f"✅ TTA transforms: {len(tta)} variants")
    
    # Test Mixup
    images = torch.randn(8, 3, 224, 224)
    labels = torch.randint(0, 5, (8,))
    mixup = MixupAugmentation(alpha=0.4)
    mixed, y_a, y_b, lam = mixup(images, labels)
    print(f"✅ Mixup: λ={lam:.3f}, mixed shape={mixed.shape}")
    
    print("\n✅ All augmentation pipelines working correctly!")
