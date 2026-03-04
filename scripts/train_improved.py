"""
Improved Training Script for Diabetic Retinopathy Grading

This script implements all accuracy improvement techniques:
1. Focal Loss + Ordinal Loss (handles class imbalance + ordinal nature)
2. Strong Augmentations (better generalization)
3. Class Weighting (balances training)
4. Mixup (regularization)
5. Cosine Annealing with Warm Restarts (better optimization)
6. Early Stopping (prevents overfitting)

Expected improvement: +15-20% accuracy over baseline

Usage:
    python train_improved.py --epochs 50 --batch_size 32

Author: Deep Retina Grade Project
Date: January 2026
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report
from tqdm import tqdm
import cv2

# Add src to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Import our modules
from models.efficientnet import RetinaModel
from preprocessing.preprocess import RetinaPreprocessor
from training.losses import FocalLoss, CombinedLoss, compute_class_weights
from training.augmentations import get_train_transforms, get_val_transforms, MixupAugmentation, mixup_criterion


class RetinaDataset(Dataset):
    """Dataset for preprocessed fundus images."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        cache_dir: Path,
        transform=None
    ):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['id_code']
        label = row['diagnosis']
        
        # Load preprocessed image
        img_path = self.cache_dir / f"{img_id}.npy"
        if img_path.exists():
            img = np.load(img_path)
        else:
            raise FileNotFoundError(f"Preprocessed image not found: {img_path}")
        
        # Convert to uint8 for albumentations
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=img_uint8)
            img_tensor = transformed['image']
        else:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        
        return img_tensor, label


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    mixup=None,
    epoch=0
):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Apply mixup if enabled
        if mixup is not None and np.random.random() < 0.5:
            images, labels_a, labels_b, lam = mixup(images, labels)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute metrics
    avg_loss = running_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    return avg_loss, accuracy, kappa


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(val_loader, desc="Validation"):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    avg_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    return avg_loss, accuracy, kappa, all_preds, all_labels


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
        
        return self.should_stop


def main(args):
    """Main training function."""
    print("=" * 70)
    print("🚀 IMPROVED TRAINING SCRIPT")
    print("=" * 70)
    
    # Paths
    splits_dir = PROJECT_ROOT / "splits"
    cache_dir = PROJECT_ROOT / "cache" / "preprocessed_224"
    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"
    
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    print(f"📱 Device: {device}")
    
    # Load data
    df_train = pd.read_csv(splits_dir / "train.csv")
    df_val = pd.read_csv(splits_dir / "val.csv")
    df_test = pd.read_csv(splits_dir / "test.csv")
    
    print(f"📊 Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    # Class distribution
    train_labels = df_train['diagnosis'].tolist()
    class_counts = Counter(train_labels)
    print(f"📊 Class distribution: {dict(sorted(class_counts.items()))}")
    
    # Compute class weights
    class_weights = compute_class_weights(train_labels, num_classes=5)
    print(f"⚖️ Class weights: {class_weights.numpy()}")
    
    # Create datasets
    train_transform = get_train_transforms(img_size=224, level=args.aug_level)
    val_transform = get_val_transforms(img_size=224)
    
    train_dataset = RetinaDataset(df_train, cache_dir, transform=train_transform)
    val_dataset = RetinaDataset(df_val, cache_dir, transform=val_transform)
    test_dataset = RetinaDataset(df_test, cache_dir, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = RetinaModel(
        num_classes=5,
        pretrained=True,
        dropout_rate=args.dropout,
        backbone='efficientnet_b0'
    )
    model = model.to(device)
    print(f"🧠 Model: EfficientNet-B0 ({sum(p.numel() for p in model.parameters()):,} params)")
    
    # Loss function
    if args.loss == 'focal':
        criterion = FocalLoss(alpha=class_weights.to(device), gamma=args.focal_gamma)
        print(f"📉 Loss: Focal Loss (γ={args.focal_gamma})")
    elif args.loss == 'combined':
        criterion = CombinedLoss(
            num_classes=5,
            class_weights=class_weights.to(device),
            focal_gamma=args.focal_gamma,
            focal_weight=1.0,
            ordinal_weight=0.3
        )
        print(f"📉 Loss: Combined (Focal + Ordinal)")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(f"📉 Loss: CrossEntropy with class weights")
    
    # Mixup
    mixup = MixupAugmentation(alpha=args.mixup_alpha) if args.use_mixup else None
    if mixup:
        print(f"🔀 Mixup enabled (α={args.mixup_alpha})")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    
    # Training loop
    best_kappa = 0.0
    history = {
        'train_loss': [], 'train_acc': [], 'train_kappa': [],
        'val_loss': [], 'val_acc': [], 'val_kappa': []
    }
    
    print("\n" + "=" * 70)
    print("🏋️ TRAINING")
    print("=" * 70)
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc, train_kappa = train_one_epoch(
            model, train_loader, criterion, optimizer, device, mixup, epoch
        )
        
        # Validate
        val_loss, val_acc, val_kappa, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_kappa'].append(train_kappa)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_kappa'].append(val_kappa)
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Kappa: {train_kappa:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Kappa: {val_kappa:.4f}")
        
        # Save best model
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_kappa': best_kappa,
                'val_acc': val_acc,
            }, models_dir / 'efficientnet_b0_improved.pth')
            print(f"  ✅ New best model saved! Kappa: {best_kappa:.4f}")
        
        # Early stopping
        if early_stopping(val_kappa):
            print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    print(f"\n⏱️ Training time: {training_time/60:.1f} minutes")
    
    # Load best model and evaluate on test set
    print("\n" + "=" * 70)
    print("📊 FINAL EVALUATION")
    print("=" * 70)
    
    checkpoint = torch.load(models_dir / 'efficientnet_b0_improved.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, test_acc, test_kappa, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\n📈 Test Results:")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"  Quadratic Weighted Kappa: {test_kappa:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(
        test_labels, test_preds,
        target_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    ))
    
    # Save results
    results = {
        'accuracy': test_acc,
        'qwk': test_kappa,
        'best_val_kappa': best_kappa,
        'training_time_minutes': training_time / 60,
        'epochs_trained': len(history['train_loss']),
        'config': vars(args),
        'history': history
    }
    
    with open(results_dir / 'improved_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {results_dir / 'improved_training_results.json'}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved DR Training")
    
    # Training params
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    
    # Loss params
    parser.add_argument('--loss', type=str, default='combined', 
                       choices=['ce', 'focal', 'combined'], help='Loss function')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    
    # Augmentation params
    parser.add_argument('--aug_level', type=str, default='strong',
                       choices=['light', 'medium', 'strong'], help='Augmentation level')
    parser.add_argument('--use_mixup', action='store_true', help='Use mixup')
    parser.add_argument('--mixup_alpha', type=float, default=0.4, help='Mixup alpha')
    
    # Other params
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    
    args = parser.parse_args()
    
    main(args)
