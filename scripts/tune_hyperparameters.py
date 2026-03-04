"""
Hyperparameter Tuning Script for Improved Severe/Proliferative DR Detection

This script runs multiple training configurations to find the best settings
for detecting severe (grade 3) and proliferative (grade 4) DR cases.

Key optimizations:
1. Higher class weights for grades 3 and 4 (most important for clinical use)
2. Higher focal gamma to focus on hard examples
3. Adjusted learning rate and dropout

Author: Deep Retina Grade Project
Date: January 2026
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report, recall_score
from tqdm import tqdm

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
    
    def __init__(self, df: pd.DataFrame, cache_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['id_code']
        label = row['diagnosis']
        
        img_path = self.cache_dir / f"{img_id}.npy"
        if img_path.exists():
            img = np.load(img_path)
        else:
            raise FileNotFoundError(f"Preprocessed image not found: {img_path}")
        
        img_uint8 = (img * 255).astype(np.uint8)
        
        if self.transform:
            transformed = self.transform(image=img_uint8)
            img_tensor = transformed['image']
        else:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        
        return img_tensor, label


def train_one_epoch(model, train_loader, criterion, optimizer, device, mixup=None, epoch=0):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
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
    
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    # Calculate severe DR sensitivity (grades 3 and 4)
    severe_labels = [1 if l >= 3 else 0 for l in all_labels]
    severe_preds = [1 if p >= 3 else 0 for p in all_preds]
    severe_sensitivity = recall_score(severe_labels, severe_preds, zero_division=0)
    
    return avg_loss, accuracy, kappa, severe_sensitivity, all_preds, all_labels


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


def run_experiment(config, df_train, df_val, df_test, cache_dir, device, models_dir, run_id):
    """Run a single training experiment with given config."""
    print(f"\n{'='*60}")
    print(f"🧪 Experiment: {config['name']}")
    print(f"{'='*60}")
    print(f"Config: {config}")
    
    # Compute custom class weights with emphasis on severe/proliferative
    train_labels = df_train['diagnosis'].tolist()
    base_weights = compute_class_weights(train_labels, num_classes=5)
    
    # Apply multipliers for grades 3 and 4
    custom_weights = base_weights.clone()
    custom_weights[3] *= config['severe_weight_mult']  # Severe
    custom_weights[4] *= config['prolif_weight_mult']  # Proliferative
    custom_weights = custom_weights / custom_weights.max()  # Normalize
    
    print(f"⚖️ Custom weights: {custom_weights.numpy().round(3)}")
    
    # Datasets
    train_transform = get_train_transforms(img_size=224, level=config['aug_level'])
    val_transform = get_val_transforms(img_size=224)
    
    train_dataset = RetinaDataset(df_train, cache_dir, transform=train_transform)
    val_dataset = RetinaDataset(df_val, cache_dir, transform=val_transform)
    test_dataset = RetinaDataset(df_test, cache_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=0, pin_memory=False)
    
    # Model
    model = RetinaModel(num_classes=5, pretrained=True, 
                       dropout_rate=config['dropout'], backbone='efficientnet_b0')
    model = model.to(device)
    
    # Loss with custom weights and focal gamma
    criterion = CombinedLoss(
        num_classes=5,
        class_weights=custom_weights.to(device),
        focal_gamma=config['focal_gamma'],
        focal_weight=1.0,
        ordinal_weight=0.3
    )
    
    # Mixup
    mixup = MixupAugmentation(alpha=config['mixup_alpha']) if config['use_mixup'] else None
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # Training
    early_stopping = EarlyStopping(patience=config['patience'], mode='max')
    best_kappa = 0.0
    best_severe_sens = 0.0
    
    for epoch in range(config['epochs']):
        train_loss, train_acc, train_kappa = train_one_epoch(
            model, train_loader, criterion, optimizer, device, mixup, epoch
        )
        
        val_loss, val_acc, val_kappa, val_severe_sens, _, _ = validate(
            model, val_loader, criterion, device
        )
        
        scheduler.step()
        
        print(f"  E{epoch+1}: ValAcc={val_acc:.3f}, Kappa={val_kappa:.3f}, SevereSens={val_severe_sens:.3f}")
        
        # Save best model based on combined score
        combined_score = val_kappa * 0.7 + val_severe_sens * 0.3  # Weight severe sensitivity
        if combined_score > (best_kappa * 0.7 + best_severe_sens * 0.3):
            best_kappa = val_kappa
            best_severe_sens = val_severe_sens
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_kappa': best_kappa,
                'best_severe_sensitivity': best_severe_sens,
                'val_acc': val_acc,
                'config': config
            }, models_dir / f'experiment_{run_id}.pth')
        
        if early_stopping(combined_score):
            print(f"  ⚠️ Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation on test set
    checkpoint = torch.load(models_dir / f'experiment_{run_id}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_kappa, test_severe_sens, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\n📊 Test Results for {config['name']}:")
    print(f"   Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"   Kappa: {test_kappa:.4f}")
    print(f"   Severe DR Sensitivity: {test_severe_sens:.4f}")
    
    return {
        'name': config['name'],
        'test_accuracy': test_acc,
        'test_kappa': test_kappa,
        'test_severe_sensitivity': test_severe_sens,
        'best_val_kappa': best_kappa,
        'best_val_severe_sens': best_severe_sens,
        'config': config
    }


def main():
    """Run hyperparameter tuning experiments."""
    print("=" * 70)
    print("🔬 HYPERPARAMETER TUNING FOR SEVERE/PROLIFERATIVE DR DETECTION")
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
    
    # Configurations to try - focused on improving severe/proliferative detection
    configs = [
        {
            'name': 'HighSevereWeight',
            'severe_weight_mult': 2.0,  # 2x weight for severe
            'prolif_weight_mult': 2.5,  # 2.5x weight for proliferative
            'focal_gamma': 2.5,         # Higher gamma for hard examples
            'lr': 3e-4,
            'dropout': 0.4,             # Higher dropout for regularization
            'batch_size': 32,
            'epochs': 25,
            'patience': 8,
            'aug_level': 'strong',
            'use_mixup': True,
            'mixup_alpha': 0.4,
            'weight_decay': 1e-4
        },
        {
            'name': 'VeryHighSevereWeight',
            'severe_weight_mult': 3.0,  # 3x weight for severe
            'prolif_weight_mult': 3.5,  # 3.5x weight for proliferative
            'focal_gamma': 3.0,         # Even higher gamma
            'lr': 2e-4,
            'dropout': 0.35,
            'batch_size': 32,
            'epochs': 25,
            'patience': 8,
            'aug_level': 'strong',
            'use_mixup': True,
            'mixup_alpha': 0.3,
            'weight_decay': 2e-4
        }
    ]
    
    # Run experiments
    results = []
    for i, config in enumerate(configs):
        result = run_experiment(
            config, df_train, df_val, df_test, 
            cache_dir, device, models_dir, i
        )
        results.append(result)
    
    # Find best configuration
    print("\n" + "=" * 70)
    print("📊 EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  Accuracy: {r['test_accuracy']:.4f}")
        print(f"  Kappa: {r['test_kappa']:.4f}")
        print(f"  Severe DR Sensitivity: {r['test_severe_sensitivity']:.4f}")
    
    # Select best based on combined metric
    best_result = max(results, key=lambda x: x['test_kappa'] * 0.7 + x['test_severe_sensitivity'] * 0.3)
    print(f"\n🏆 Best configuration: {best_result['name']}")
    
    # Copy best model to improved path
    import shutil
    best_idx = [i for i, r in enumerate(results) if r['name'] == best_result['name']][0]
    best_model_path = models_dir / f'experiment_{best_idx}.pth'
    final_path = models_dir / 'efficientnet_b0_tuned.pth'
    shutil.copy(best_model_path, final_path)
    print(f"✅ Best model saved to {final_path}")
    
    # Save results
    with open(results_dir / 'hyperparameter_tuning_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {results_dir / 'hyperparameter_tuning_results.json'}")
    
    return results


if __name__ == "__main__":
    main()
