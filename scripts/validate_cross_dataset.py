#!/usr/bin/env python3
"""
Phase 4: Cross-Dataset External Validation

Validates the APTOS+DDR trained model on IDRiD dataset (516 images)
to assess generalization capability.

Author: Deep Retina Grade Team
Date: February 2026
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    """Compute Quadratic Weighted Kappa for ordinal classification."""
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def ben_graham_preprocessing(image, sigmaX=10):
    """Apply Ben Graham's preprocessing for fundus images."""
    # Convert to float
    image = image.astype(np.float32)
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX)
    
    # Subtract local mean
    result = cv2.addWeighted(image, 4, blurred, -4, 128)
    
    return result


def preprocess_image(image_path, target_size=224, cache_dir=None):
    """Preprocess a single fundus image with caching."""
    image_path = Path(image_path)
    
    # Check cache
    if cache_dir:
        cache_path = Path(cache_dir) / f"{image_path.stem}.npy"
        if cache_path.exists():
            return np.load(cache_path)
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, (target_size, target_size))
    
    # Ben Graham preprocessing
    image = ben_graham_preprocessing(image)
    
    # Clip and convert to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Save to cache
    if cache_dir:
        cache_path = Path(cache_dir) / f"{image_path.stem}.npy"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, image)
    
    return image


class IDRiDDataset(Dataset):
    """IDRiD External Validation Dataset."""
    
    def __init__(self, data_df, image_dirs, cache_dir=None, transform=None):
        """
        Args:
            data_df: DataFrame with 'Image name' and 'Retinopathy grade' columns
            image_dirs: List of directories to search for images
            cache_dir: Directory for preprocessed image cache
            transform: PyTorch transforms
        """
        self.data = data_df.reset_index(drop=True)
        self.image_dirs = [Path(d) for d in image_dirs]
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.transform = transform
        
        # Build image path mapping
        self.image_paths = {}
        for idx, row in self.data.iterrows():
            image_name = row['Image name']
            found = False
            for img_dir in self.image_dirs:
                # Try different extensions
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    path = img_dir / f"{image_name}{ext}"
                    if path.exists():
                        self.image_paths[idx] = path
                        found = True
                        break
                if found:
                    break
            if not found:
                print(f"Warning: Image not found for {image_name}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if idx not in self.image_paths:
            raise ValueError(f"No image path for index {idx}")
        
        image_path = self.image_paths[idx]
        label = int(self.data.iloc[idx]['Retinopathy grade'])
        
        # Load/preprocess image
        image = preprocess_image(image_path, cache_dir=self.cache_dir)
        
        # Convert to tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, str(image_path)


def load_idrid_labels(project_root):
    """Load and combine IDRiD training and testing labels."""
    idrid_base = project_root / "B. Disease Grading"
    
    # Load training labels
    train_csv = idrid_base / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"
    train_df = pd.read_csv(train_csv)
    train_df = train_df[['Image name', 'Retinopathy grade']].copy()
    train_df['split'] = 'train'
    
    # Load testing labels
    test_csv = idrid_base / "2. Groundtruths" / "b. IDRiD_Disease Grading_Testing Labels.csv"
    test_df = pd.read_csv(test_csv)
    test_df = test_df[['Image name', 'Retinopathy grade']].copy()
    test_df['split'] = 'test'
    
    # Combine
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"IDRiD Dataset: {len(train_df)} train + {len(test_df)} test = {len(combined_df)} total")
    print(f"Grade distribution:\n{combined_df['Retinopathy grade'].value_counts().sort_index()}")
    
    return combined_df


def load_model(model_path, num_classes=5, device='cpu'):
    """Load the trained EfficientNet-B0 model using project's RetinaModel class."""
    from src.models.efficientnet import RetinaModel
    
    # Create model architecture matching the training setup
    model = RetinaModel(num_classes=num_classes, pretrained=False)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
            print(f"  Best kappa: {checkpoint.get('best_kappa', 'N/A'):.4f}")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def run_inference(model, dataloader, device):
    """Run inference on a dataset."""
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Running inference"):
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), all_paths


def compute_metrics(y_true, y_pred, y_probs):
    """Compute comprehensive metrics."""
    grade_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    # Overall metrics
    qwk = quadratic_weighted_kappa(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    
    # Per-class results
    per_class = {}
    for i, name in enumerate(grade_names):
        per_class[f"grade_{i}_{name.lower().replace(' ', '_')}"] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        }
    
    # Confidence analysis
    pred_confidence = y_probs.max(axis=1)
    
    results = {
        "overall": {
            "quadratic_weighted_kappa": float(qwk),
            "accuracy": float(accuracy),
            "total_samples": int(len(y_true))
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confidence": {
            "mean": float(pred_confidence.mean()),
            "std": float(pred_confidence.std()),
            "min": float(pred_confidence.min()),
            "max": float(pred_confidence.max())
        }
    }
    
    return results


def print_results(results, title="External Validation Results"):
    """Pretty print validation results."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)
    
    overall = results['overall']
    print(f"\n📊 Overall Metrics:")
    print(f"   QWK:      {overall['quadratic_weighted_kappa']:.4f}")
    print(f"   Accuracy: {overall['accuracy']:.4f} ({overall['accuracy']*100:.1f}%)")
    print(f"   Samples:  {overall['total_samples']}")
    
    print(f"\n📈 Per-Class Performance:")
    print(f"   {'Grade':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"   {'-'*55}")
    
    grade_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    for i, name in enumerate(grade_names):
        key = f"grade_{i}_{name.lower().replace(' ', '_')}"
        cls = results['per_class'][key]
        print(f"   {name:<15} {cls['precision']:>10.3f} {cls['recall']:>10.3f} {cls['f1']:>10.3f} {cls['support']:>10}")
    
    print(f"\n🎯 Confusion Matrix:")
    cm = np.array(results['confusion_matrix'])
    print(f"   {'':>12}", end='')
    for name in ['No DR', 'Mild', 'Mod', 'Sev', 'Prol']:
        print(f"{name:>8}", end='')
    print()
    
    for i, name in enumerate(['No DR', 'Mild', 'Moderate', 'Severe', 'Prolif']):
        print(f"   {name:<12}", end='')
        for j in range(5):
            print(f"{cm[i, j]:>8}", end='')
        print()
    
    conf = results['confidence']
    print(f"\n💡 Confidence Analysis:")
    print(f"   Mean: {conf['mean']:.3f} | Std: {conf['std']:.3f} | Range: [{conf['min']:.3f}, {conf['max']:.3f}]")
    
    print("\n" + "=" * 60)


def main():
    """Main validation pipeline."""
    print("=" * 60)
    print(" Phase 4: External Validation on IDRiD Dataset")
    print("=" * 60)
    
    # Paths
    project_root = PROJECT_ROOT
    model_path = project_root / "models" / "efficientnet_b0_combined.pth"
    cache_dir = project_root / "cache" / "preprocessed_224"
    results_path = project_root / "results" / "external_validation.json"
    
    # Check model exists
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Trying alternative model paths...")
        for alt in ["efficientnet_b0_best.pth", "efficientnet_b0_improved.pth"]:
            alt_path = project_root / "models" / alt
            if alt_path.exists():
                model_path = alt_path
                print(f"   Found: {model_path}")
                break
    
    if not model_path.exists():
        raise FileNotFoundError(f"No model found in {project_root / 'models'}")
    
    print(f"\n📁 Model: {model_path.name}")
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"🖥️  Device: {device}")
    
    # Load IDRiD labels
    print("\n📂 Loading IDRiD dataset...")
    idrid_df = load_idrid_labels(project_root)
    
    # Image directories
    idrid_base = project_root / "B. Disease Grading" / "1. Original Images"
    image_dirs = [
        idrid_base / "a. Training Set",
        idrid_base / "b. Testing Set"
    ]
    
    # Transforms (normalization for EfficientNet)
    transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Create dataset and dataloader
    print("\n🔄 Preprocessing images...")
    dataset = IDRiDDataset(
        idrid_df,
        image_dirs,
        cache_dir=cache_dir,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,  # MPS doesn't support multiprocessing well
        pin_memory=False
    )
    
    # Load model
    print(f"\n🧠 Loading model...")
    model = load_model(model_path, device=device)
    print(f"   Model loaded successfully")
    
    # Run inference
    print(f"\n🚀 Running inference on {len(dataset)} images...")
    start_time = datetime.now()
    
    y_pred, y_true, y_probs, paths = run_inference(model, dataloader, device)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"   Completed in {elapsed:.1f}s ({len(dataset)/elapsed:.1f} images/sec)")
    
    # Compute metrics
    print("\n📊 Computing metrics...")
    results = compute_metrics(y_true, y_pred, y_probs)
    
    # Add metadata
    results['metadata'] = {
        "model": model_path.name,
        "dataset": "IDRiD",
        "total_images": len(dataset),
        "inference_time_seconds": elapsed,
        "device": str(device),
        "timestamp": datetime.now().isoformat()
    }
    
    # Print results
    print_results(results, "IDRiD External Validation Results")
    
    # Grade 3 specific analysis
    grade3_mask = y_true == 3
    if grade3_mask.sum() > 0:
        grade3_preds = y_pred[grade3_mask]
        grade3_correct = (grade3_preds == 3).sum()
        grade3_total = grade3_mask.sum()
        grade3_as_moderate = (grade3_preds == 2).sum()
        grade3_as_prolif = (grade3_preds == 4).sum()
        
        print(f"\n🔍 Grade 3 (Severe) Deep Dive:")
        print(f"   Total Grade 3 samples: {grade3_total}")
        print(f"   Correctly classified: {grade3_correct} ({100*grade3_correct/grade3_total:.1f}%)")
        print(f"   Misclassified as Moderate: {grade3_as_moderate} ({100*grade3_as_moderate/grade3_total:.1f}%)")
        print(f"   Misclassified as Proliferative: {grade3_as_prolif} ({100*grade3_as_prolif/grade3_total:.1f}%)")
        
        results['grade3_analysis'] = {
            "total": int(grade3_total),
            "correct": int(grade3_correct),
            "recall": float(grade3_correct / grade3_total),
            "misclassified_as_moderate": int(grade3_as_moderate),
            "misclassified_as_proliferative": int(grade3_as_prolif)
        }
    
    # Save results
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {results_path}")
    
    # Summary
    qwk = results['overall']['quadratic_weighted_kappa']
    target_met = "✅" if qwk >= 0.60 else "❌"
    print(f"\n📋 Summary:")
    print(f"   External QWK: {qwk:.4f} {target_met} (target: ≥0.60)")
    print(f"   This validates that the model generalizes to unseen data from a different source.")
    
    return results


if __name__ == "__main__":
    results = main()
