#!/usr/bin/env python3
"""
Cache Preprocessed Images Script

This script precomputes and caches Ben Graham + CLAHE preprocessed images
to dramatically speed up training (from hours to minutes per epoch).

Usage:
    python scripts/cache_preprocessed_images.py

Output:
    Creates cached .npy files in PROJECT_ROOT/cache/preprocessed_224/
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse


def find_project_root(start_path: Path = None) -> Path:
    """
    Find project root by searching upward for marker directories.
    Looks for a directory containing both 'src/' and 'notebooks/'.
    """
    if start_path is None:
        start_path = Path(__file__).resolve().parent
    
    current = start_path
    for _ in range(10):  # Max 10 levels up
        if (current / "src").exists() and (current / "notebooks").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    
    raise RuntimeError(
        f"Could not find project root from {start_path}. "
        "Expected directory with 'src/' and 'notebooks/' subdirectories."
    )


def find_data_root(project_root: Path) -> Path:
    """
    Find the APTOS dataset directory by checking common locations.
    """
    candidates = [
        project_root / "aptos2019-blindness-detection",
        project_root.parent / "aptos2019-blindness-detection",
        project_root / "data" / "aptos2019-blindness-detection",
    ]
    
    for candidate in candidates:
        if candidate.exists() and (candidate / "train_images").exists():
            return candidate
    
    raise RuntimeError(
        f"Could not find APTOS dataset. Checked:\n" +
        "\n".join(f"  - {c}" for c in candidates)
    )


def cache_preprocessed_images(
    project_root: Path,
    img_size: int = 224,
    splits: list = None,
    force_recache: bool = False
):
    """
    Cache preprocessed images as uint8 numpy arrays.
    
    Args:
        project_root: Path to project root
        img_size: Target image size
        splits: List of splits to cache ('train', 'val', 'test')
        force_recache: If True, overwrite existing cache
    """
    if splits is None:
        splits = ['train', 'val', 'test']
    
    # Setup paths
    data_root = find_data_root(project_root)
    train_images_dir = data_root / "train_images"
    splits_dir = project_root / "splits"
    cache_dir = project_root / "cache" / f"preprocessed_{img_size}"
    src_dir = project_root / "src"
    
    # Add src to path for imports
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    # Import preprocessor
    from preprocessing.preprocess import RetinaPreprocessor
    
    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = RetinaPreprocessor(
        img_size=img_size,
        ben_graham_sigma=10,
        clahe_clip=2.0,
        clahe_grid=(8, 8)
    )
    
    print(f"{'='*70}")
    print(f"🗄️  PREPROCESSING IMAGE CACHE")
    print(f"{'='*70}")
    print(f"Project Root: {project_root}")
    print(f"Data Root: {data_root}")
    print(f"Train Images: {train_images_dir}")
    print(f"Cache Dir: {cache_dir}")
    print(f"Image Size: {img_size}x{img_size}")
    print(f"Splits to cache: {splits}")
    print(f"{'='*70}\n")
    
    total_cached = 0
    total_skipped = 0
    
    for split in splits:
        split_csv = splits_dir / f"{split}.csv"
        if not split_csv.exists():
            print(f"⚠️  Skipping {split}: {split_csv} not found")
            continue
        
        df = pd.read_csv(split_csv)
        print(f"\n📂 Processing {split} split ({len(df)} images)...")
        
        cached = 0
        skipped = 0
        errors = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split}"):
            img_id = row['id_code']
            cache_path = cache_dir / f"{img_id}.npy"
            
            # Skip if already cached (unless force_recache)
            if cache_path.exists() and not force_recache:
                skipped += 1
                continue
            
            img_path = train_images_dir / f"{img_id}.png"
            
            try:
                # Preprocess: returns float32 [0,1]
                img_float = preprocessor.preprocess(
                    img_path,
                    apply_ben_graham=True,
                    apply_clahe=True
                )
                
                # Convert to uint8 for storage efficiency
                img_u8 = (img_float * 255).clip(0, 255).astype(np.uint8)
                
                # Save as .npy
                np.save(cache_path, img_u8)
                cached += 1
                
            except Exception as e:
                errors.append((img_id, str(e)))
        
        total_cached += cached
        total_skipped += skipped
        
        print(f"  ✅ Cached: {cached}, Skipped (exists): {skipped}")
        if errors:
            print(f"  ⚠️  Errors: {len(errors)}")
            for img_id, err in errors[:5]:
                print(f"      {img_id}: {err}")
    
    print(f"\n{'='*70}")
    print(f"🎉 CACHING COMPLETE!")
    print(f"{'='*70}")
    print(f"Total cached: {total_cached}")
    print(f"Total skipped: {total_skipped}")
    print(f"Cache location: {cache_dir}")
    
    # Count total cached files
    cached_files = list(cache_dir.glob("*.npy"))
    print(f"Total .npy files in cache: {len(cached_files)}")
    print(f"{'='*70}")
    
    return cache_dir


def main():
    parser = argparse.ArgumentParser(description="Cache preprocessed fundus images")
    parser.add_argument("--img-size", type=int, default=224, help="Image size (default: 224)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], 
                       help="Splits to cache (default: train val test)")
    parser.add_argument("--force", action="store_true", help="Force recache existing files")
    
    args = parser.parse_args()
    
    # Find project root
    project_root = find_project_root()
    
    # Run caching
    cache_preprocessed_images(
        project_root=project_root,
        img_size=args.img_size,
        splits=args.splits,
        force_recache=args.force
    )


if __name__ == "__main__":
    main()
