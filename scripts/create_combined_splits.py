"""
Create combined APTOS + DDR splits with stratified sampling.
Focuses on adding more Grade 3 samples from DDR.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from preprocessing.preprocess import RetinaPreprocessor

def main():
    project_root = Path(__file__).parent.parent
    cache_dir = project_root / 'cache' / 'preprocessed_224'
    cache_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = project_root / 'splits'
    
    # Load APTOS data (already preprocessed)
    aptos_train = pd.read_csv(splits_dir / 'train.csv')
    aptos_val = pd.read_csv(splits_dir / 'val.csv')
    aptos_test = pd.read_csv(splits_dir / 'test.csv')
    
    print("=== APTOS (already preprocessed) ===")
    print(f"Train: {len(aptos_train)}, Val: {len(aptos_val)}, Test: {len(aptos_test)}")
    print(f"APTOS Grade 3: {(aptos_train['diagnosis'] == 3).sum()}")
    
    # Load DDR data
    ddr_csv = project_root / 'DDR Dataset' / 'DR_grading.csv'
    ddr_images = project_root / 'DDR Dataset' / 'DR_grading' / 'DR_grading'
    ddr_df = pd.read_csv(ddr_csv)
    
    # Remove .jpg extension for consistency
    ddr_df['id_code'] = ddr_df['id_code'].str.replace('.jpg', '', regex=False)
    
    print(f"\n=== DDR Dataset ===")
    print(f"Total: {len(ddr_df)}")
    for grade in range(5):
        count = (ddr_df['diagnosis'] == grade).sum()
        print(f"  Grade {grade}: {count}")
    
    # Strategy: Sample DDR to add ~2000 images, weighted toward minority classes
    # Goal: Double Grade 3 samples, add substantial Grade 4
    sample_per_class = {
        0: 300,   # No DR - just a few for diversity
        1: 200,   # Mild - some extra
        2: 500,   # Moderate - good representation  
        3: 200,   # Severe - ALL available (only 236 in DDR)
        4: 400,   # Proliferative - good amount
    }
    
    ddr_samples = []
    for grade, n_samples in sample_per_class.items():
        grade_df = ddr_df[ddr_df['diagnosis'] == grade]
        n_available = len(grade_df)
        n_to_sample = min(n_samples, n_available)
        sampled = grade_df.sample(n=n_to_sample, random_state=42)
        ddr_samples.append(sampled)
        print(f"  Sampling Grade {grade}: {n_to_sample} of {n_available}")
    
    ddr_sampled = pd.concat(ddr_samples, ignore_index=True)
    print(f"\nTotal DDR samples: {len(ddr_sampled)}")
    
    # Preprocess sampled DDR images
    print("\n=== Preprocessing DDR samples ===")
    preprocessor = RetinaPreprocessor(img_size=224)
    
    processed_ids = []
    errors = []
    
    for _, row in tqdm(ddr_sampled.iterrows(), total=len(ddr_sampled), desc='Preprocessing'):
        img_id = row['id_code']
        cache_path = cache_dir / f'{img_id}.npy'
        
        # Skip if already cached
        if cache_path.exists():
            processed_ids.append(img_id)
            continue
        
        img_path = ddr_images / f'{img_id}.jpg'
        if not img_path.exists():
            errors.append(img_id)
            continue
        
        try:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed = preprocessor.preprocess_array(img)
            np.save(cache_path, processed)
            processed_ids.append(img_id)
        except Exception as e:
            errors.append(f'{img_id}: {e}')
    
    print(f"Successfully processed: {len(processed_ids)}")
    if errors:
        print(f"Errors: {len(errors)}")
    
    # Filter to only successfully processed
    ddr_final = ddr_sampled[ddr_sampled['id_code'].isin(processed_ids)]
    
    # Split DDR into train/val (80/20)
    ddr_train, ddr_val = train_test_split(
        ddr_final, test_size=0.2, random_state=42, stratify=ddr_final['diagnosis']
    )
    
    # Combine with APTOS
    combined_train = pd.concat([aptos_train, ddr_train], ignore_index=True)
    combined_val = pd.concat([aptos_val, ddr_val], ignore_index=True)
    
    # Shuffle
    combined_train = combined_train.sample(frac=1, random_state=42).reset_index(drop=True)
    combined_val = combined_val.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n=== COMBINED DATASET ===")
    print(f"Train: {len(combined_train)}")
    print(f"Val: {len(combined_val)}")
    print(f"Test: {len(aptos_test)} (APTOS only, held out)")
    
    print("\nCombined Train distribution:")
    for grade in range(5):
        count = (combined_train['diagnosis'] == grade).sum()
        pct = count / len(combined_train) * 100
        print(f"  Grade {grade}: {count} ({pct:.1f}%)")
    
    # Save combined splits
    combined_dir = splits_dir / 'combined'
    combined_dir.mkdir(exist_ok=True)
    
    combined_train.to_csv(combined_dir / 'train.csv', index=False)
    combined_val.to_csv(combined_dir / 'val.csv', index=False)
    aptos_test.to_csv(combined_dir / 'test.csv', index=False)
    
    print(f"\n✅ Saved combined splits to {combined_dir}")

if __name__ == '__main__':
    main()
