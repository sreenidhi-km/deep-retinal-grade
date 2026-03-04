"""Preprocess DDR dataset images."""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from preprocessing.preprocess import RetinaPreprocessor

def main():
    # Paths
    project_root = Path(__file__).parent.parent
    ddr_csv = project_root / 'DDR Dataset' / 'DR_grading.csv'
    ddr_images = project_root / 'DDR Dataset' / 'DR_grading' / 'DR_grading'
    cache_dir = project_root / 'cache' / 'preprocessed_224'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load DDR data
    df = pd.read_csv(ddr_csv)
    print(f'DDR Dataset: {len(df)} images')
    
    # Check how many already cached
    already_cached = 0
    to_process = []
    for _, row in df.iterrows():
        img_id = row['id_code'].replace('.jpg', '')
        cache_path = cache_dir / f'{img_id}.npy'
        if cache_path.exists():
            already_cached += 1
        else:
            to_process.append(row)
    
    print(f'Already cached: {already_cached}')
    print(f'To process: {len(to_process)}')
    
    if len(to_process) == 0:
        print('All DDR images already preprocessed!')
        return
    
    # Initialize preprocessor
    preprocessor = RetinaPreprocessor(img_size=224)
    
    # Process images
    errors = []
    for row in tqdm(to_process, desc='Preprocessing DDR'):
        img_file = row['id_code']
        img_id = img_file.replace('.jpg', '')
        img_path = ddr_images / img_file
        
        if not img_path.exists():
            errors.append(img_file)
            continue
        
        try:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed = preprocessor.preprocess_array(img)
            np.save(cache_dir / f'{img_id}.npy', processed)
        except Exception as e:
            errors.append(f'{img_file}: {e}')
    
    print(f'Processed: {len(to_process) - len(errors)}')
    if errors:
        print(f'Errors: {len(errors)}')
        for e in errors[:5]:
            print(f'  {e}')

if __name__ == '__main__':
    main()
