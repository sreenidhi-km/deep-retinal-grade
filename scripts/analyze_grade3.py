"""Analyze Config B results - confusion matrix and Grade 3 misclassifications."""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models.efficientnet import RetinaModel
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    project_root = Path(__file__).parent.parent
    cache_dir = project_root / 'cache' / 'preprocessed_224'
    
    # Load model
    model = RetinaModel(num_classes=5, pretrained=False, backbone='efficientnet_b0')
    checkpoint = torch.load(project_root / 'models' / 'efficientnet_b0_combined.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test data
    df_test = pd.read_csv(project_root / 'splits' / 'test.csv')
    print(f'Test set: {len(df_test)} images')
    print(f'Grade 3 in test: {(df_test["diagnosis"] == 3).sum()} images')
    
    # Simple dataset
    class SimpleDataset(Dataset):
        def __init__(self, df, cache_dir):
            self.df = df.reset_index(drop=True)
            self.cache_dir = cache_dir
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img = np.load(self.cache_dir / f"{row['id_code']}.npy")
            img_uint8 = (img * 255).astype(np.uint8)
            transformed = self.transform(image=img_uint8)
            return transformed['image'], row['diagnosis'], row['id_code']
    
    dataset = SimpleDataset(df_test, cache_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Predict
    all_preds, all_labels, all_ids, all_probs = [], [], [], []
    
    with torch.no_grad():
        for images, labels, ids in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_ids.extend(ids)
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_ids = np.array(all_ids)
    
    # Confusion Matrix
    print('\n' + '='*60)
    print('CONFUSION MATRIX')
    print('='*60)
    cm = confusion_matrix(all_labels, all_preds)
    labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Prolif']
    print(f'{"":>10} | ' + ' | '.join(f'{l:>8}' for l in labels))
    print('-' * 70)
    for i, (row, label) in enumerate(zip(cm, labels)):
        print(f'{label:>10} | ' + ' | '.join(f'{v:>8}' for v in row))
    
    # Grade 3 analysis
    print('\n' + '='*60)
    print('GRADE 3 (SEVERE) MISCLASSIFICATION ANALYSIS')
    print('='*60)
    grade3_mask = all_labels == 3
    grade3_preds = all_preds[grade3_mask]
    grade3_probs = all_probs[grade3_mask]
    grade3_ids = all_ids[grade3_mask]
    
    correct = (grade3_preds == 3).sum()
    total = len(grade3_preds)
    print(f'Total Grade 3 samples: {total}')
    print(f'Correctly predicted: {correct} ({correct/total*100:.1f}%)')
    print(f'\nMisclassified as:')
    for pred_class, name in enumerate(['No DR', 'Mild', 'Moderate', 'Severe', 'Prolif']):
        if pred_class != 3:
            count = (grade3_preds == pred_class).sum()
            if count > 0:
                print(f'  {name}: {count} ({count/total*100:.1f}%)')
    
    # Detailed misclassified cases
    print('\nMisclassified Grade 3 cases (top 10):')
    misclassified_mask = grade3_preds != 3
    for i, (img_id, pred, probs) in enumerate(zip(
        grade3_ids[misclassified_mask][:10],
        grade3_preds[misclassified_mask][:10],
        grade3_probs[misclassified_mask][:10]
    )):
        pred_name = ['No DR', 'Mild', 'Moderate', 'Severe', 'Prolif'][pred]
        print(f'  {img_id}: pred={pred_name}, P(Severe)={probs[3]:.3f}, P(pred)={probs[pred]:.3f}')
    
    # Confidence analysis
    print('\n' + '='*60)
    print('CONFIDENCE ANALYSIS BY CLASS')
    print('='*60)
    for grade, name in enumerate(['No DR', 'Mild', 'Moderate', 'Severe', 'Prolif']):
        mask = all_labels == grade
        correct_mask = mask & (all_preds == all_labels)
        if correct_mask.sum() > 0:
            avg_conf = all_probs[correct_mask, grade].mean()
            print(f'{name:>10}: Avg confidence when correct = {avg_conf:.3f}')
    
    # Grade 3 confidence distribution
    print('\nGrade 3 confidence distribution:')
    grade3_conf = all_probs[grade3_mask, 3]
    print(f'  Min:    {grade3_conf.min():.3f}')
    print(f'  Max:    {grade3_conf.max():.3f}')
    print(f'  Mean:   {grade3_conf.mean():.3f}')
    print(f'  Median: {np.median(grade3_conf):.3f}')
    
    # Key insight
    print('\n' + '='*60)
    print('KEY INSIGHTS')
    print('='*60)
    grade2_as_3 = cm[2, 3]  # Moderate predicted as Severe
    grade3_as_2 = cm[3, 2]  # Severe predicted as Moderate
    grade3_as_4 = cm[3, 4]  # Severe predicted as Prolif
    grade4_as_3 = cm[4, 3]  # Prolif predicted as Severe
    
    print(f'Grade 2 (Moderate) misclassified as Grade 3: {grade2_as_3}')
    print(f'Grade 3 (Severe) misclassified as Grade 2: {grade3_as_2}')
    print(f'Grade 3 (Severe) misclassified as Grade 4: {grade3_as_4}')
    print(f'Grade 4 (Prolif) misclassified as Grade 3: {grade4_as_3}')
    
    # Save analysis
    analysis = {
        'confusion_matrix': cm.tolist(),
        'grade_3_total': int(total),
        'grade_3_correct': int(correct),
        'grade_3_recall': float(correct/total),
        'grade_3_as_grade_2': int(grade3_as_2),
        'grade_3_as_grade_4': int(grade3_as_4),
        'grade_3_confidence_mean': float(grade3_conf.mean()),
        'misclassified_ids': grade3_ids[misclassified_mask].tolist()
    }
    
    import json
    with open(project_root / 'results' / 'grade3_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f'\nAnalysis saved to results/grade3_analysis.json')

if __name__ == '__main__':
    main()
