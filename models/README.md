# Model Checkpoints

## Available Models

| Checkpoint | QWK | Accuracy | Description |
|-----------|-----|----------|-------------|
| `efficientnet_b0_best.pth` | 0.82 | 76.2% | Phase 2 baseline |
| `efficientnet_b0_improved.pth` | 0.87 | 78.5% | Phase 3 with FocalLoss + Mixup |
| `efficientnet_b0_tuned.pth` | 0.85 | 77.8% | Hyperparameter tuned (better Grade 3) |
| `efficientnet_b0_combined.pth` | 0.8745 | 78.5% | Combined dataset (APTOS + DDR) |

## Recommended Model

For production inference, use **`efficientnet_b0_combined.pth`** — it achieves the best overall QWK (0.8745) on the APTOS test set.

For Grade 3 sensitivity-prioritized use cases, consider **`efficientnet_b0_tuned.pth`** which has higher severe DR recall.

## Architecture

- **Backbone:** EfficientNet-B0 (via `timm`)
- **Input:** 224×224 RGB, ImageNet-normalized
- **Output:** 5-class logits (DR Grades 0–4)
- **Head:** GlobalAvgPool → Dropout(0.3) → Linear(1280, 5)
- **Parameters:** ~5.3M

## Loading a Checkpoint

```python
import torch
from src.models.efficientnet import RetinaModel

model = RetinaModel(num_classes=5, pretrained=False, backbone='efficientnet_b0')
checkpoint = torch.load("models/efficientnet_b0_combined.pth", map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Best Kappa: {checkpoint.get('best_kappa', 'N/A')}")
print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
```

## Verifying Integrity

To generate and compare SHA-256 checksums for each checkpoint:

```bash
cd models/
shasum -a 256 *.pth
# Record the output alongside each release to verify downloads.
```

## Version Control Note

Model weights are large binary files (>20MB each) and are **not tracked by Git**.
For reproducibility, use [DVC](https://dvc.org) or store checksums here.
If you need the weights, contact the author or retrain using `scripts/train_improved.py`.
