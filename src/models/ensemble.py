"""
Ensemble Predictor - Combining Multiple Model Checkpoints

Ensembling multiple checkpoints improves accuracy and reduces variance
by averaging predictions from models trained with different hyperparameters.

Strategy:
- Arithmetic mean (simple averaging) for class probabilities
- Geometric mean for sharper ensemble predictions
- Weighted averaging based on validation performance

Author: Deep Retina Grade Project
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple model checkpoints.
    
    Supports:
    - Equal-weight averaging
    - Performance-weighted averaging (QWK-based)
    - Geometric mean for sharper predictions
    
    Args:
        model_class: Model class to instantiate
        model_paths: List of checkpoint paths
        device: Compute device
        weights: Optional weights per model (defaults to equal)
        method: Averaging method ('arithmetic', 'geometric', 'weighted')
    """
    
    def __init__(
        self,
        model_class: type,
        model_paths: List[str],
        device: torch.device,
        weights: Optional[List[float]] = None,
        method: str = 'arithmetic',
        **model_kwargs
    ):
        self.device = device
        self.method = method
        self.models = []
        self.model_names = []
        
        # Load all checkpoints
        for path in model_paths:
            path = Path(path)
            if not path.exists():
                logger.warning(f"Checkpoint not found: {path}")
                continue
                
            model = model_class(**model_kwargs)
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            if 'model_state_dict' not in checkpoint:
                logger.warning(f"Checkpoint missing 'model_state_dict' key: {path}")
                continue
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            self.models.append(model)
            self.model_names.append(path.stem)
            
            kappa = checkpoint.get('best_kappa', 'N/A')
            logger.info(f"Loaded {path.name} (QWK: {kappa})")
        
        if not self.models:
            raise ValueError("No valid model checkpoints found")
        
        # Set weights
        if weights is not None:
            if len(weights) != len(self.models):
                raise ValueError(
                    f"Expected {len(self.models)} weights, got {len(weights)}"
                )
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            # Equal weights
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        logger.info(
            f"Ensemble initialized with {len(self.models)} models, "
            f"method={method}, weights={[f'{w:.2f}' for w in self.weights]}"
        )
    
    def _compute_ensemble_probs(self, all_probs: np.ndarray) -> np.ndarray:
        """Combine per-model probabilities using the configured method."""
        if self.method == 'arithmetic' or self.method == 'weighted':
            return np.average(all_probs, axis=0, weights=self.weights)
        elif self.method == 'geometric':
            log_probs = np.log(np.clip(all_probs, 1e-10, 1.0))
            weighted_log = np.average(log_probs, axis=0, weights=self.weights)
            ensemble_probs = np.exp(weighted_log)
            return ensemble_probs / ensemble_probs.sum()
        else:
            raise ValueError(f"Unknown method: {self.method}")

    @torch.no_grad()
    def predict(self, input_tensor: torch.Tensor) -> Tuple[int, float, np.ndarray]:
        """
        Make ensemble prediction.
        
        Args:
            input_tensor: Preprocessed image tensor [1, 3, H, W]
            
        Returns:
            Tuple of (predicted_grade, confidence, probabilities)
        """
        input_tensor = input_tensor.to(self.device)
        all_probs = []
        
        for model in self.models:
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            all_probs.append(probs)
        
        all_probs = np.array(all_probs)  # [N_models, num_classes]
        
        # Combine predictions using configured method
        ensemble_probs = self._compute_ensemble_probs(all_probs)
        
        predicted_grade = int(np.argmax(ensemble_probs))
        confidence = float(ensemble_probs[predicted_grade])
        
        return predicted_grade, confidence, ensemble_probs
    
    @torch.no_grad()
    def predict_with_disagreement(
        self, input_tensor: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Predict with model disagreement analysis.
        
        Returns additional information about how much the models agree,
        which is useful for uncertainty estimation.
        
        Args:
            input_tensor: Preprocessed image tensor [1, 3, H, W]
            
        Returns:
            Dictionary with prediction, disagreement metrics, per-model outputs
        """
        input_tensor = input_tensor.to(self.device)
        all_probs = []
        all_grades = []
        
        for model in self.models:
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            grade = int(np.argmax(probs))
            all_probs.append(probs)
            all_grades.append(grade)
        
        all_probs = np.array(all_probs)
        
        # Ensemble prediction using configured method
        ensemble_probs = self._compute_ensemble_probs(all_probs)
        predicted_grade = int(np.argmax(ensemble_probs))
        confidence = float(ensemble_probs[predicted_grade])
        
        # Disagreement metrics
        grade_agreement = sum(1 for g in all_grades if g == predicted_grade) / len(all_grades)
        prob_std = float(np.std(all_probs[:, predicted_grade]))
        max_grade_diff = max(all_grades) - min(all_grades)
        
        return {
            'predicted_grade': predicted_grade,
            'confidence': confidence,
            'ensemble_probs': ensemble_probs,
            'per_model_grades': all_grades,
            'per_model_names': self.model_names,
            'grade_agreement': grade_agreement,
            'probability_std': prob_std,
            'max_grade_difference': max_grade_diff,
            'is_unanimous': len(set(all_grades)) == 1,
            'num_models': len(self.models)
        }
    
    @property
    def num_models(self) -> int:
        return len(self.models)
