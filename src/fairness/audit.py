"""
Fairness Audit Module for Diabetic Retinopathy Grading

Provides tools for:
- Pigmentation proxy estimation
- Per-group performance analysis
- Fairness metrics computation

Author: Deep Retina Grade Project
Date: January 2026
"""

import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, cohen_kappa_score


def estimate_pigmentation(img_path: str) -> float:
    """
    Estimate retinal pigmentation from fundus image.
    Uses mean luminance in LAB color space as proxy.

    Args:
        img_path: Path to fundus image

    Returns:
        Mean luminance value [0, 100]
    """
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    margin = int(min(h, w) * 0.1)
    img_cropped = img[margin:h-margin, margin:w-margin]

    lab = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    mask = L > 10

    if mask.sum() == 0:
        return np.nan

    return float(L[mask].mean())


def stratify_pigmentation(
    luminance_values: pd.Series,
    bins: int = 3
) -> pd.Series:
    """
    Stratify images by pigmentation level.

    Args:
        luminance_values: Series of luminance values
        bins: Number of bins (default: 3)

    Returns:
        Series of group labels
    """
    q33 = luminance_values.quantile(0.33)
    q66 = luminance_values.quantile(0.66)

    def assign_group(lum):
        if pd.isna(lum):
            return "Unknown"
        elif lum < q33:
            return "Dark"
        elif lum < q66:
            return "Medium"
        else:
            return "Light"

    return luminance_values.apply(assign_group)


def compute_group_metrics(
    df: pd.DataFrame,
    group_col: str,
    pred_col: str = "prediction",
    true_col: str = "diagnosis"
) -> pd.DataFrame:
    """
    Compute performance metrics for each group.

    Returns:
        DataFrame with per-group metrics
    """
    groups = df[group_col].unique()
    results = []

    for group in groups:
        if group == "Unknown":
            continue

        mask = df[group_col] == group
        y_true = df.loc[mask, true_col]
        y_pred = df.loc[mask, pred_col]

        acc = accuracy_score(y_true, y_pred)
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

        # Binary metrics for DR detection
        y_true_bin = (y_true >= 2).astype(int)
        y_pred_bin = (y_pred >= 2).astype(int)

        tp = ((y_true_bin == 1) & (y_pred_bin == 1)).sum()
        fn = ((y_true_bin == 1) & (y_pred_bin == 0)).sum()
        fp = ((y_true_bin == 0) & (y_pred_bin == 1)).sum()
        tn = ((y_true_bin == 0) & (y_pred_bin == 0)).sum()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        results.append({
            "Group": group,
            "N": mask.sum(),
            "Accuracy": acc,
            "QWK": qwk,
            "Sensitivity": sensitivity,
            "Specificity": specificity
        })

    return pd.DataFrame(results)


def compute_fairness_metrics(group_metrics: pd.DataFrame) -> Dict:
    """
    Compute fairness metrics from group performance.

    Returns:
        Dictionary of fairness metrics
    """
    metrics = {}

    # Demographic Parity
    acc_min = group_metrics["Accuracy"].min()
    acc_max = group_metrics["Accuracy"].max()
    metrics["demographic_parity_ratio"] = acc_min / acc_max if acc_max > 0 else 0
    metrics["accuracy_disparity"] = acc_max - acc_min

    # Equalized Odds
    sens_min = group_metrics["Sensitivity"].min()
    sens_max = group_metrics["Sensitivity"].max()
    metrics["equalized_odds_ratio"] = sens_min / sens_max if sens_max > 0 else 0
    metrics["sensitivity_disparity"] = sens_max - sens_min

    # 80% Rule
    metrics["passes_80_rule_accuracy"] = metrics["demographic_parity_ratio"] >= 0.8
    metrics["passes_80_rule_sensitivity"] = metrics["equalized_odds_ratio"] >= 0.8

    return metrics