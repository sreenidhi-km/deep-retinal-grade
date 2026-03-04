"""
FastAPI Backend for Diabetic Retinopathy Grading System

Endpoints:
- POST /predict: Grade a fundus image (with safety decision flags)
- POST /explain: Get XAI explanations
- GET /health: Health check
- GET /metrics: Model performance metrics

Author: Deep Retina Grade Project
Date: January 2026
"""

import os
import io
import json
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger('deep_retina_grade.api')

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.efficientnet import RetinaModel
from models.ensemble import EnsemblePredictor
from preprocessing.preprocess import RetinaPreprocessor
from preprocessing.quality import ImageQualityAssessor
from xai.gradcam import GradCAM, overlay_heatmap
from uncertainty.mc_dropout import MCDropoutPredictor
from reporting.pdf_report import ClinicalReportGenerator
from training.tta import TTAPredictor
from training.calibration import TemperatureScaler


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Environment variables for safety thresholds (configurable via .env)
DEMO_MODE = os.getenv("DR_DEMO_MODE", "false").lower() == "true"
QUALITY_THRESHOLD = float(os.getenv("DR_QUALITY_THRESHOLD", "0.4"))
UNCERTAINTY_THRESHOLD = float(os.getenv("DR_UNCERTAINTY_THRESHOLD", "0.15"))
OOD_ENTROPY_THRESHOLD = float(os.getenv("DR_OOD_ENTROPY_THRESHOLD", "1.5"))
MC_SAMPLES = int(os.getenv("DR_MC_SAMPLES", "5"))
CONFIDENCE_THRESHOLD = float(os.getenv("DR_CONFIDENCE_THRESHOLD", "0.6"))

# Rate limiting
RATE_LIMIT_MAX = int(os.getenv("DR_RATE_LIMIT_MAX", "60"))
RATE_LIMIT_WINDOW = int(os.getenv("DR_RATE_LIMIT_WINDOW", "60"))

# Ensemble
ENSEMBLE_ENABLED = os.getenv("DR_ENSEMBLE_ENABLED", "false").lower() == "true"

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================================
# Decision Flag Enum
# ============================================================================

class DecisionFlag(str, Enum):
    """
    Clinical decision flags for AI predictions.
    
    - OK: High confidence, good quality - safe to use as guidance
    - REVIEW: Moderate confidence or severe finding - clinician review required
    - RETAKE: Poor image quality - request new image
    - OOD: Out-of-distribution - image may not be fundus photo
    """
    OK = "OK"
    REVIEW = "REVIEW"
    RETAKE = "RETAKE"
    OOD = "OOD"

# DR Grade labels
DR_GRADES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# Clinical recommendations
RECOMMENDATIONS = {
    0: "No diabetic retinopathy detected. Continue annual screening.",
    1: "Mild NPDR detected. Re-screen in 9-12 months. Optimize diabetes control.",
    2: "Moderate NPDR detected. Refer to ophthalmologist within 3-6 months.",
    3: "Severe NPDR detected. Urgent referral to ophthalmologist within 2-4 weeks.",
    4: "Proliferative DR detected. IMMEDIATE referral to retina specialist."
}

# Decision flag explanations for API responses
DECISION_EXPLANATIONS = {
    DecisionFlag.OK: "Prediction confidence and image quality are acceptable for clinical guidance.",
    DecisionFlag.REVIEW: "Requires clinician review due to borderline confidence, high severity, or elevated uncertainty.",
    DecisionFlag.RETAKE: "Image quality is insufficient. Please retake the fundus photograph.",
    DecisionFlag.OOD: "Image may be out-of-distribution (not a typical fundus photo). Verify image source."
}


# ============================================================================
# Initialize FastAPI App
# ============================================================================

app = FastAPI(
    title="Deep Retina Grade API",
    description="AI-powered Diabetic Retinopathy Grading System with XAI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security, logging, and rate-limiting middleware
try:
    from app.middleware import (
        StructuredLoggingMiddleware,
        RateLimitMiddleware,
        SecurityHeadersMiddleware
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware, max_requests=RATE_LIMIT_MAX, window_seconds=RATE_LIMIT_WINDOW)
    app.add_middleware(StructuredLoggingMiddleware)
    logger.info("Middleware loaded: logging, rate-limiting, security headers")
except ImportError:
    if not DEMO_MODE:
        logger.error("Middleware import failed in production mode — aborting")
        raise
    logger.warning("Middleware not available, running without rate limiting and security headers")


# ============================================================================
# Pydantic Models
# ============================================================================

class PredictionResponse(BaseModel):
    """
    Enhanced prediction response with safety decision flags.
    
    The 'decision' field indicates whether the prediction should be:
    - OK: Used as clinical guidance
    - REVIEW: Reviewed by a clinician
    - RETAKE: Image retaken due to quality issues
    - OOD: Flagged as potentially out-of-distribution
    """
    grade: int
    grade_name: str
    confidence: float
    probabilities: Dict[str, float]
    recommendation: str
    referral_urgency: str
    # NEW: Safety contract fields
    decision: str  # DecisionFlag enum value
    decision_reason: str  # Human-readable explanation
    quality_score: float  # Image quality score [0, 1]
    quality_issues: List[str]  # List of detected quality issues
    uncertainty: Optional[float] = None  # MC Dropout uncertainty if computed
    entropy: Optional[float] = None  # Prediction entropy
    is_ood: bool = False  # Out-of-distribution flag


class ExplanationResponse(BaseModel):
    grade: int
    grade_name: str
    confidence: float
    gradcam_base64: str
    interpretation: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class MetricsResponse(BaseModel):
    overall_accuracy: float
    overall_qwk: float
    per_class_accuracy: Dict[str, float]
    fairness_metrics: Optional[Dict[str, Any]]


class UncertaintyResponse(BaseModel):
    predicted_grade: int
    grade_name: str
    confidence: float
    uncertainty: float
    entropy: float
    agreement: float
    is_borderline: bool
    grade_distribution: Dict[int, int]
    recommendation: str


class ReportResponse(BaseModel):
    success: bool
    report_path: str
    report_url: str
    grade: int
    confidence: float


# ============================================================================
# Load Model & Components
# ============================================================================

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() 
                     else 'mps' if torch.backends.mps.is_available() 
                     else 'cpu')

# Global variables for model and components
model = None
preprocessor = None
gradcam = None
transform = None
mc_predictor = None
report_generator = None
tta_predictor = None
quality_assessor = None
ensemble_predictor = None
temperature_scaler = None
MODEL_LOADED = False


def load_model():
    """Load model and initialize components."""
    global model, preprocessor, gradcam, transform, mc_predictor, report_generator, tta_predictor, quality_assessor, MODEL_LOADED
    
    # Load model - Using EfficientNet-B0 (optimized for M1 Mac)
    model = RetinaModel(num_classes=5, pretrained=False, backbone='efficientnet_b0')
    
    # Try improved model first, then fall back to original
    improved_path = MODELS_DIR / 'efficientnet_b0_improved.pth'
    original_path = MODELS_DIR / 'efficientnet_b0_best.pth'
    
    model_path = None
    if improved_path.exists():
        model_path = improved_path
    elif original_path.exists():
        model_path = original_path
    
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from {model_path}")
        print(f"   Best Kappa: {checkpoint.get('best_kappa', 'N/A')}")
        print(f"   Val Accuracy: {checkpoint.get('val_acc', 'N/A')}")
        MODEL_LOADED = True
    else:
        # CRITICAL SAFETY FIX: Do NOT use random weights for medical diagnosis!
        print(f"❌ CRITICAL: No model checkpoint found!")
        print(f"   Checked: {improved_path}")
        print(f"   Checked: {original_path}")
        print(f"   API will return 503 for all predictions.")
        MODEL_LOADED = False
        return  # Don't initialize other components
    
    model = model.to(device)
    model.eval()
    
    # Initialize preprocessor
    preprocessor = RetinaPreprocessor(img_size=224)
    
    # Initialize GradCAM
    target_layer = model.backbone.conv_head
    gradcam = GradCAM(model, target_layer)
    
    # Initialize transform
    transform = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    
    # Initialize MC Dropout predictor for uncertainty
    mc_predictor = MCDropoutPredictor(model, n_samples=MC_SAMPLES, uncertainty_threshold=UNCERTAINTY_THRESHOLD)
    
    # Initialize TTA predictor for improved accuracy
    tta_predictor = TTAPredictor(model, device, mode='full')
    print(f"   TTA enabled with {len(tta_predictor.transforms)} augmentations")
    
    # Initialize PDF report generator
    report_generator = ClinicalReportGenerator(output_dir=str(PROJECT_ROOT / "artifacts" / "reports"))
    
    # Initialize image quality assessor
    quality_assessor = ImageQualityAssessor()
    
    # Initialize ensemble predictor (if multiple checkpoints available)
    global ensemble_predictor
    ensemble_paths = [
        MODELS_DIR / 'efficientnet_b0_improved.pth',
        MODELS_DIR / 'efficientnet_b0_tuned.pth',
        MODELS_DIR / 'efficientnet_b0_combined.pth',
    ]
    available_paths = [p for p in ensemble_paths if p.exists()]
    if len(available_paths) >= 2 and ENSEMBLE_ENABLED:
        try:
            ensemble_predictor = EnsemblePredictor(
                model_class=RetinaModel,
                model_paths=[str(p) for p in available_paths],
                device=device,
                method='arithmetic',
                num_classes=5, pretrained=False, backbone='efficientnet_b0'
            )
            print(f"   Ensemble: {ensemble_predictor.num_models} models loaded")
        except Exception as e:
            print(f"   Ensemble: disabled ({e})")
    else:
        print(f"   Ensemble: {'disabled (set DR_ENSEMBLE_ENABLED=true)' if len(available_paths) >= 2 else 'not enough checkpoints'}")

    # Load temperature scaler if available
    global temperature_scaler
    scaler_path = MODELS_DIR / 'temperature_scaler.pth'
    if scaler_path.exists():
        try:
            temperature_scaler = TemperatureScaler.load(str(scaler_path), device=device)
            print(f"   Calibration: T={temperature_scaler.temperature.item():.3f}")
        except Exception as e:
            print(f"   Calibration: disabled ({e})")
    else:
        print(f"   Calibration: no scaler found (run calibration script to create)")

    # Log configuration
    logger.info(f"All components initialized on {device}")
    print(f"✅ All components initialized on {device}")
    print(f"   DEMO_MODE: {DEMO_MODE}")
    print(f"   Quality threshold: {QUALITY_THRESHOLD}")
    print(f"   Uncertainty threshold: {UNCERTAINTY_THRESHOLD}")
    print(f"   OOD entropy threshold: {OOD_ENTROPY_THRESHOLD}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


# ============================================================================
# Helper Functions
# ============================================================================

def compute_decision(
    grade: int,
    confidence: float,
    quality_score: float,
    quality_issues: List[str],
    uncertainty: Optional[float] = None,
    entropy: Optional[float] = None
) -> tuple[DecisionFlag, str, bool]:
    """
    Compute clinical decision flag based on prediction quality and confidence.
    
    Decision logic (priority order):
    1. RETAKE: Poor image quality (quality_score < QUALITY_THRESHOLD)
    2. OOD: High entropy suggesting out-of-distribution (entropy > OOD_ENTROPY_THRESHOLD)
    3. REVIEW: Low confidence, high uncertainty, or severe grade (3-4)
    4. OK: All checks passed
    
    Args:
        grade: Predicted DR grade (0-4)
        confidence: Model confidence [0, 1]
        quality_score: Image quality score [0, 1]
        quality_issues: List of quality issues detected
        uncertainty: MC Dropout uncertainty (optional)
        entropy: Prediction entropy (optional)
        
    Returns:
        Tuple of (DecisionFlag, reason_string, is_ood_flag)
    """
    reasons = []
    is_ood = False
    
    # 1. Check image quality first
    if quality_score < QUALITY_THRESHOLD:
        issues_str = ", ".join(quality_issues) if quality_issues else "low overall quality"
        return DecisionFlag.RETAKE, f"Poor image quality ({issues_str})", False
    
    # 2. Check for out-of-distribution (high entropy)
    if entropy is not None and entropy > OOD_ENTROPY_THRESHOLD:
        is_ood = True
        return DecisionFlag.OOD, f"High prediction entropy ({entropy:.2f}), image may not be typical fundus photo", True
    
    # 3. Check conditions that require clinical review
    needs_review = False
    
    # Low confidence
    if confidence < CONFIDENCE_THRESHOLD:
        needs_review = True
        reasons.append(f"low confidence ({confidence:.1%})")
    
    # High uncertainty (if available)
    if uncertainty is not None and uncertainty > UNCERTAINTY_THRESHOLD:
        needs_review = True
        reasons.append(f"high uncertainty ({uncertainty:.2f})")
    
    # Severe grades always need review for safety
    if grade >= 3:
        needs_review = True
        reasons.append(f"severe finding (Grade {grade}: {DR_GRADES[grade]})")
    
    if needs_review:
        reason_str = "; ".join(reasons)
        return DecisionFlag.REVIEW, f"Clinician review required: {reason_str}", False
    
    # 4. All checks passed
    return DecisionFlag.OK, DECISION_EXPLANATIONS[DecisionFlag.OK], False


def compute_entropy(probs: torch.Tensor) -> float:
    """
    Compute Shannon entropy from probability distribution.
    
    Higher entropy = more uncertain prediction.
    For 5-class: max entropy ≈ 1.61 (uniform distribution)
    
    Args:
        probs: Probability tensor [num_classes]
        
    Returns:
        Entropy value (float)
    """
    # Avoid log(0) by adding small epsilon
    probs_np = probs.cpu().numpy()
    probs_np = np.clip(probs_np, 1e-10, 1.0)
    entropy = -np.sum(probs_np * np.log(probs_np))
    return float(entropy)


def load_image_from_upload(file: UploadFile) -> np.ndarray:
    """Load image from uploaded file."""
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess_image(img: np.ndarray) -> torch.Tensor:
    """Preprocess image for model input."""
    # Apply Ben Graham + CLAHE preprocessing
    processed = preprocessor.preprocess_array(img)
    processed_uint8 = (processed * 255).astype(np.uint8)
    
    # Apply normalization and convert to tensor
    transformed = transform(image=processed_uint8)
    img_tensor = transformed['image'].unsqueeze(0).to(device)
    
    return img_tensor, processed


def numpy_to_base64(img: np.ndarray) -> str:
    """Convert numpy array to base64 string."""
    if img.max() <= 1:
        img = (img * 255).astype(np.uint8)
    
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def get_referral_urgency(grade: int) -> str:
    """Get referral urgency based on grade."""
    urgency_map = {
        0: "None",
        1: "Routine (9-12 months)",
        2: "Non-urgent (3-6 months)",
        3: "Urgent (2-4 weeks)",
        4: "Emergent (Immediate)"
    }
    return urgency_map.get(grade, "Unknown")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint."""
    return {
        "name": "Deep Retina Grade API",
        "version": "1.0.0",
        "description": "AI-powered Diabetic Retinopathy Grading System",
        "endpoints": {
            "POST /predict": "Grade a fundus image",
            "POST /explain": "Get XAI explanation",
            "GET /health": "Health check",
            "GET /metrics": "Model metrics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if MODEL_LOADED else "model_not_loaded",
        model_loaded=MODEL_LOADED,
        device=str(device)
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Grade a fundus image for diabetic retinopathy with safety decision flags.
    
    This endpoint performs:
    1. Image quality assessment (blur, brightness, contrast)
    2. DR grade prediction with confidence
    3. Uncertainty quantification (optional, via MC Dropout)
    4. Decision flag computation (OK, REVIEW, RETAKE, OOD)
    
    Args:
        file: Uploaded fundus image (PNG, JPG)
        
    Returns:
        PredictionResponse with grade, confidence, decision flag, and recommendations
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded. Cannot make predictions with random weights.")
    
    try:
        # Load image
        img = load_image_from_upload(file)
        
        # Step 1: Assess image quality BEFORE preprocessing
        quality_score, quality_issues = quality_assessor.assess_quality(img)
        
        # Step 2: Preprocess and predict
        img_tensor, _ = preprocess_image(img)
        
        with torch.no_grad():
            output = model(img_tensor)
            if temperature_scaler is not None:
                output = temperature_scaler(output)
            probs = F.softmax(output, dim=1)[0]
            grade = output.argmax(dim=1).item()
            confidence = probs[grade].item()
        
        # Step 3: Compute entropy from probability distribution
        entropy = compute_entropy(probs)
        
        # Step 4: Get uncertainty via MC Dropout (lightweight)
        uncertainty = None
        if mc_predictor is not None and not DEMO_MODE:
            try:
                unc_result = mc_predictor.predict_with_uncertainty(img_tensor)
                uncertainty = unc_result.get('uncertainty')
            except Exception:
                pass  # Skip uncertainty if it fails
        
        # Step 5: Compute decision flag
        decision, decision_reason, is_ood = compute_decision(
            grade=grade,
            confidence=confidence,
            quality_score=quality_score,
            quality_issues=quality_issues,
            uncertainty=uncertainty,
            entropy=entropy
        )
        
        # Build probabilities dict
        probabilities = {
            DR_GRADES[i]: float(probs[i]) for i in range(5)
        }
        
        return PredictionResponse(
            grade=grade,
            grade_name=DR_GRADES[grade],
            confidence=confidence,
            probabilities=probabilities,
            recommendation=RECOMMENDATIONS[grade],
            referral_urgency=get_referral_urgency(grade),
            # Safety contract fields
            decision=decision.value,
            decision_reason=decision_reason,
            quality_score=quality_score,
            quality_issues=quality_issues,
            uncertainty=uncertainty,
            entropy=entropy,
            is_ood=is_ood
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


class TTAPredictionResponse(BaseModel):
    """Response model for TTA prediction."""
    grade: int
    grade_name: str
    confidence: float
    tta_confidence: float  # Confidence from TTA averaging
    probabilities: Dict[str, float]
    recommendation: str
    referral_urgency: str
    tta_mode: str
    num_augmentations: int


@app.post("/predict-with-tta", response_model=TTAPredictionResponse)
async def predict_with_tta(file: UploadFile = File(...)):
    """
    Grade a fundus image with Test-Time Augmentation (TTA) for improved accuracy.
    
    TTA averages predictions across multiple augmented versions of the image,
    reducing variance and improving accuracy by 2-5%.
    
    Args:
        file: Uploaded fundus image (PNG, JPG)
        
    Returns:
        TTAPredictionResponse with averaged predictions and confidence
    """
    if not MODEL_LOADED or tta_predictor is None:
        raise HTTPException(status_code=503, detail="Model or TTA predictor not loaded")
    
    try:
        # Load image (use preprocessed format for TTA)
        img = load_image_from_upload(file)
        processed = preprocessor.preprocess_array(img)
        processed_uint8 = (processed * 255).astype(np.uint8)
        
        # Predict with TTA
        grade, tta_confidence, probs = tta_predictor.predict(processed_uint8)
        
        # Build probabilities dict
        probabilities = {
            DR_GRADES[i]: float(probs[i]) for i in range(5)
        }
        
        return TTAPredictionResponse(
            grade=grade,
            grade_name=DR_GRADES[grade],
            confidence=float(probs[grade]),
            tta_confidence=float(tta_confidence),
            probabilities=probabilities,
            recommendation=RECOMMENDATIONS[grade],
            referral_urgency=get_referral_urgency(grade),
            tta_mode=tta_predictor.mode,
            num_augmentations=len(tta_predictor.transforms)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTA prediction failed: {str(e)}")


@app.post("/explain", response_model=ExplanationResponse)
async def explain(file: UploadFile = File(...)):
    """
    Get XAI explanation (GradCAM) for prediction.
    
    Args:
        file: Uploaded fundus image
        
    Returns:
        ExplanationResponse with GradCAM heatmap overlay
    """
    if not MODEL_LOADED or gradcam is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Load and preprocess image
        img = load_image_from_upload(file)
        img_tensor, processed = preprocess_image(img)
        
        # Generate GradCAM
        heatmap, pred_class, confidence = gradcam.generate(img_tensor)
        
        # Create overlay
        overlay = overlay_heatmap(processed, heatmap, alpha=0.5)
        overlay_b64 = numpy_to_base64(overlay)
        
        # Generate interpretation
        interpretation = generate_interpretation(pred_class, confidence, heatmap)
        
        return ExplanationResponse(
            grade=pred_class,
            grade_name=DR_GRADES[pred_class],
            confidence=confidence,
            gradcam_base64=overlay_b64,
            interpretation=interpretation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


def generate_interpretation(grade: int, confidence: float, heatmap: np.ndarray) -> str:
    """Generate human-readable interpretation of GradCAM."""
    
    # Analyze heatmap distribution
    bright_regions = (heatmap > 0.7).sum() / heatmap.size
    
    base_interpretation = f"The model predicts {DR_GRADES[grade]} with {confidence:.1%} confidence. "
    
    if grade == 0:
        return base_interpretation + "No significant lesions detected. The heatmap shows the model examined the macula and optic disc regions."
    
    elif grade == 1:
        return base_interpretation + "The highlighted regions indicate subtle microaneurysms or early vascular changes. Monitor for progression."
    
    elif grade == 2:
        return base_interpretation + "The model identifies multiple areas of concern including possible hemorrhages and exudates. Professional evaluation recommended."
    
    elif grade == 3:
        return base_interpretation + "Significant vascular abnormalities detected. The intense highlighted areas suggest severe non-proliferative changes requiring urgent attention."
    
    else:  # grade == 4
        return base_interpretation + "Critical findings detected. The heatmap highlights areas suggestive of neovascularization or significant hemorrhaging. Immediate specialist referral required."


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get model performance metrics."""
    
    # Try to load metrics from results
    metrics_file = RESULTS_DIR / "test_metrics.json"
    fairness_file = RESULTS_DIR / "fairness_audit.json"
    
    metrics = {
        "overall_accuracy": 0.0,
        "overall_qwk": 0.0,
        "per_class_accuracy": {},
        "fairness_metrics": None
    }
    
    if metrics_file.exists():
        with open(metrics_file) as f:
            test_metrics = json.load(f)
            metrics["overall_accuracy"] = test_metrics.get("accuracy", 0.0)
            metrics["overall_qwk"] = test_metrics.get("qwk", 0.0)
            metrics["per_class_accuracy"] = test_metrics.get("per_class_accuracy", {})
    
    if fairness_file.exists():
        with open(fairness_file) as f:
            metrics["fairness_metrics"] = json.load(f)
    
    return MetricsResponse(**metrics)


@app.post("/predict-with-uncertainty", response_model=UncertaintyResponse)
async def predict_with_uncertainty(file: UploadFile = File(...)):
    """
    Grade a fundus image with uncertainty quantification (MC Dropout).
    
    Args:
        file: Uploaded fundus image
        
    Returns:
        UncertaintyResponse with grade, confidence, uncertainty metrics
    """
    if not MODEL_LOADED or mc_predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Load and preprocess image
        img = load_image_from_upload(file)
        img_tensor, _ = preprocess_image(img)
        
        # Predict with uncertainty
        result = mc_predictor.predict_with_uncertainty(img_tensor)
        
        return UncertaintyResponse(
            predicted_grade=result['predicted_grade'],
            grade_name=DR_GRADES[result['predicted_grade']],
            confidence=result['confidence'],
            uncertainty=result['uncertainty'],
            entropy=result['entropy'],
            agreement=result['agreement'],
            is_borderline=result['is_borderline'],
            grade_distribution=result['grade_distribution'],
            recommendation=result['recommendation']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/generate-report", response_model=ReportResponse)
async def generate_report(
    file: UploadFile = File(...),
    patient_id: str = "anonymous",
    patient_name: str = "N/A",
    include_xai: bool = True
):
    """
    Generate a comprehensive PDF clinical report.
    
    Args:
        file: Uploaded fundus image
        patient_id: Patient identifier
        patient_name: Patient name
        include_xai: Whether to include XAI heatmaps
        
    Returns:
        ReportResponse with path to generated PDF
    """
    if model is None or report_generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Load and preprocess image
        img = load_image_from_upload(file)
        img_tensor, processed = preprocess_image(img)
        
        # Get prediction
        with torch.no_grad():
            output = model(img_tensor)
            if temperature_scaler is not None:
                output = temperature_scaler(output)
            probs = F.softmax(output, dim=1)[0]
            grade = output.argmax(dim=1).item()
            confidence = probs[grade].item()
        
        # Build prediction dict
        prediction = {
            'grade': grade,
            'confidence': confidence,
            'probabilities': {DR_GRADES[i]: float(probs[i]) for i in range(5)}
        }
        
        # Get uncertainty
        uncertainty = None
        if mc_predictor:
            uncertainty = mc_predictor.predict_with_uncertainty(img_tensor)
        
        # Generate heatmaps if requested
        heatmaps = None
        if include_xai and gradcam:
            heatmap, _, _ = gradcam.generate(img_tensor)
            overlay = overlay_heatmap(processed, heatmap, alpha=0.5)
            heatmaps = {'gradcam': overlay}
        
        # Patient info
        patient_info = {
            'id': patient_id,
            'name': patient_name,
            'dob': 'N/A',
            'mrn': 'N/A'
        }
        
        # Generate report
        report_path = report_generator.generate_report(
            original_image=processed,
            prediction=prediction,
            heatmaps=heatmaps,
            uncertainty=uncertainty,
            patient_info=patient_info
        )
        
        return ReportResponse(
            success=True,
            report_path=report_path,
            report_url=f"/reports/{Path(report_path).name}",
            grade=grade,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


# ============================================================================
# Ensemble Endpoint
# ============================================================================

class EnsembleResponse(BaseModel):
    """Response model for ensemble prediction."""
    predicted_grade: int
    grade_name: str
    confidence: float
    probabilities: Dict[str, float]
    recommendation: str
    referral_urgency: str
    num_models: int
    grade_agreement: float
    per_model_grades: List[int]
    is_unanimous: bool
    decision: str = "OK"
    decision_reason: str = ""
    is_ood: bool = False
    quality_score: Optional[float] = None
    quality_issues: List[str] = []


@app.post("/predict-ensemble", response_model=EnsembleResponse)
async def predict_ensemble(file: UploadFile = File(...)):
    """
    Grade a fundus image using model ensemble (multiple checkpoints).
    
    Combines predictions from multiple trained models for improved
    accuracy and reliability estimation.
    
    Args:
        file: Uploaded fundus image
        
    Returns:
        EnsembleResponse with averaged predictions and agreement metrics
    """
    if ensemble_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Ensemble not available. Set DR_ENSEMBLE_ENABLED=true and ensure multiple checkpoints exist."
        )
    
    try:
        img = load_image_from_upload(file)
        img_tensor, _ = preprocess_image(img)
        
        # Quality assessment
        quality_score, quality_issues = quality_assessor.assess_quality(img)
        
        result = ensemble_predictor.predict_with_disagreement(img_tensor)
        
        probabilities = {
            DR_GRADES[i]: float(result['ensemble_probs'][i]) for i in range(5)
        }
        
        grade = result['predicted_grade']
        confidence = result['confidence']
        
        # Compute entropy from ensemble probabilities
        ensemble_probs_tensor = torch.tensor(result['ensemble_probs'])
        entropy = compute_entropy(ensemble_probs_tensor)
        
        # Safety decision using same logic as /predict
        decision, decision_reason, is_ood = compute_decision(
            grade=grade,
            confidence=confidence,
            entropy=entropy,
            quality_score=quality_score,
            quality_issues=quality_issues,
            uncertainty=None
        )
        
        return EnsembleResponse(
            predicted_grade=grade,
            grade_name=DR_GRADES[grade],
            confidence=confidence,
            probabilities=probabilities,
            recommendation=RECOMMENDATIONS[grade],
            referral_urgency=get_referral_urgency(grade),
            num_models=result['num_models'],
            grade_agreement=result['grade_agreement'],
            per_model_grades=result['per_model_grades'],
            is_unanimous=result['is_unanimous'],
            decision=decision,
            decision_reason=decision_reason,
            is_ood=is_ood,
            quality_score=quality_score,
            quality_issues=quality_issues
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ensemble prediction failed: {str(e)}")


# ============================================================================
# Run with uvicorn
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
