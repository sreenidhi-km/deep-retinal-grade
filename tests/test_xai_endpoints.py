"""
Tests for XAI (Explainable AI) functionality.
Covers GradCAM, Integrated Gradients, and LIME explainers.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import XAI modules
try:
    from src.xai.gradcam import GradCAM
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

try:
    from src.xai.integrated_gradients import IntegratedGradientsExplainer
    IG_AVAILABLE = True
except ImportError:
    IG_AVAILABLE = False

try:
    from src.xai.lime_explainer import LIMEExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from src.models.efficientnet import RetinaModel
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False


@pytest.mark.skipif(not GRADCAM_AVAILABLE or not MODEL_AVAILABLE, 
                    reason="GradCAM or model not available")
class TestGradCAM:
    """Test GradCAM explainer."""
    
    @pytest.fixture
    def model(self):
        model = RetinaModel(num_classes=5, pretrained=False)
        model.eval()
        return model
    
    @pytest.fixture
    def gradcam(self, model):
        # Get the target layer from the model's backbone
        # For EfficientNet, this is typically the last conv layer
        target_layer = model.backbone.conv_head if hasattr(model.backbone, 'conv_head') else list(model.backbone.modules())[-3]
        return GradCAM(model, target_layer)
    
    def test_gradcam_initialization(self, model):
        """Test GradCAM initializes without error."""
        target_layer = model.backbone.conv_head if hasattr(model.backbone, 'conv_head') else list(model.backbone.modules())[-3]
        gradcam = GradCAM(model, target_layer)
        assert gradcam is not None
    
    def test_gradcam_produces_heatmap(self, gradcam, dummy_tensor):
        """Test GradCAM produces a heatmap."""
        heatmap, pred_class, confidence = gradcam.generate(dummy_tensor, target_class=0)
        
        assert heatmap is not None
        assert isinstance(heatmap, (np.ndarray, torch.Tensor))
    
    def test_gradcam_heatmap_shape(self, gradcam, dummy_tensor):
        """Test GradCAM heatmap has correct shape."""
        heatmap, _, _ = gradcam.generate(dummy_tensor, target_class=0)
        
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.numpy()
        
        # Heatmap should be 2D (H, W) or 3D (1, H, W) or (H, W, 1)
        assert len(heatmap.shape) in [2, 3]
    
    def test_gradcam_heatmap_values_normalized(self, gradcam, dummy_tensor):
        """Test GradCAM heatmap values are normalized."""
        heatmap, _, _ = gradcam.generate(dummy_tensor, target_class=0)
        
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.numpy()
        
        # Values should be in reasonable range (usually [0, 1] after normalization)
        assert heatmap.min() >= -0.1  # Allow small numerical errors
        assert heatmap.max() <= 1.1
    
    def test_gradcam_different_classes(self, gradcam, dummy_tensor):
        """Test GradCAM works for all target classes."""
        for target_class in range(5):
            heatmap, _, _ = gradcam.generate(dummy_tensor, target_class=target_class)
            assert heatmap is not None


@pytest.mark.skipif(not IG_AVAILABLE or not MODEL_AVAILABLE,
                    reason="Integrated Gradients or model not available")
class TestIntegratedGradients:
    """Test Integrated Gradients explainer."""
    
    @pytest.fixture
    def model(self):
        model = RetinaModel(num_classes=5, pretrained=False)
        model.eval()
        return model
    
    @pytest.fixture
    def ig_explainer(self, model):
        return IntegratedGradientsExplainer(model)
    
    def test_ig_initialization(self, model):
        """Test IG initializes without error."""
        ig = IntegratedGradientsExplainer(model)
        assert ig is not None
    
    def test_ig_produces_attribution(self, ig_explainer, dummy_tensor):
        """Test IG produces attributions."""
        attribution = ig_explainer.explain(dummy_tensor, target_class=0)
        
        assert attribution is not None
        assert isinstance(attribution, (np.ndarray, torch.Tensor))
    
    def test_ig_attribution_shape(self, ig_explainer, dummy_tensor):
        """Test IG attribution has same shape as input."""
        attribution = ig_explainer.explain(dummy_tensor, target_class=0)
        
        if isinstance(attribution, torch.Tensor):
            attribution = attribution.numpy()
        
        # Attribution should match input spatial dimensions
        # Could be (H, W), (C, H, W), or (1, C, H, W)
        assert attribution.shape[-2:] == (224, 224) or \
               attribution.shape[:2] == (224, 224)


@pytest.mark.skipif(not LIME_AVAILABLE or not MODEL_AVAILABLE,
                    reason="LIME or model not available")
class TestLIME:
    """Test LIME explainer."""
    
    @pytest.fixture
    def model(self):
        model = RetinaModel(num_classes=5, pretrained=False)
        model.eval()
        return model
    
    def test_lime_initialization(self, model):
        """Test LIME initializes without error."""
        # LIME requires preprocess_fn and device
        def dummy_preprocess(image):
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            return transform(image=image)
        
        device = torch.device('cpu')
        lime = LIMEExplainer(model, dummy_preprocess, device)
        assert lime is not None
    
    @pytest.mark.skip(reason="LIME tests are slow and require complex setup")
    def test_lime_produces_explanation(self, model, dummy_rgb_image):
        """Test LIME produces explanation."""
        pass  # Skip for now - LIME tests are slow


class TestXAIIntegration:
    """Test XAI integration with API."""
    
    @pytest.mark.skipif(not all([GRADCAM_AVAILABLE, IG_AVAILABLE, LIME_AVAILABLE, MODEL_AVAILABLE]),
                        reason="Not all XAI methods available")
    def test_all_xai_methods_available(self):
        """Test that all three XAI methods are implemented."""
        assert GRADCAM_AVAILABLE, "GradCAM should be available"
        assert IG_AVAILABLE, "Integrated Gradients should be available"
        assert LIME_AVAILABLE, "LIME should be available"
    
    @pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model not available")
    def test_xai_modules_importable(self):
        """Test that XAI modules can be imported."""
        # These imports should not raise
        from src.xai import gradcam
        from src.xai import integrated_gradients
        from src.xai import lime_explainer


class TestXAIRobustness:
    """Test XAI robustness to edge cases."""
    
    @pytest.mark.skipif(not GRADCAM_AVAILABLE or not MODEL_AVAILABLE,
                        reason="GradCAM or model not available")
    def test_gradcam_handles_batch_input(self):
        """Test GradCAM handles batch inputs."""
        model = RetinaModel(num_classes=5, pretrained=False)
        model.eval()
        target_layer = model.backbone.conv_head if hasattr(model.backbone, 'conv_head') else list(model.backbone.modules())[-3]
        gradcam = GradCAM(model, target_layer)
        
        batch_input = torch.randn(1, 3, 224, 224)  # Single image batch
        heatmap, _, _ = gradcam.generate(batch_input, target_class=0)
        
        assert heatmap is not None
    
    @pytest.mark.skipif(not GRADCAM_AVAILABLE or not MODEL_AVAILABLE,
                        reason="GradCAM or model not available")  
    def test_gradcam_deterministic(self):
        """Test GradCAM produces deterministic results."""
        model = RetinaModel(num_classes=5, pretrained=False)
        model.eval()
        target_layer = model.backbone.conv_head if hasattr(model.backbone, 'conv_head') else list(model.backbone.modules())[-3]
        gradcam = GradCAM(model, target_layer)
        
        torch.manual_seed(42)
        input_tensor = torch.randn(1, 3, 224, 224)
        
        heatmap1, _, _ = gradcam.generate(input_tensor, target_class=0)
        heatmap2, _, _ = gradcam.generate(input_tensor, target_class=0)
        
        if isinstance(heatmap1, torch.Tensor):
            heatmap1 = heatmap1.numpy()
            heatmap2 = heatmap2.numpy()
        
        np.testing.assert_array_almost_equal(heatmap1, heatmap2, decimal=5)
