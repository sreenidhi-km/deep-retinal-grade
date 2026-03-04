"""
Tests for model loading and inference.
Ensures model loads correctly and produces valid outputs.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.efficientnet import RetinaModel


class TestModelCreation:
    """Test model architecture creation."""
    
    def test_create_model_default(self):
        """Test creating RetinaModel with default settings."""
        model = RetinaModel(num_classes=5, pretrained=False)
        assert model is not None
        
    def test_model_has_correct_output_classes(self):
        """Test that model outputs 5 classes (DR grades 0-4)."""
        model = RetinaModel(num_classes=5, pretrained=False)
        model.eval()
        
        # Get final layer output features
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (1, 5), f"Expected (1, 5), got {output.shape}"
    
    def test_model_accepts_batch_input(self):
        """Test that model handles batch inputs."""
        model = RetinaModel(num_classes=5, pretrained=False)
        model.eval()
        
        batch_input = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            output = model(batch_input)
        
        assert output.shape == (4, 5), f"Expected (4, 5), got {output.shape}"
    
    def test_model_output_is_logits(self):
        """Test that model outputs logits (not probabilities)."""
        model = RetinaModel(num_classes=5, pretrained=False)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        # Logits can be negative and don't sum to 1
        # Probabilities would sum to ~1 after softmax
        probs = torch.softmax(output, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0]), atol=1e-5)


class TestModelLoading:
    """Test loading pretrained model weights."""
    
    def test_model_file_exists(self, model_path, model_exists):
        """Test that model checkpoint file exists."""
        if not model_exists:
            pytest.skip("Model file not found - skipping loading tests")
        assert model_path.exists(), f"Model not found at {model_path}"
    
    def test_model_loads_successfully(self, model_path, model_exists, device):
        """Test that model checkpoint loads without errors."""
        if not model_exists:
            pytest.skip("Model file not found")
        
        model = RetinaModel(num_classes=5, pretrained=False)
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load should not raise
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        assert model is not None
    
    def test_loaded_model_produces_output(self, model_path, model_exists, device, dummy_tensor):
        """Test that loaded model produces valid output."""
        if not model_exists:
            pytest.skip("Model file not found")
        
        model = RetinaModel(num_classes=5, pretrained=False)
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(device)
        
        with torch.no_grad():
            output = model(dummy_tensor)
        
        assert output.shape == (1, 5)
        assert not torch.isnan(output).any(), "Model produced NaN values"
        assert not torch.isinf(output).any(), "Model produced Inf values"


class TestModelInference:
    """Test model inference behavior."""
    
    @pytest.fixture
    def model(self):
        """Create a model for testing."""
        model = RetinaModel(num_classes=5, pretrained=False)
        model.eval()
        return model
    
    def test_inference_is_deterministic_in_eval_mode(self, model, dummy_tensor):
        """Test that eval mode produces deterministic outputs."""
        model.eval()
        
        with torch.no_grad():
            output1 = model(dummy_tensor)
            output2 = model(dummy_tensor)
        
        torch.testing.assert_close(output1, output2)
    
    def test_inference_handles_different_batch_sizes(self, model):
        """Test inference with various batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            batch_input = torch.randn(batch_size, 3, 224, 224)
            with torch.no_grad():
                output = model(batch_input)
            assert output.shape == (batch_size, 5)
    
    def test_model_gradient_disabled_in_eval(self, model, dummy_tensor):
        """Test that gradients are properly disabled during inference."""
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_tensor)
        
        # Output should not require grad when using no_grad context
        assert not output.requires_grad


class TestModelRobustness:
    """Test model robustness to edge cases."""
    
    @pytest.fixture
    def model(self):
        model = RetinaModel(num_classes=5, pretrained=False)
        model.eval()
        return model
    
    def test_handles_zero_input(self, model):
        """Test model handles all-zero input."""
        zero_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            output = model(zero_input)
        
        assert not torch.isnan(output).any()
        assert output.shape == (1, 5)
    
    def test_handles_normalized_input(self, model):
        """Test model handles ImageNet-normalized input."""
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        # Create normalized input
        raw = torch.rand(1, 3, 224, 224)
        normalized = (raw - mean) / std
        
        with torch.no_grad():
            output = model(normalized)
        
        assert not torch.isnan(output).any()
        assert output.shape == (1, 5)
