import pytest
from src.models.model_manager import ModelManager

def test_model_manager_initialization(config):
    """Test that ModelManager initializes correctly."""
    model_manager = ModelManager(None, config)
    assert hasattr(model_manager, 'models')
    assert model_manager.models == {}

def test_create_embedding(mock_model_manager):
    """Test embedding creation."""
    embedding = mock_model_manager.create_embedding("mock_model", "Test message")
    assert len(embedding) == 768  # Should match the mock model's dimension
    assert all(x == 0.5 for x in embedding)  # Our mock returns all 0.5s
