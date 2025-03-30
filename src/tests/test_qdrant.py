import pytest
from unittest.mock import patch, MagicMock
from src.clients import Qdrant

@pytest.fixture
def mock_qdrant_client():
    with patch('qdrant_client.QdrantClient') as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.collection_exists.return_value = False
        mock_instance.create_collection.return_value = None
        mock_instance.delete_collection.return_value = True
        mock_instance.query_points.return_value = MagicMock(points=[])
        yield mock_instance

def test_qdrant_initialization(config, mock_qdrant_client):
    """Test Qdrant client initialization."""
    with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
        qdrant = Qdrant(None, config)
        assert qdrant.client is not None

def test_create_collection(config, mock_qdrant_client):
    """Test collection creation."""
    with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
        qdrant = Qdrant(None, config)
        result = qdrant.create_collection("test_collection", 384)
        assert result is True
        mock_qdrant_client.create_collection.assert_called_once()

def test_check_collection(config, mock_qdrant_client):
    """Test checking if a collection exists."""
    with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
        qdrant = Qdrant(None, config)
        result = qdrant.check_collection("test_collection")
        assert result is False
        mock_qdrant_client.collection_exists.assert_called_once_with("test_collection")

def test_delete_collection(config, mock_qdrant_client):
    """Test deleting a collection."""
    with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
        qdrant = Qdrant(None, config)
        result = qdrant.delete_collection("test_collection")
        assert result is True
        mock_qdrant_client.delete_collection.assert_called_once_with("test_collection")
