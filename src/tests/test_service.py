import pytest
from unittest.mock import patch, MagicMock
from src.service import Service


@pytest.fixture
def mock_integrated_service(mock_data_manager, mock_model_manager):
    with patch('src.clients.Qdrant') as mock_qdrant:
        mock_qdrant_instance = mock_qdrant.return_value
        mock_qdrant_instance.check_collection.return_value = False
        mock_qdrant_instance.create_collection.return_value = True
        mock_qdrant_instance.query_points.return_value = []
        mock_qdrant_instance.insert_point.return_value = "OK"

        service = Service()
        service.data_manager = mock_data_manager
        service.model_manager = mock_model_manager
        service.qdrant = mock_qdrant_instance
        service.groups = {}
        yield service


def test_service_initialization():
    """Test Service initialization."""
    service = Service()
    assert hasattr(service, 'config')
    assert hasattr(service, 'qdrant')
    assert hasattr(service, 'model_manager')
    assert hasattr(service, 'data_manager')
    assert hasattr(service, 'groups')
    assert hasattr(service, 'metrics')


def test_create_new_group(mock_integrated_service):
    """Test creating a new alert group."""
    mock_integrated_service._create_new_group(1, "mock_model", "English")
    assert "mock_model" in mock_integrated_service.groups
    assert "English" in mock_integrated_service.groups["mock_model"]
    assert "Group1" in mock_integrated_service.groups["mock_model"]["English"]
    assert mock_integrated_service.groups["mock_model"]["English"]["Group1"] == [1]


def test_handle_similarity(mock_integrated_service):
    """Test handling similarity between alerts."""
    # First create a group
    mock_integrated_service._create_new_group(1, "mock_model", "English")

    # Create a mock similarity point
    class MockPoint:
        def __init__(self, id, score):
            self.id = id
            self.score = score

    similarity = [MockPoint(1, 0.95)]

    # Handle a new alert with similarity to an existing one
    mock_integrated_service._handle_similarity(2, similarity, "mock_model", "English")

    # The new alert should be added to the existing group
    assert mock_integrated_service.groups["mock_model"]["English"]["Group1"] == [1, 2]


def test_merge_groups(mock_integrated_service):
    """Test merging multiple groups."""
    # Create two groups
    mock_integrated_service._create_new_group(1, "mock_model", "English")
    mock_integrated_service._create_new_group(2, "mock_model", "English")

    # Merge the groups
    target_groups = ["Group1", "Group2"]
    mock_integrated_service._merge_groups(target_groups, "mock_model", "English")

    # Check that groups were merged
    assert "Group2" not in mock_integrated_service.groups["mock_model"]["English"]
    assert mock_integrated_service.groups["mock_model"]["English"]["Group1"] == [1, 2]
