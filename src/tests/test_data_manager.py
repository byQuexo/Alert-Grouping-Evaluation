import pytest
import os
import pandas as pd
from src.data import DataManager


def test_data_manager_initialization(config):
    """Test that DataManager initializes correctly."""
    data_manager = DataManager(None, config)
    assert data_manager.data == {}
    assert data_manager.default_dir == "src/data"
    assert isinstance(data_manager.validation_data, pd.DataFrame)


def test_load_data(mock_data_manager):
    """Test that data is loaded correctly."""
    assert 'English' in mock_data_manager.data
    assert len(mock_data_manager.data['English']) == 5
    assert len(mock_data_manager.validation_data) == 3


def test_data_structure(mock_data_manager):
    """Test that data has the expected structure."""
    df = mock_data_manager.data['English']
    assert 'ID' in df.columns
    assert 'SUBJECT' in df.columns
    assert 'MESSAGE' in df.columns

    # Check validation data structure
    assert 'Group' in mock_data_manager.validation_data.columns
    assert 'IDs' in mock_data_manager.validation_data.columns
