import os
import sys
import pytest
import pandas as pd
from collections import defaultdict

from src import Service
from src.config import App
from src.data import DataManager
from src.models.model_manager import ModelManager

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



@pytest.fixture
def config():
    return App.config()


@pytest.fixture
def mock_service():
    return Service()


@pytest.fixture
def sample_data():

    # Create sample data

    data = {
        'ID': [1, 2, 3, 4, 5],
        'SUBJECT': [
            'Unavailable by ICMP ping: PROBLEM for 10.116.156.108',
            'VPN Tunnel Down [ADM 73 - 10.122.96.0/17]: PROBLEM for FW-DC',
            'Health Check State Failed to FW_NA on member 2: PROBLEM for FW-DC',
            'MSSQL: Service is unavailable: PROBLEM for RES-PA-W01-002_SQL-PA-W01-002',
            'Unavailable by ICMP ping: PROBLEM for FW-WL'
        ],
        'MESSAGE': [
            '[PROBLEM: Disaster] Host: 10.116.156.108 Description: Last three attempts returned timeout.',
            '[PROBLEM: High] Host: FW-DC Description: IPsec VPN Tunnel Phase 2 is down, please check!',
            '[PROBLEM: High] Host: FW-DC Description: Problem started at 01:09:51 on 2024.02.19',
            '[PROBLEM: Disaster] Host: RES-PA-W01-002_SQL-PA-W01-002 Description: The TCP port of the MS SQL Server service is currently unavailable.',
            '[PROBLEM: High] Host: FW-WL Description: Last three attempts returned timeout.'
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_validation_data():
    # Create sample validation data
    data = {
        'Group': ['Group1', 'Group2', 'Group3'],
        'IDs': ['1,5', '2,3', '4']
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_data_manager(config, sample_data, sample_validation_data):
    data_manager = DataManager(None, config)
    data_manager.data = {'English': sample_data}
    data_manager.validation_data = sample_validation_data
    return data_manager


@pytest.fixture
def mock_embedding_model():
    # TODO: Use a real embedding model for testing

    class MockModel:
        def __init__(self):
            self.name = "mock_model"
            self.dim = 768

        def create_embedding(self, text, **kwargs):
            # Return a fixed-size vector for testing
            import numpy as np
            return np.ones(self.dim) * 0.5

    return MockModel()


@pytest.fixture
def mock_model_manager(config, mock_embedding_model):
    model_manager = ModelManager(None, config)
    model_manager.models = {"mock_model": mock_embedding_model}
    return model_manager
