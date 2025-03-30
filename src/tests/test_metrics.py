import pytest
import numpy as np
from src.evaluation.metrics import Metrics


def test_metrics_initialization():
    """Test Metrics class initialization."""
    metrics = Metrics("English", "test_model")
    assert metrics.language == "English"
    assert metrics.model_name == "test_model"


def test_create_metrics():
    """Test metrics calculation."""
    metrics = Metrics("English", "test_model")

    # Create sample predicted and validation groups
    predicted_groups = [[1, 5], [2, 3], [4]]
    validation_groups = [[1, 5], [2, 3], [4]]

    result = metrics.create_metrics(predicted_groups, validation_groups)

    # Perfect clustering should have perfect scores
    assert result["adjusted_rand_index"] == 1.0
    assert result["normalized_mutual_info"] == 1.0
    assert result["adjusted_mutual_info"] == 1.0
    assert result["v_measure"] == 1.0
    assert result["pairwise_precision"] == 1.0
    assert result["pairwise_recall"] == 1.0
    assert result["f1_score"] == 1.0


def test_imperfect_clustering():
    """Test metrics with imperfect clustering."""
    metrics = Metrics("English", "test_model")

    # Create sample predicted and validation groups with some differences
    predicted_groups = [[1, 5, 2], [3], [4]]
    validation_groups = [[1, 5], [2, 3], [4]]

    result = metrics.create_metrics(predicted_groups, validation_groups)

    # Scores should be less than 1.0
    assert result["adjusted_rand_index"] < 1.0
    assert result["pairwise_precision"] < 1.0
    assert result["pairwise_recall"] < 1.0
    assert result["f1_score"] < 1.0
