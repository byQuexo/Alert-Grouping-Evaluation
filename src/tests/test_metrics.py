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


def test_completely_different_clustering():
    """Test metrics with completely different clustering."""
    metrics = Metrics("English", "test_model")

    predicted_groups = [[1, 2, 3], [4, 5]]
    validation_groups = [[1, 4], [2, 5], [3]]

    result = metrics.create_metrics(predicted_groups, validation_groups)

    # Scores should be low for completely different clustering
    assert result["adjusted_rand_index"] < 0.0
    assert result["f1_score"] < 0.5


def test_empty_groups():
    """Test metrics with empty groups."""
    metrics = Metrics("English", "test_model")

    predicted_groups = []
    validation_groups = [[1, 2], [3, 4]]

    result = metrics.create_metrics(predicted_groups, validation_groups)

    # Check that metrics handle empty groups properly
    assert result["predicted_groups"] == 0
    assert result["validation_groups"] == 2


def test_single_item_groups():
    """Test metrics with single item groups."""
    metrics = Metrics("English", "test_model")

    predicted_groups = [[1], [2], [3], [4], [5]]
    validation_groups = [[1], [2], [3], [4], [5]]

    result = metrics.create_metrics(predicted_groups, validation_groups)

    # Perfect clustering of single items should have perfect scores
    assert result["adjusted_rand_index"] == 1.0
    assert result["f1_score"] == 1.0


def test_different_number_of_groups():
    """Test metrics with different number of groups."""
    metrics = Metrics("English", "test_model")

    predicted_groups = [[1, 2, 3, 4, 5]]
    validation_groups = [[1, 2], [3, 4], [5]]

    result = metrics.create_metrics(predicted_groups, validation_groups)

    # Check that the number of groups is correctly reported
    assert result["predicted_groups"] == 1
    assert result["validation_groups"] == 3
    assert result["adjusted_rand_index"] < 1.0


def test_overlapping_groups():
    """Test metrics with overlapping groups."""
    metrics = Metrics("English", "test_model")

    # In real clustering, groups shouldn't overlap, but testing edge cases
    predicted_groups = [[1, 2, 3], [3, 4, 5]]
    validation_groups = [[1, 2, 3], [4, 5]]

    # This should use the last assignment for item 3
    result = metrics.create_metrics(predicted_groups, validation_groups)

    # Scores should be less than 1.0 due to the overlap
    assert result["adjusted_rand_index"] < 1.0


def test_different_item_sets():
    """Test metrics with different sets of items."""
    metrics = Metrics("English", "test_model")

    predicted_groups = [[1, 2, 3], [4, 5]]
    validation_groups = [[1, 2], [6, 7]]

    result = metrics.create_metrics(predicted_groups, validation_groups)

    # Check that metrics handle different item sets properly
    assert result["adjusted_rand_index"] < 1.0


def test_large_number_of_groups():
    """Test metrics with a large number of groups."""
    metrics = Metrics("English", "test_model")

    # Create 100 groups with one item each
    predicted_groups = [[i] for i in range(100)]
    validation_groups = [[i] for i in range(100)]

    result = metrics.create_metrics(predicted_groups, validation_groups)

    # Perfect clustering should have perfect scores even with many groups
    assert result["adjusted_rand_index"] == 1.0
    assert result["f1_score"] == 1.0


def test_pairwise_precision_recall_calculation():
    """Test the pairwise precision and recall calculation directly."""
    metrics = Metrics("English", "test_model")

    # Create a simple contingency matrix
    contingency = np.array([[5, 0], [0, 5]])

    precision, recall = metrics._pairwise_precision_recall(contingency)

    # For this perfect clustering, both should be 1.0
    assert precision == 1.0
    assert recall == 1.0

    # Create an imperfect contingency matrix
    contingency = np.array([[3, 2], [1, 4]])

    precision, recall = metrics._pairwise_precision_recall(contingency)

    # For this imperfect clustering, both should be less than 1.0
    assert precision < 1.0
    assert recall < 1.0


def test_f1_score_calculation():
    """Test the F1 score calculation directly."""
    metrics = Metrics("English", "test_model")

    # Test with perfect precision and recall
    f1 = metrics._calculate_f1_score(1.0, 1.0)
    assert f1 == 1.0

    # Test with zero precision and recall
    f1 = metrics._calculate_f1_score(0.0, 0.0)
    assert f1 == 0.0

    # Test with precision=0.8 and recall=0.6
    f1 = metrics._calculate_f1_score(0.8, 0.6)
    assert abs(f1 - 0.6857142857142857) < 1e-10
