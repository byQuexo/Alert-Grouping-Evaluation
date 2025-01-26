from loguru import logger
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, v_measure_score
from sklearn.metrics.cluster import contingency_matrix
import numpy as np

class Metrics:
    def __init__(self, language, model_name):
        self.model_name = model_name
        self.language = language

    def create_metrics(self, predicted_groups, validation_groups):
        """
        Create metrics to evaluate the model performance for clustering.

        :param predicted_groups: List of lists, where each inner list represents a group of item IDs
        :param validation_groups: List of lists, where each inner list represents a true group of item IDs
        :return: Dictionary of evaluation metrics
        """
        logger.debug(f"Predicted groups: {predicted_groups}, validation groups: {validation_groups}")

        # Create mappings of items to their group labels
        pred_mapping = self._create_item_to_group_mapping(predicted_groups)
        val_mapping = self._create_item_to_group_mapping(validation_groups)

        # Get all unique items
        all_items = set(pred_mapping.keys()).union(set(val_mapping.keys()))

        # Create label lists
        pred_labels = [pred_mapping.get(item, -1) for item in all_items]
        val_labels = [val_mapping.get(item, -1) for item in all_items]

        # Calculate metrics
        ari = adjusted_rand_score(val_labels, pred_labels)
        nmi = normalized_mutual_info_score(val_labels, pred_labels)
        ami = adjusted_mutual_info_score(val_labels, pred_labels)
        v_measure = v_measure_score(val_labels, pred_labels)

        # Calculate pairwise precision and recall
        cont_matrix = contingency_matrix(val_labels, pred_labels)
        pair_precision, pair_recall = self._pairwise_precision_recall(cont_matrix)

        # Calculate F1 score
        f1_score = self._calculate_f1_score(pair_precision, pair_recall)

        return {
            "model_name": self.model_name,
            "language": self.language,
            "predicted_groups": len(predicted_groups),
            "validation_groups": len(validation_groups),
            "adjusted_rand_index": ari,
            "normalized_mutual_info": nmi,
            "adjusted_mutual_info": ami,
            "v_measure": v_measure,
            "pairwise_precision": pair_precision,
            "pairwise_recall": pair_recall,
            "f1_score": f1_score
        }

    def _create_item_to_group_mapping(self, groups):
        """
        Create a mapping of items to their group labels.

        :param groups: List of lists, where each inner list represents a group of item IDs
        :return: Dictionary mapping each item to its group label
        """
        mapping = {}
        for i, group in enumerate(groups):
            for item in group:
                mapping[item] = i
        return mapping

    def _pairwise_precision_recall(self, contingency):
        """
        Calculate pairwise precision and recall from a contingency matrix.

        :param contingency: Contingency matrix
        :return: Tuple of (precision, recall)
        """
        true_positives = np.sum(contingency * (contingency - 1)) / 2
        false_positives = np.sum(np.sum(contingency, axis=0) * (np.sum(contingency, axis=0) - 1)) / 2 - true_positives
        false_negatives = np.sum(np.sum(contingency, axis=1) * (np.sum(contingency, axis=1) - 1)) / 2 - true_positives

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        return precision, recall

    def _calculate_f1_score(self, precision, recall):
        """
        Calculate F1 score from precision and recall.

        :param precision: Precision value
        :param recall: Recall value
        :return: F1 score
        """
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0



    def _create_overall_metrics(self, model, metrics):
        """
        Create overall metrics by averaging the metrics over all languages.

        :param metrics: List of metrics dictionaries
        :return: Dictionary of overall metrics
        """

        overall_metrics = {
            "model_name": model,
            "adjusted_rand_index": np.mean([m["adjusted_rand_index"] for m in metrics]),
            "normalized_mutual_info": np.mean([m["normalized_mutual_info"] for m in metrics]),
            "adjusted_mutual_info": np.mean([m["adjusted_mutual_info"] for m in metrics]),
            "v_measure": np.mean([m["v_measure"] for m in metrics]),
            "pairwise_precision": np.mean([m["pairwise_precision"] for m in metrics]),
            "pairwise_recall": np.mean([m["pairwise_recall"] for m in metrics]),
            "f1_score": np.mean([m["f1_score"] for m in metrics])
        }

        return overall_metrics
