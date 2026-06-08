import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Dict, Any

from .base_evaluator import BaseAnomalyEvaluator
from .registry import AnomalyEvaluationRegistry

# @AnomalyEvaluationRegistry.register("sir")
class MpoxMultivariateNoveltyEvaluator(BaseAnomalyEvaluator):
    def evaluate(
        self,
        true_anomalies: Dict[str, Dict[str, Any]],
        detected_anomalies: Dict[str, Dict[str, np.ndarray]],
        modified_data: pd.DataFrame,
        verbose: bool = True
    ) -> dict:
        """
        Evaluates multivariate novelty anomaly detection performance for the mpox scenario.

        Args:
            true_anomalies (Dict[str, Dict[str, Any]]): True anomalies for each symptom.
            detected_anomalies (Dict[str, Dict[str, np.ndarray]]): Detected anomaly scores and indices.
            modified_data (pd.DataFrame): The dataset with injected anomalies.
            verbose (bool): If True, prints the evaluation metrics.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        # Find the overall anomaly range across all symptoms
        min_start_index = float('inf')
        max_end_index = -float('inf')

        for column, anomaly_types in true_anomalies.items():
            for anomaly_type, anomaly_data in anomaly_types.items():
                if anomaly_type in ['sir_mpox', 'sir_measles']:
                    indices = anomaly_data['indices']
                    min_start_index = min(min_start_index, indices.min())
                    max_end_index = max(max_end_index, indices.max())

        # Create true labelsß
        true_labels = np.zeros(len(modified_data), dtype=int)
        true_labels[min_start_index:max_end_index+1] = 1

        pred_labels = np.zeros(len(modified_data), dtype=int)
        anomaly_indices = detected_anomalies['multivariate']['anomaly_indices']
        pred_labels[anomaly_indices] = 1

        # Detect change point using the predicted labels
        pred_change_point = self.find_robust_change_point(pred_labels)

        if pred_change_point is not None:
            # ttd = max(0, pred_change_point - min_start_index)
            ttd = (pred_change_point - min_start_index)
        else:
            ttd = None

        # Calculate false alarms before the true anomaly starts
        false_alarms = np.sum(pred_labels[:min_start_index])
        false_alarm_rate = false_alarms / min_start_index if min_start_index > 0 else 0

        # Calculate evaluation metrics
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)

        results = {
            'time_to_detect': ttd,
            'false_alarm_rate': false_alarm_rate,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        if verbose:
            print("Mpox Multivariate Novelty Anomaly Detection Evaluation Metrics:")
            for key, value in results.items():
                if value is not None:
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")

        return results

    @staticmethod
    def find_robust_change_point(y: np.ndarray, window_size: int = 30, threshold: float = 0.01) -> int:
        """
        Finds the index of the change point based on an increase in frequency of detected anomalies.

        Args:
            y (np.ndarray): Binary labels (0 for normal, 1 for anomaly).
            window_size (int): Size of the sliding window for calculating frequency.
            threshold (float): Threshold for considering a significant increase in frequency.

        Returns:
            int: Index of the change point, or None if not found.
        """
        if len(y) < window_size:
            return None

        anomaly_freq = np.convolve(y, np.ones(window_size), 'valid') / window_size
        freq_diff = np.diff(anomaly_freq)
        change_points = np.where(freq_diff > threshold)[0]

        if len(change_points) > 0:
            return change_points[0] + window_size
        else:
            return None

    def _find_change_point(self, y: np.ndarray) -> int:
        """
        Finds the index of the change point.

        Args:
            y (np.ndarray): Binary labels.

        Returns:
            int: Index of the change point, or None if not found.
        """
        change_points = np.where(np.diff(y) == 1)[0] + 1  # +1 to correct the index after diff
        if len(change_points) > 0:
            return change_points[0]
        else:
            return None