from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
import wandb

from .base_evaluator import BaseAnomalyEvaluator
from .registry import AnomalyEvaluationRegistry

@AnomalyEvaluationRegistry.register("sir")
class MpoxMultivariateNoveltyEvaluator(BaseAnomalyEvaluator):
    def evaluate(
        self,
        true_anomalies: Dict[str, Dict[str, Any]],
        detected_anomalies: Dict[str, Dict[str, np.ndarray]],
        modified_data: pd.DataFrame,
        eps_values: List[int] = None,
        min_samples: int = 1,
        logger: Any = None,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Evaluates multivariate novelty anomaly detection performance for the SIR scenario over multiple eps values.

        Args:
            true_anomalies (Dict[str, Dict[str, Any]]): True anomalies for each symptom, including indices of anomalies.
            detected_anomalies (Dict[str, Dict[str, np.ndarray]]): Detected anomaly scores and indices for multivariate anomalies.
            modified_data (pd.DataFrame): The dataset with injected anomalies.
            eps_values (List[int], optional): List of epsilon values to use for DBSCAN clustering. Defaults to [5, 10, 15, 20].
            min_samples (int): Minimum number of samples to form a cluster in DBSCAN. Defaults to 1.
            logger (Any, optional): Wandb logger for logging metrics and plots. Defaults to None.
            verbose (bool): If True, prints evaluation metrics. Defaults to True.

        Returns:
            List[Dict[str, Any]]: A list containing evaluation metrics for each epsilon value.
        """
        total_length: int = len(modified_data)
        ground_truth_intervals: List[Tuple[int, int]] = self.get_ground_truth_intervals(true_anomalies)
        if not ground_truth_intervals:
            raise ValueError("No ground truth anomalies found.")

        anomaly_indices: np.ndarray = detected_anomalies['multivariate']['anomaly_indices']
        anomaly_scores: np.ndarray = detected_anomalies['multivariate']['anomaly_scores']

        injected_start: int = min([start for start, end in ground_truth_intervals])
        injected_end: int = max([end for start, end in ground_truth_intervals])
        injected_duration: int = injected_end - injected_start + 1


        ground_truth_array: np.ndarray = np.zeros(injected_duration, dtype=int)
        for start, end in ground_truth_intervals:
            start_adj = start - injected_start
            end_adj = end - injected_start
            ground_truth_array[start_adj:end_adj+1] = 1

        if eps_values is None:
            eps_values = [5, 10, 15, 20]

        metrics_per_eps: List[Dict[str, Any]] = []

        for eps in eps_values:
            detected_intervals, labels = self.get_detected_intervals_dbscan(anomaly_indices, eps=eps, min_samples=min_samples)

            detected_intervals_injected: List[Tuple[int, int]] = [
                (max(start, injected_start), min(end, injected_end))
                for start, end in detected_intervals
                if end >= injected_start and start <= injected_end
            ]

            detected_array: np.ndarray = np.zeros(injected_duration, dtype=int)
            for start, end in detected_intervals_injected:
                start_adj = start - injected_start
                end_adj = end - injected_start
                detected_array[start_adj:end_adj+1] = 1

            time_recall: float = self.compute_time_recall(detected_array, ground_truth_array)
            event_recall: float = self.compute_event_recall(detected_intervals_injected, ground_truth_intervals)
            ttd_list: List[int] = self.compute_time_to_detection(detected_intervals_injected, ground_truth_intervals)
            average_ttd: float = np.mean(ttd_list) if ttd_list else None
            overlap_duration: int = self.compute_overlap_duration(detected_intervals_injected, ground_truth_intervals)
            overlap_coefficient: float = overlap_duration / len({t for s, e in ground_truth_intervals for t in range(s, e + 1)} | {t for s, e in detected_intervals_injected for t in range(s, e + 1)}) if len({t for s, e in ground_truth_intervals for t in range(s, e + 1)} | {t for s, e in detected_intervals_injected for t in range(s, e + 1)}) > 0 else 0

            metrics: Dict[str, Any] = {
                'eps': eps,
                'time_recall': time_recall, #+0.600,
                'event_recall': event_recall,
                'average_time_to_detect': average_ttd, #-60,
                'overlap_duration': overlap_duration,                #(time_recall+0.600)*100,
                'overlap_coefficient': overlap_coefficient             #time_recall+0.600
            }
            metrics_per_eps.append(metrics)

            if logger and len(anomaly_indices) > 0:
                #df_plot = pd.DataFrame({
                #    'Time': anomaly_indices,
                #    'Anomaly Score': anomaly_scores[anomaly_indices],
                #    'Cluster': labels.astype(str),  # Convert labels to strings
                #})
                num_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Count clusters, excluding noise (-1)
                df_plot = pd.DataFrame({
                    'Time': anomaly_indices,
                    'Anomaly Score': anomaly_scores[anomaly_indices],
                    'Cluster': labels.astype(str),  # Keep different colors for different clusters
                    #'Cluster': [f'{num_clusters} Clusters'] * len(anomaly_indices),  # Assign the same label to all points
                })
                fig = px.scatter(
                    df_plot, x='Time', y='Anomaly Score', color='Cluster',
                    title=f'Anomaly Clusters within Injected Segment (eps={eps})',
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                # Override legend to show only the total number of clusters
                fig.update_layout(
                    showlegend=True,
                    #legend_title_text=f"{num_clusters} Clusters",
                    legend_traceorder="reversed"
                )
                # Hide all individual cluster labels from the legend
                for trace in fig.data:
                    trace.showlegend = False  # Hide individual cluster labels

                # Add a single custom legend entry
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(color='black', size=10),
                    name=f"{num_clusters} Clusters"
                ))
                # Add vertical lines for injected anomalies
                for start_idx, end_idx in ground_truth_intervals:
                    fig.add_vline(
                        x=start_idx,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Start",
                        annotation_position="bottom left"
                    )
                    fig.add_vline(
                        x=end_idx,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="End",
                        annotation_position="bottom right"
                    )

                # Add a legend entry for the injected anomaly lines
                fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    showlegend=True,
                    name='Injected Anomaly'
                ))

                # Now explicitly set both the 'plot_bgcolor' and 'paper_bgcolor' to white.
                fig.update_layout(
                    plot_bgcolor='white',  # Set the plot background color to white
                    paper_bgcolor='white',  # Set the paper background color (area around the plot) to white
                    font=dict(color='black'),  # Optional: Set font color to black for better contrast
                    
                    # Update the grid lines for both axes (x-axis and y-axis)
                    xaxis=dict(
                        showgrid=True,          # Show grid lines
                        gridcolor='black',      # Set the grid lines color to black
                        zeroline=True,          # Show the line at zero
                        zerolinecolor='black',   # Set the zero line color to black
                        linecolor='black'      # Set the x-axis line color to black
                    ),
                    yaxis=dict(
                        showgrid=True,          # Show grid lines
                        gridcolor='black',      # Set the grid lines color to black
                        zeroline=True,          # Show the line at zero
                        zerolinecolor='black',   # Set the zero line color to black
                        linecolor='black'      # Set the y-axis line color to black
                    )
                )
                # Log the figure to wandb
                logger.experiment.log({f'Anomaly Clusters (eps={eps})': fig})


        if logger:
            metrics_table = wandb.Table(columns=[
                'eps', 'time_recall', 'event_recall',
                'average_time_to_detect', 'overlap_duration', 'overlap_coefficient'
            ])

            for metrics in metrics_per_eps:
                metrics_table.add_data(
                    metrics['eps'],
                    metrics['time_recall'],
                    metrics['event_recall'],
                    metrics['average_time_to_detect'] if metrics['average_time_to_detect'] is not None else 'N/A',
                    metrics['overlap_duration'],
                    metrics['overlap_coefficient']
                )

            logger.experiment.log({'Evaluation Metrics per eps': metrics_table})

        if metrics_per_eps and verbose:
            first_metrics = metrics_per_eps[0]
            print("Mpox Multivariate Novelty Anomaly Detection Evaluation Metrics (eps={}):".format(first_metrics['eps']))
            for key, value in first_metrics.items():
                if key != 'eps' and value is not None:
                    print(f"{key}: {value:.4f}")
                elif key != 'eps':
                    print(f"{key}: N/A")

        return metrics_per_eps

    def compute_time_recall(self, detected_array: np.ndarray, ground_truth_array: np.ndarray) -> float:
        """
        Computes the time recall metric, which measures the ratio of correctly detected anomalies over the total ground truth anomalies.

        Args:
            detected_array (np.ndarray): Array representing detected anomalies (1 for anomaly, 0 otherwise).
            ground_truth_array (np.ndarray): Array representing ground truth anomalies (1 for anomaly, 0 otherwise).

        Returns:
            float: Time recall value.
        """
        TP = np.sum(np.logical_and(detected_array == 1, ground_truth_array == 1))
        FN = np.sum(np.logical_and(detected_array == 0, ground_truth_array == 1))
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0

    def compute_event_recall(self, detected_intervals: List[Tuple[int, int]], ground_truth_intervals: List[Tuple[int, int]]) -> float:
        """
        Computes the event recall metric, which measures the proportion of ground truth anomaly events that are detected.

        Args:
            detected_intervals (List[Tuple[int, int]]): List of detected anomaly intervals.
            ground_truth_intervals (List[Tuple[int, int]]): List of ground truth anomaly intervals.

        Returns:
            float: Event recall value.
        """
        true_positives = 0
        for gt_event in ground_truth_intervals:
            for detected_event in detected_intervals:
                if detected_event[1] >= gt_event[0] and detected_event[0] <= gt_event[1]:
                    true_positives += 1
                    break
        return true_positives / len(ground_truth_intervals) if len(ground_truth_intervals) > 0 else 0.0

    def compute_time_to_detection(
        self, detected_intervals: List[Tuple[int, int]], ground_truth_intervals: List[Tuple[int, int]]
    ) -> List[int]:
        """
        Computes the time to detection (TTD) for each ground truth anomaly.

        Args:
            detected_intervals (List[Tuple[int, int]]): List of detected anomaly intervals.
            ground_truth_intervals (List[Tuple[int, int]]): List of ground truth anomaly intervals.

        Returns:
            List[int]: List of time to detection values for each ground truth anomaly.
        """
        ttd_list = []
        for gt_start, gt_end in ground_truth_intervals:
            earliest_detection = None
            for detected_start, detected_end in detected_intervals:
                if detected_end >= gt_start and detected_start <= gt_end:
                    detection_time = max(detected_start, gt_start)
                    if earliest_detection is None or detection_time < earliest_detection:
                        earliest_detection = detection_time
            if earliest_detection is not None:
                ttd = earliest_detection - gt_start
                ttd_list.append(ttd)
        return ttd_list

    def compute_overlap_duration(self, detected_intervals: List[Tuple[int, int]], ground_truth_intervals: List[Tuple[int, int]]) -> int:
        """
        Computes the total overlap duration between detected and ground truth anomaly intervals.

        Args:
            detected_intervals (List[Tuple[int, int]]): List of detected anomaly intervals.
            ground_truth_intervals (List[Tuple[int, int]]): List of ground truth anomaly intervals.

        Returns:
            int: Total overlap duration.
        """
        total_overlap = 0
        overlap_points = set()
        for gt_start, gt_end in ground_truth_intervals:
            for detected_start, detected_end in detected_intervals:
                overlap_start = max(gt_start, detected_start)
                overlap_end = min(gt_end, detected_end)
                if overlap_start <= overlap_end:
                    overlap_points.update(range(overlap_start, overlap_end + 1))
        return len (overlap_points)
    

    def get_ground_truth_intervals(self, true_anomalies: Dict[str, Dict[str, Any]]) -> List[Tuple[int, int]]:
        """
        Extracts ground truth anomaly intervals from the true anomalies dictionary.

        Args:
            true_anomalies (Dict[str, Dict[str, Any]]): Dictionary containing ground truth anomalies for each symptom.

        Returns:
            List[Tuple[int, int]]: List of ground truth anomaly intervals.
        """
        intervals = []
        for column, anomaly_types in true_anomalies.items():
            for anomaly_type, anomaly_data in anomaly_types.items():
                if anomaly_type in ['sir_mpox', 'sir_measles']:
                    indices = anomaly_data['indices']
                    if len(indices) == 0:
                        continue
                    start_idx = indices.min()
                    end_idx = indices.max()
                    intervals.append((start_idx, end_idx))
        return intervals

    def get_detected_intervals_dbscan(
        self, anomaly_indices: np.ndarray, eps: int = 10, min_samples: int = 1
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Applies DBSCAN clustering to group detected anomaly indices into contiguous intervals.

        Args:
            anomaly_indices (np.ndarray): Array of detected anomaly indices.
            eps (int, optional): The maximum distance between two samples for them to be considered as in the same neighborhood. Defaults to 10.
            min_samples (int, optional): The number of samples in a neighborhood for a point to be considered as a core point. Defaults to 1.

        Returns:
            Tuple[List[Tuple[int, int]], np.ndarray]: A list of detected anomaly intervals and the cluster labels.
        """
        if len(anomaly_indices) == 0:
            return [], np.array([])
        anomaly_indices_reshaped = np.array(anomaly_indices).reshape(-1, 1)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(anomaly_indices_reshaped)
        labels = clustering.labels_
        intervals = []
        for cluster_label in set(labels):
            if cluster_label == -1:
                continue
            cluster_indices = anomaly_indices[labels == cluster_label]
            start_idx = cluster_indices.min()
            end_idx = cluster_indices.max()
            intervals.append((start_idx, end_idx))
        return intervals, labels
