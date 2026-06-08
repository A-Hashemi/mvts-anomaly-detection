import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb


class SklearnAnomalyDetector(ABC):
    def __init__(self, hparams, logger=None):
        """
        Base class for scikit-learn anomaly detection models.

        Args:
            hparams: Hyperparameters for the model.
            logger: Optional logger for logging plots and metrics.
        """
        self.hparams = hparams
        self.logger = logger
        self.model = None  # To be defined in child classes
        self.data = None   # Data to fit the model
        self.anomaly_scores = None
        self.anomaly_indices = None

    @abstractmethod
    def fit(self, X):
        """
        Fit the model to the data.

        Args:
            X (np.ndarray): Training data.
        """
        pass

    @abstractmethod
    def compute_anomaly_scores(self, X):
        """
        Compute anomaly scores for the data.

        Args:
            X (np.ndarray): Data to compute anomaly scores on.
        """
        pass

    def detect_anomalies(self, X):
        """
        Detect anomalies in the data.

        Args:
            X (np.ndarray): Data to detect anomalies in.

        Returns:
            dict: Dictionary containing anomaly scores and indices.
        """
        self.compute_anomaly_scores(X)
        # Determine the threshold
        threshold = np.percentile(self.anomaly_scores, 99)
        self.anomaly_indices = np.where(self.anomaly_scores > threshold)[0]

        detected_anomalies = {
            'multivariate': {
                'anomaly_scores': self.anomaly_scores,
                'anomaly_indices': self.anomaly_indices
            }
        }

        # Plot and log the anomaly scores
        if self.logger:
            self.plot_anomaly_scores()

        return detected_anomalies

    def plot_anomaly_scores(self):
        """
        Plot the anomaly scores and highlight the detected anomalies.
        """
        time_axis = np.arange(len(self.anomaly_scores))

        fig = go.Figure()

        # Plot anomaly scores
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=self.anomaly_scores,
            mode='lines',
            name='Anomaly Scores'
        ))

        # Highlight detected anomalies
        if len(self.anomaly_indices) > 0:
            fig.add_trace(go.Scatter(
                x=self.anomaly_indices,
                y=self.anomaly_scores[self.anomaly_indices],
                mode='markers',
                marker=dict(color='red', size=6),
                name='Detected Anomalies'
            ))

        fig.update_layout(
            title=f'{self.__class__.__name__} Anomaly Scores',
            xaxis_title='Time',
            yaxis_title='Anomaly Score',
            hovermode='closest',
            legend_title="Legend"
        )

        # Save the figure as HTML and log to wandb
        output_dir = self.logger.save_dir if hasattr(self.logger, 'save_dir') else './wandb_logs'
        os.makedirs(output_dir, exist_ok=True)
        plotly_html_path = os.path.join(output_dir, f'{self.__class__.__name__}_anomaly_scores.html')
        fig.write_html(plotly_html_path, auto_play=False)

        # Create a wandb Table and add the Plotly figure as HTML
        table = wandb.Table(columns=["Anomaly Scores Plot"])
        table.add_data(wandb.Html(plotly_html_path))

        # Log the Table and custom line plot to wandb
        self.logger.experiment.log({f"{self.__class__.__name__} Anomaly Scores Plot": table})

        # Also log the custom line plot
        data = [[x, y] for (x, y) in zip(time_axis, self.anomaly_scores)]
        line_table = wandb.Table(data=data, columns=["Time", "Anomaly Score"])
        self.logger.experiment.log(
            {
                f"{self.__class__.__name__} Anomaly Scores Line Plot": wandb.plot.line(
                    line_table, "Time", "Anomaly Score", title=f"{self.__class__.__name__} Anomaly Scores Over Time"
                )
            }
        )
        print("plots are logged")
