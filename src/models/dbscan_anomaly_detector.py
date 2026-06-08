import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from src.models.base_sklearn_model import SklearnAnomalyDetector
from src.models.model_registery import ModelRegistry


@ModelRegistry.register('dbscan')
class DBSCANAnomalyDetector(SklearnAnomalyDetector):
    """
    DBSCAN-based anomaly detection model.

    This class leverages the DBSCAN clustering algorithm to detect anomalies in the data.
    Points labeled as noise (-1) by DBSCAN are considered anomalies.
    """
    def __init__(self, hparams: dict, logger=None) -> None:
        """
        Initialize the DBSCANAnomalyDetector.

        Args:
            hparams (dict): Dictionary of hyperparameters for the DBSCAN model.
                - 'eps' (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
                - 'min_samples' (int): The number of samples in a neighborhood for a point to be considered as a core point.
            logger (optional): Logger instance for logging model-related information.
        """
        super().__init__(hparams, logger=logger)
        self.scaler = StandardScaler()
        self.model = DBSCAN(
            eps=hparams.get('eps', 0.5),
            min_samples=hparams.get('min_samples', 5)
        )

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the DBSCAN model on the input data.

        Args:
            X (np.ndarray): The input data to fit the model on.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.labels_ = self.model.labels_

    def compute_anomaly_scores(self, X: np.ndarray) -> None:
        """
        Compute anomaly scores for the input data.

        Args:
            X (np.ndarray): The input data to compute anomaly scores for.

        Returns:
            None
        """
        X_scaled = self.scaler.transform(X)
        labels = self.model.fit_predict(X_scaled)
        # Label -1 is considered as noise (anomaly)
        anomaly_mask = labels == -1
        # Assign anomaly scores: 1 for anomalies, 0 for normal points
        self.anomaly_scores = anomaly_mask.astype(float)