import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.models.base_sklearn_model import SklearnAnomalyDetector
from src.models.model_registery import ModelRegistry

@ModelRegistry.register('pca')
class PCAAnomalyDetector(SklearnAnomalyDetector):
    """
    PCA-based anomaly detection model.

    This class leverages Principal Component Analysis (PCA) to detect anomalies in the data.
    Anomalies are identified based on the reconstruction error between the original and reconstructed data.
    """
    def __init__(self, hparams: dict, logger=None) -> None:
        """
        Initialize the PCAAnomalyDetector.

        Args:
            hparams (dict): Dictionary of hyperparameters for the PCA model.
                - 'n_components' (int): The number of principal components to use for dimensionality reduction.
            logger (optional): Logger instance for logging model-related information.
        """
        super().__init__(hparams, logger=logger)
        self.n_components = hparams.get('n_components', 2)
        self.scaler = StandardScaler()
        self.model = PCA(n_components=self.n_components)

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the PCA model on the input data.

        Args:
            X (np.ndarray): The input data to fit the model on.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

    def compute_anomaly_scores(self, X: np.ndarray) -> None:
        """
        Compute anomaly scores for the input data.

        Args:
            X (np.ndarray): The input data to compute anomaly scores for.

        Returns:
            None
        """
        X_scaled = self.scaler.transform(X)
        X_reduced = self.model.transform(X_scaled)
        X_reconstructed = self.model.inverse_transform(X_reduced)
        reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        self.anomaly_scores = reconstruction_error