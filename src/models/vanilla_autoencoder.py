import torch.nn as nn

from src.models.model_registery import ModelRegistry

from .base_reconstruction_model import ReconstructionAnomalyDetector


@ModelRegistry.register('vanilla_autoencoder')
class VanillaAutoencoder(ReconstructionAnomalyDetector):
    """
    A Vanilla Autoencoder model for anomaly detection using reconstruction.
    It consists of a simple feedforward encoder and decoder to reconstruct the input data.
    
    Attributes:
        encoder (nn.Sequential): The encoder part of the autoencoder.
        decoder (nn.Sequential): The decoder part of the autoencoder.
        model (nn.Sequential): The complete autoencoder model consisting of the encoder and decoder.
    """
    def __init__(self, hparams):
        """
        Initializes the VanillaAutoencoder.

        Args:
            hparams: Hyperparameters, including input dimension and hidden dimension.
        """
        super(VanillaAutoencoder, self).__init__(hparams)
        self.encoder = nn.Sequential(
            nn.Linear(hparams.input_dim, hparams.hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hparams.hidden_dim, hparams.input_dim),
            nn.ReLU(),
        )
        self.model = nn.Sequential(self.encoder, self.decoder)