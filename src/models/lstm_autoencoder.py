import torch
import torch.nn as nn

from src.models.model_registery import ModelRegistry

from .base_reconstruction_model import ReconstructionAnomalyDetector


@ModelRegistry.register('lstm_autoencoder')
class LSTMAutoencoder(ReconstructionAnomalyDetector):
    """
    An LSTM-based Autoencoder model for anomaly detection using reconstruction.
    It consists of an LSTM encoder and an LSTM decoder for reconstructing sequential data.
    
    Attributes:
        encoder (nn.LSTM): The LSTM encoder to encode the input sequence.
        decoder (nn.LSTM): The LSTM decoder to reconstruct the input sequence from the encoded state.
        hidden_dim (int): The hidden dimension size used in both encoder and decoder.
    """
    def __init__(self, hparams):
        """
        Initializes the LSTMAutoencoder.

        Args:
            hparams: Hyperparameters, including input dimension, hidden dimension, and sequence length.
        """
        super(LSTMAutoencoder, self).__init__(hparams)
        self.encoder = nn.LSTM(input_size=hparams.input_dim, hidden_size=hparams.hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_size=hparams.hidden_dim, hidden_size=hparams.input_dim, batch_first=True)
        self.hidden_dim = hparams.hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the LSTM Autoencoder.

        Args:
            x (torch.Tensor): The input tensor of shape [batch_size, seq_len, input_dim].
        
        Returns:
            torch.Tensor: The reconstructed output tensor of the same shape as the input.
        """
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.size()
        # Encode
        _, (hidden, _) = self.encoder(x)
        # Prepare repeated hidden state for decoding
        hidden_repeated = hidden.repeat(seq_len, 1, 1).permute(1, 0, 2)
        # Decode
        decoded, _ = self.decoder(hidden_repeated)
        return decoded

    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Computes the reconstruction loss for the LSTM Autoencoder.

        Args:
            x (torch.Tensor): The original input tensor.
            x_hat (torch.Tensor): The reconstructed output tensor.
        
        Returns:
            torch.Tensor: The mean squared error between the input and reconstructed output.
        """
        return nn.functional.mse_loss(x_hat, x)