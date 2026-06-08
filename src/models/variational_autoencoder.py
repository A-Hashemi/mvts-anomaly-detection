import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.model_registery import ModelRegistry
from .base_reconstruction_model import ReconstructionAnomalyDetector


@ModelRegistry.register('variational_autoencoder')
class VariationalAutoencoder(ReconstructionAnomalyDetector):
    """
    A Variational Autoencoder (VAE) for anomaly detection using probabilistic reconstruction.
    """

    def __init__(self, hparams):
        super(VariationalAutoencoder, self).__init__(hparams)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(hparams.input_dim, hparams.hidden_dim),
            nn.ReLU()
        )

        # Latent Space Parameters
        self.fc_mu = nn.Linear(hparams.hidden_dim, hparams.latent_dim)
        self.fc_logvar = nn.Linear(hparams.hidden_dim, hparams.latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hparams.latent_dim, hparams.hidden_dim),
            nn.ReLU(),
            nn.Linear(hparams.hidden_dim, hparams.input_dim),
            nn.Sigmoid()  # Ensures output is between 0-1
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from a normal distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Sample from standard normal
        return mu + eps * std

    def forward(self, x):
        """Forward pass returning only the reconstructed output to match the Vanilla AE behavior."""
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)

        # Store mu and logvar for KL divergence (avoiding tuple return issue)
        self.mu = mu
        self.logvar = logvar

        return recon_x  # Now returns the same shape as Vanilla AE

    def loss_function(self, recon_x, x):
        """
        Computes the VAE loss (Reconstruction loss + KL divergence).
        Uses `self.mu` and `self.logvar` stored in forward() to avoid changing training logic.
        """
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        return recon_loss + kl_div
