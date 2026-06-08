import torch
import torch.nn as nn
import math

from src.models.model_registery import ModelRegistry
from .base_reconstruction_model import ReconstructionAnomalyDetector

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return x


@ModelRegistry.register('anomaly_transformer')
class AnomalyTransformer(ReconstructionAnomalyDetector):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.input_dim = hparams.input_dim
        self.hidden_dim = hparams.hidden_dim
        self.seq_len = hparams.seq_len
        nhead = hparams.nhead if hasattr(hparams, 'nhead') else 4
        num_layers = hparams.num_layers if hasattr(hparams, 'num_layers') else 2

        # Input projection
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.hidden_dim, max_len=self.seq_len)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, self.hidden_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=nhead, batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=nhead, batch_first=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            Reconstructed x: [batch_size, seq_len, input_dim]
        """
        self.train()
        batch_size, seq_len, _ = x.size()

        # Transform input
        x = self.input_proj(x)  # [batch_size, seq_len, hidden_dim]
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]
        x = x + self.pos_embedding.permute(1, 0, 2)  # Match [seq_len, batch, hidden_dim]

        # Encode
        memory = self.encoder(x)

        # Use zero tensor as decoder input (teacher forcing not used here)
        tgt = torch.zeros_like(x)
        tgt = self.positional_encoding(tgt)

        # Decode
        out = self.decoder(tgt, memory)  # [seq_len, batch_size, hidden_dim]
        out = out.permute(1, 0, 2)  # [batch_size, seq_len, hidden_dim]
        out = self.output_proj(out)  # [batch_size, seq_len, input_dim]

        return out

    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(x_hat, x)


#import torch
#import torch.nn as nn
#import torch.nn.functional as F

#from src.models.model_registery import ModelRegistry
#from .base_reconstruction_model import ReconstructionAnomalyDetector


#@ModelRegistry.register('anomaly_transformer')
#class AnomalyTransformer(ReconstructionAnomalyDetector):
#    """
#    Anomaly Transformer based on attention with prior association discrepancy.
#    For reconstruction-style anomaly detection on multivariate time series.
#    """
#    def __init__(self, hparams):
#        super().__init__(hparams)
#        self.input_dim = hparams.input_dim
#        self.hidden_dim = hparams.hidden_dim
#        self.seq_len = hparams.seq_len

#        self.input_proj = nn.Linear(hparams.input_dim, hparams.embed_dim)
#        self.encoder_layer = nn.TransformerEncoderLayer(
#            d_model=hparams.embed_dim,
#            nhead=hparams.num_heads,
#            dim_feedforward=hparams.hidden_dim,
#            batch_first=True
#        )
#        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

#        self.decoder = nn.Sequential(
#            nn.Linear(self.input_dim, self.hidden_dim),
#            nn.ReLU(),
#            nn.Linear(self.hidden_dim, self.input_dim)
#        )

#    def forward(self, x: torch.Tensor) -> torch.Tensor:
#        """
#        x: [batch_size, seq_len, input_dim]
#        Returns: [batch_size, seq_len, input_dim]
#        """
#        self.train()
#        x = self.input_proj(x)
#        encoded = self.encoder(x)
#        decoded = self.decoder(encoded)  # shape: [B, L, D]
#        return decoded

#    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
#        return F.mse_loss(x_hat, x)
