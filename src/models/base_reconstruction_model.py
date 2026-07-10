import os

import numpy as np
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb


class ReconstructionAnomalyDetector(pl.LightningModule):
    """
    A PyTorch Lightning Module for detecting anomalies using a reconstruction-based approach.
    The model aims to reconstruct input data and detects anomalies based on reconstruction errors.
    
    Attributes:
        hparams: Hyperparameters, including learning rate and sequence length.
        model: The underlying neural network model to be used for reconstruction (to be defined in child classes).
        test_outputs (list): Stores the outputs from the test step for inference.
        data: The dataset used for inference (set externally).
    """
    def __init__(self, hparams):
        """
        Initializes the ReconstructionAnomalyDetector.

        Args:
            hparams: Hyperparameters, including learning rate and sequence length.
        """
        super(ReconstructionAnomalyDetector, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = None  # To be defined in child classes
        self.test_outputs = []  # To store outputs from test_step
        self.data = None  # Will be set externally for inference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The reconstructed output tensor.
        """
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Defines a single training step.

        Args:
            batch (tuple): A batch of data, where the input and target are the same for reconstruction.
            batch_idx (int): The index of the batch.
        
        Returns:
            torch.Tensor: The computed training loss.
        """
        x, _ = batch  # Assuming target is same as input for reconstruction
        x_hat = self(x)
        loss = self.compute_loss(x, x_hat)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """
        Defines a single validation step.

        Args:
            batch (tuple): A batch of data, where the input and target are the same for reconstruction.
            batch_idx (int): The index of the batch.
        """
        x, _ = batch
        x_hat = self(x)
        loss = self.compute_loss(x, x_hat)
        self.log('val_loss', loss)
        
    def on_test_epoch_start(self) -> None:
    """Clear outputs collected during any previous test run."""
    self.test_outputs.clear()

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """
        Defines a single test step.

        Args:
            batch (tuple): A batch of data, where the input and target are the same for reconstruction.
            batch_idx (int): The index of the batch.
        """
        x, _ = batch
        x_hat = self(x)
        loss = self.compute_loss(x, x_hat)
        self.test_outputs.append({'loss': loss.detach(), 'x': x.detach(), 'x_hat': x_hat.detach()})

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer used for training.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Computes the reconstruction loss.

        Args:
            x (torch.Tensor): The input tensor.
            x_hat (torch.Tensor): The reconstructed output tensor.
        
        Returns:
            torch.Tensor: The mean squared error between the input and reconstructed output.
        """
        return F.mse_loss(x_hat, x, reduction='mean')

    def inference(self, data_len = None) -> dict:
        """
        Computes anomaly scores over the entire test dataset.
        Anomalies are detected based on the reconstruction error.
        
        Returns:
            dict: A dictionary containing the anomaly scores and anomaly indices.
        """
        # Collect all reconstruction errors
        all_errors = []
        seq_len = self.hparams.seq_len

        for output in self.test_outputs:
            x = output['x']  # Shape: (batch_size, seq_len, input_dim)
            x_hat = output['x_hat']  # Same shape as x
            batch_errors = torch.mean((x - x_hat) ** 2, dim=2)  # Shape: (batch_size, seq_len)
            all_errors.append(batch_errors.cpu().numpy())

        # Concatenate all errors
        all_errors = np.concatenate(all_errors, axis=0)  # Shape: (num_sequences, seq_len)

        # Determine the correct total length
        if data_len is not None:
            total_length = data_len
        else:
            total_length = len(self.data) if self.data is not None else (len(all_errors) + seq_len - 1)
            
        pointwise_errors = np.zeros(total_length)
        counts = np.zeros(total_length)

        # Iterate over each sequence and distribute errors to the corresponding time points
        for seq_idx in range(len(all_errors)):
            start_idx = seq_idx  # Assuming sliding window with step size 1 (Should be able to find a cleaner way)
            #TODO incase of playig around with the sliding widnow this should be handled properly with configs
            end_idx = start_idx + seq_len
            # Handle the case where end_idx exceeds total_length
            if end_idx > total_length:
                end_idx = total_length
                seq_errors = all_errors[seq_idx][:end_idx - start_idx]
            else:
                seq_errors = all_errors[seq_idx]
            try:
                pointwise_errors[start_idx:end_idx] += seq_errors
            except:
                #TODO should find the error source
                continue
            counts[start_idx:end_idx] += 1

        # Avoid division by zero
        counts = np.maximum(counts, 1)
        # Compute average error per time point
        anomaly_scores = pointwise_errors / counts

        # Detect anomalies based on a threshold (e.g., 99th percentile)
        #TODO might want to move to configs
        threshold = np.percentile(anomaly_scores, 99)
        self.threshold = threshold
        anomaly_indices = np.where(anomaly_scores > threshold)[0]

        detected_anomalies = {
            'multivariate': {
                'anomaly_scores': anomaly_scores,
                'anomaly_indices': anomaly_indices
            }
        }

        if self.logger:
            self.plot_anomaly_scores(anomaly_scores, anomaly_indices)

        return detected_anomalies
    
    def compute_anomaly_scores(self, x: torch.Tensor, x_hat: torch.Tensor) -> np.ndarray:
        """
        Computes anomaly scores for a batch based on the reconstruction error.

        Args:
            x (torch.Tensor): The original input tensor of shape [batch_size, seq_len, input_dim].
            x_hat (torch.Tensor): The reconstructed output tensor of the same shape as x.

        Returns:
            np.ndarray: An array of anomaly scores for each time point in the batch.
        """
        # Compute reconstruction errors per time point
        errors = torch.mean((x - x_hat) ** 2, dim=2)  # Shape: (batch_size, seq_len)
        return errors.cpu().numpy()

    def detect_anomalies(self, anomaly_scores: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        Detects anomalies based on the anomaly scores and a threshold.

        Args:
            anomaly_scores (np.ndarray): An array of anomaly scores.
            threshold (float, optional): The threshold for detecting anomalies. If None, uses the 99th percentile.

        Returns:
            np.ndarray: Indices of detected anomalies within the batch.
        """
        if threshold is None:
            threshold = np.percentile(anomaly_scores, 99)
        anomaly_indices = np.where(anomaly_scores > threshold)
        return anomaly_indices
    
    def detect_anomalies_online(self, anomaly_scores: np.ndarray) -> np.ndarray:
        """
        Detects anomalies based on the anomaly scores and a threshold.

        Args:
            anomaly_scores (np.ndarray): An array of anomaly scores.
            threshold (float, optional): The threshold for detecting anomalies. If None, uses the 99th percentile.

        Returns:
            np.ndarray: Indices of detected anomalies within the batch.
        """
        anomaly_indices = np.where(anomaly_scores > self.threshold)
        return anomaly_indices
    
    def compute_threshold(self, x: torch.Tensor, x_hat: torch.Tensor, threshold: float = 99) -> np.ndarray:
        """
        Computes anomaly scores threshold for a batch based on the reconstruction error.

        Args:
            x (torch.Tensor): The original input tensor of shape [batch_size, seq_len, input_dim].
            x_hat (torch.Tensor): The reconstructed output tensor of the same shape as x.

        Returns:
            np.ndarray: An array of anomaly scores for each time point in the batch.
        """
        # Compute reconstruction errors per time point
        anomaly_scores = torch.mean((x - x_hat) ** 2, dim=2)  # Shape: (batch_size, seq_len)
        anomaly_scores = anomaly_scores.cpu().numpy()
        self.threshold = np.percentile(anomaly_scores, 99)
            

    def plot_anomaly_scores(self, anomaly_scores, anomaly_indices):
        """
        Plot the anomaly scores and highlight the detected anomalies.
        """
        time_axis = np.arange(len(anomaly_scores))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=time_axis,
            y=anomaly_scores,
            mode='lines',
            name='Anomaly Scores'
        ))

        if len(anomaly_indices) > 0:
            fig.add_trace(go.Scatter(
                x=anomaly_indices,
                y=anomaly_scores[anomaly_indices],
                mode='markers',
                marker=dict(color='red', size=6),
                name='Detected Anomalies'
            ))

        fig.update_layout(
            title='Anomaly Scores with Detected Anomalies',
            xaxis_title='Time',
            yaxis_title='Anomaly Score',
            hovermode='closest',
            legend_title="Legend"
        )

        output_dir = self.logger.save_dir if hasattr(self.logger, 'save_dir') else './wandb_logs'
        os.makedirs(output_dir, exist_ok=True)
        plotly_html_path = os.path.join(output_dir, 'anomaly_scores_plot.html')
        fig.write_html(plotly_html_path, auto_play=False)

        table = wandb.Table(columns=["Anomaly Scores Plot"])
        table.add_data(wandb.Html(plotly_html_path))

        self.logger.experiment.log({"Anomaly Scores Plot": table})

        data = [[x, y] for (x, y) in zip(time_axis, anomaly_scores)]
        line_table = wandb.Table(data=data, columns=["Time", "Anomaly Score"])

        self.logger.experiment.log(
            {
                "Anomaly Scores Line Plot": wandb.plot.line(
                    line_table, "Time", "Anomaly Score", title="Anomaly Scores Over Time"
                )
            }
        )
