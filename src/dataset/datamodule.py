from typing import Optional, Union, Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class TimeSeriesDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning Data Module for preparing time series data for training, validation, and testing.
    Supports different methods for sequence generation, including fixed-length sequences (with or without overlap),
    variable-length sequences with padding, and sliding window sequences.
    
    Attributes:
        data (pd.DataFrame): The input time series data.
        sequence_method (str): The method used for generating sequences.
        scaler (StandardScaler): Scaler used for normalizing the data.
        seq_len (int): The length of each sequence.
        batch_size (int): The batch size for the DataLoaders.
        train_ratio (float): The ratio of training data.
        val_ratio (float): The ratio of validation data.
        input_dim (int): The dimensionality of the input data.
    """
    def __init__(self, data, hparams, sequence_method: str = 'sliding_window'):
        """
        Initializes the TimeSeriesDataModule.

        Args:
            data (pd.DataFrame): The input time series data.
            hparams: Hyperparameters, including sequence length, batch size, train/validation ratios, and input dimension.
            sequence_method (str, optional): The method used for generating sequences ('sliding_window',
                'fixed_length_overlap', 'fixed_length_no_overlap', 'variable_length_padding'). Default is 'sliding_window'.
        """
        super().__init__()
        self.data = data
        self.sequence_method = sequence_method
        self.scaler = StandardScaler()
        
        self.seq_len = hparams.seq_len
        self.batch_size = hparams.batch_size
        self.train_ratio = hparams.train_ratio
        self.val_ratio = hparams.val_ratio
        self.input_dim = hparams.input_dim  

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Prepares the dataset for training, validation, and testing.

        Args:
            stage (Optional[str]): The stage to set up (unused in this implementation).
        """
        data_values = self.data.values.astype('float32')
        # Fit the scaler on the data (we are fitting on the whole available data for scaling)
        data_values = self.scaler.fit_transform(data_values)
        
        if self.sequence_method == 'fixed_length_overlap':
            sequences = self.create_fixed_length_sequences(data_values, self.seq_len, overlap=True)
        elif self.sequence_method == 'fixed_length_no_overlap':
            sequences = self.create_fixed_length_sequences(data_values, self.seq_len, overlap=False)
        elif self.sequence_method == 'variable_length_padding':
            sequences = self.create_variable_length_sequences_with_padding(data_values, self.seq_len)
        elif self.sequence_method == 'sliding_window':
            sequences = self.create_sliding_window_sequences(data_values, self.seq_len)
        else:
            raise ValueError(f"Unknown sequence_method: {self.sequence_method}")
        
        n_samples = len(sequences)
        n_train = int(n_samples * self.train_ratio)
        n_val = int(n_samples * self.val_ratio)
        n_test = n_samples - n_train - n_val
        
        self.train_data = sequences[:n_train]
        self.val_data = sequences[n_train:n_train + n_val]
        # self.test_data = sequences[n_train + n_val:]
        self.test_data = sequences

    def create_fixed_length_sequences(self, data: np.ndarray, seq_len: int, overlap: bool = True) -> torch.Tensor:
        """
        Creates fixed-length sequences with or without overlap.

        Args:
            data (np.ndarray): The input data.
            seq_len (int): The length of each sequence.
            overlap (bool, optional): Whether to create overlapping sequences. Default is True.
        
        Returns:
            torch.Tensor: A tensor containing the sequences.
        """
        sequences = []
        step = 1 if overlap else seq_len
        for i in range(0, len(data) - seq_len + 1, step):
            seq = data[i:i + seq_len]
            sequences.append(seq)
        # Handle the last sequence if necessary (otherwise the last days are being droped)
        # Might be able to find a more cleaner way
        if (len(data) - seq_len) % step != 0 and overlap:
            seq = data[-seq_len:]
            sequences.append(seq)
        return torch.tensor(sequences, dtype=torch.float32)
    
    def create_variable_length_sequences_with_padding(self, data: np.ndarray, seq_len: int) -> torch.Tensor:
        """
        Creates variable-length sequences with padding to match the specified sequence length.

        Args:
            data (np.ndarray): The input data.
            seq_len (int): The length of each sequence.
        
        Returns:
            torch.Tensor: A tensor containing the sequences with padding.
        """
        sequences = []
        num_full_sequences = len(data) // seq_len
        remainder = len(data) % seq_len
        for i in range(num_full_sequences):
            seq = data[i * seq_len:(i + 1) * seq_len]
            sequences.append(seq)
        if remainder != 0:
            # Pad the last sequence
            last_seq = data[-remainder:]
            padding = np.zeros((seq_len - remainder, data.shape[1]), dtype=np.float32)
            last_seq_padded = np.vstack((last_seq, padding))
            sequences.append(last_seq_padded)
        return torch.tensor(sequences, dtype=torch.float32)
    
    def create_sliding_window_sequences(self, data: np.ndarray, seq_len: int) -> torch.Tensor:
        """
        Creates sequences using a sliding window approach.

        Args:
            data (np.ndarray): The input data.
            seq_len (int): The length of each sequence.
        
        Returns:
            torch.Tensor: A tensor containing the sliding window sequences.
        """
        sequences = []
        for i in range(len(data) - seq_len + 1):
            seq = data[i:i + seq_len]
            sequences.append(seq)
        return torch.tensor(sequences, dtype=torch.float32)
    
    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: The training data loader.
        """
        return DataLoader(TensorDataset(self.train_data, self.train_data), batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: The validation data loader.
        """
        return DataLoader(TensorDataset(self.val_data, self.val_data), batch_size=self.batch_size)
    
    def test_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the test dataset.

        Returns:
            DataLoader: The test data loader.
        """
        return DataLoader(TensorDataset(self.test_data, self.test_data), batch_size=self.batch_size)



class OnlineTimeSeriesDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule designed for online time series data processing. 
    This module enables dynamic dataset updates and adapts to various sequence creation methods, 
    allowing for continuous training on time-evolving data. Supports a customizable headstart 
    size and flexible batch generation.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset from which online batches are derived.
    hparams : Any
        Hyperparameters dictionary containing configuration such as sequence length 
        and batch size.
    sequence_method : str, optional
        Method for sequence creation, chosen from 'fixed_length_overlap', 
        'fixed_length_no_overlap', 'variable_length_padding', or 'sliding_window'.
    headstart_size : int, optional
        Initial subset of data to use for training, after which online updates begin.
    """
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        hparams: Union[dict, Any], 
        sequence_method: str = 'sliding_window', 
        headstart_size: int = 356
    ):
        super().__init__()
        self.full_data: pd.DataFrame = data
        self.headstart_size: int = headstart_size

        self.sequence_method: str = sequence_method
        self.seq_len = hparams.seq_len
        self.batch_size = hparams.batch_size
        self.train_ratio = hparams.train_ratio
        self.val_ratio = hparams.val_ratio
        self.input_dim = hparams.input_dim

        self.scaler: StandardScaler = StandardScaler()
        
        self.data: pd.DataFrame = data.iloc[:headstart_size]
        self.current_index: int = headstart_size
        self.current_data_length = headstart_size 

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up the initial training data by scaling and preparing sequences.

        Parameters
        ----------
        stage : Optional[str], optional
            The stage of the setup process, used in Lightning for setting up data in 
            training, validation, or test phases.
        """
        data_values = self.data.values.astype('float32')
        self.scaler.fit(data_values)
        data_values = self.scaler.transform(data_values)
        self.train_data = self.create_sequences(data_values)

    def create_sequences(self, data_values: pd.DataFrame) -> pd.DataFrame:
        """
        Generates time series sequences from the scaled data using the specified 
        sequence creation method.

        Parameters
        ----------
        data_values : pd.DataFrame
            Scaled data values prepared for sequence generation.

        Returns
        -------
        pd.DataFrame
            Transformed data structured as sequences for model training.
        """
        if self.sequence_method == 'fixed_length_overlap':
            return self.create_fixed_length_sequences(data_values, self.seq_len, overlap=True)
        elif self.sequence_method == 'fixed_length_no_overlap':
            return self.create_fixed_length_sequences(data_values, self.seq_len, overlap=False)
        elif self.sequence_method == 'variable_length_padding':
            return self.create_variable_length_sequences_with_padding(data_values, self.seq_len)
        elif self.sequence_method == 'sliding_window':
            return self.create_sliding_window_sequences(data_values, self.seq_len)
        else:
            raise ValueError(f"Unknown sequence_method: {self.sequence_method}")
        
    # def create_sequences(self, data_values):
    #     sequences = []
    #     seq_start_indices = []
    #     seq_len = self.seq_len
    #     step_size = 1  # Adjust as needed (assuming sliding window with step size 1)

    #     for i in range(0, len(data_values) - seq_len + 1, step_size):
    #         seq = data_values[i:i + seq_len]
    #         sequences.append(seq)
    #         seq_start_indices.append(i)
    #     sequences = np.array(sequences, dtype='float32')
    #     return sequences, seq_start_indices
    
    def create_fixed_length_sequences(self, data: np.ndarray, seq_len: int, overlap: bool = True) -> torch.Tensor:
        """
        Creates fixed-length sequences with or without overlap.

        Args:
            data (np.ndarray): The input data.
            seq_len (int): The length of each sequence.
            overlap (bool, optional): Whether to create overlapping sequences. Default is True.
        
        Returns:
            torch.Tensor: A tensor containing the sequences.
        """
        sequences = []
        step = 1 if overlap else seq_len
        for i in range(0, len(data) - seq_len + 1, step):
            seq = data[i:i + seq_len]
            sequences.append(seq)
        # Handle the last sequence if necessary (otherwise the last days are being droped)
        # Might be able to find a more cleaner way
        if (len(data) - seq_len) % step != 0 and overlap:
            seq = data[-seq_len:]
            sequences.append(seq)
        return torch.tensor(sequences, dtype=torch.float32)
    
    def create_variable_length_sequences_with_padding(self, data: np.ndarray, seq_len: int) -> torch.Tensor:
        """
        Creates variable-length sequences with padding to match the specified sequence length.

        Args:
            data (np.ndarray): The input data.
            seq_len (int): The length of each sequence.
        
        Returns:
            torch.Tensor: A tensor containing the sequences with padding.
        """
        sequences = []
        num_full_sequences = len(data) // seq_len
        remainder = len(data) % seq_len
        for i in range(num_full_sequences):
            seq = data[i * seq_len:(i + 1) * seq_len]
            sequences.append(seq)
        if remainder != 0:
            # Pad the last sequence
            last_seq = data[-remainder:]
            padding = np.zeros((seq_len - remainder, data.shape[1]), dtype=np.float32)
            last_seq_padded = np.vstack((last_seq, padding))
            sequences.append(last_seq_padded)
        return torch.tensor(sequences, dtype=torch.float32)
    
    def create_sliding_window_sequences(self, data: np.ndarray, seq_len: int) -> torch.Tensor:
        """
        Creates sequences using a sliding window approach.

        Args:
            data (np.ndarray): The input data.
            seq_len (int): The length of each sequence.
        
        Returns:
            torch.Tensor: A tensor containing the sliding window sequences.
        """
        sequences = []
        for i in range(len(data) - seq_len + 1):
            seq = data[i:i + seq_len]
            sequences.append(seq)
        return torch.tensor(sequences, dtype=torch.float32)

    def get_online_batch(self) -> Optional[pd.DataFrame]:
        """
        Retrieves the next batch of data from the dataset for online training.

        Returns
        -------
        Optional[pd.DataFrame]
            The next batch of data or None if there is no remaining data.
        """
        if self.current_index >= len(self.full_data):
            return None
        batch_end_index = min(self.current_index + self.batch_size, len(self.full_data))
        online_batch = self.full_data.iloc[self.current_index:batch_end_index]
        self.current_index = batch_end_index
        return online_batch

    def update_dataset(self, new_data: pd.DataFrame) -> None:
        """
        Updates the dataset with new data and retrains the scaler.

        Parameters
        ----------
        new_data : pd.DataFrame
            New incoming data used to update the existing dataset.
        """
        self.data = pd.concat([self.data, new_data], ignore_index=True)
        self.current_data_length = len(self.data)  # Update current data length

        data_values = self.data.values.astype('float32')
        self.scaler.fit(data_values)
        data_values = self.scaler.transform(data_values)
        self.train_data = self.create_sequences(data_values)

    def get_current_length(self):
        """
        Returns the current length of the dataset.
        """
        return self.current_data_length
    
    def train_dataloader(self) -> DataLoader:
        """
        Provides a DataLoader for training using the current training data.

        Returns
        -------
        DataLoader
            A PyTorch DataLoader for batch sampling from the training data.
        """
        return DataLoader(
            TensorDataset(self.train_data, self.train_data),
            batch_size=self.batch_size,
            shuffle=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the test dataset.

        Returns:
            DataLoader: The test data loader.
        """
        return DataLoader(
            TensorDataset(self.train_data, self.train_data),
            batch_size=self.batch_size,
            shuffle=False
        )
