# An Epidemiology-Guided Machine Learning Framework for Real-Time Anomaly Detection in Disease Outbreak Monitoring using Symptom Surveillance Data

This repository contains the code for 'An Epidemiology-Guided Machine Learning Framework for Real-Time Anomaly Detection in Disease Outbreak Monitoring using Symptom Surveillance Data'. 

## Overview

The project evaluates reconstruction-based and classical anomaly detection methods on multivariate time-series surveillance data. Synthetic outbreaks are generated and injected into symptom-related time series, after which anomaly detection performance is evaluated using point-level and event-level metrics.

Implemented components include:

- Synthetic outbreak generation and injection
- Offline and online anomaly detection experiments
- PCA, autoencoder-based models, LSTM autoencoder, variational autoencoder, and transformer autoencoder models
- Event-level clustering of detected anomaly points
- Evaluation metrics for injected outbreak windows
- Weights & Biases logging for experiment tracking and visualization

## Repository structure

```text
.
├── config/                  # Hydra configuration files
│   ├── anomalies/            # Outbreak/anomaly generation settings
│   ├── dataset/              # Dataset paths
│   ├── model/                # Model-specific settings
│   └── trainer/              # Trainer settings
├── data/                    # Example/synthetic data files
├── src/                     # Source code
│   ├── anomaly_detection/    # Detector registry and detection logic
│   ├── anomaly_generation/   # Outbreak simulation and injection code
│   ├── dataset/              # Dataset loading and datamodules
│   ├── evaluation/           # Evaluation metrics and utilities
│   ├── models/               # Model implementations
│   ├── utils/                # Utility functions
│   └── visualization/        # Plotting and W&B visualization functions
├── main.py                  # Main experiment script
├── requirements.txt         # Python dependencies
├── CITATION.cff             # Citation metadata
├── LICENSE                  # License file
└── README.md                # Repository documentation
```

## Installation

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

A virtual environment or Conda environment is recommended.

```

## License

This project is released under the MIT License. See `LICENSE` for details.
