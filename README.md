# MVTS Anomaly Detection for Online Outbreak Surveillance

This repository contains the original code for multivariate time-series anomaly detection experiments for disease outbreak surveillance. The code supports synthetic outbreak injection and benchmarking of several anomaly detection models in offline and online settings.

## Overview

The project evaluates reconstruction-based and classical anomaly detection methods on multivariate time-series surveillance data. Synthetic outbreaks are generated and injected into symptom-related time series, after which anomaly detection performance is evaluated using point-level and event-level metrics.

Implemented components include:

- Synthetic outbreak generation and injection
- Offline and online anomaly detection experiments
- PCA, autoencoder-based models, LSTM autoencoder, variational autoencoder, and anomaly transformer models
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
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
pip install -r requirements.txt
```

A virtual environment or Conda environment is recommended.

## Data

This repository includes the data files currently provided with the code:

```text
data/synthetic_dataset_daily_aggregated.parquet.gzip
data/translation_df.csv
```

The original real surveillance data are not included if they are subject to access restrictions.

Before running the code on another machine, check the dataset paths in:

```text
config/dataset/dataset.yaml
```

The current code may contain local machine-specific paths. If needed, replace them with relative paths such as:

```yaml
path: data/synthetic_dataset_daily_aggregated.parquet.gzip
translation_path: data/translation_df.csv
```

## Running the experiment

From the repository root, run:

```bash
python main.py
```

The project uses Hydra configuration files. You can modify the selected model in:

```text
config/config.yaml
```

For example, the model can be selected through the defaults section:

```yaml
defaults:
  - model: anomaly_transformer
  - dataset: dataset
  - anomalies: measles_mild
```

## Weights & Biases logging

This code uses Weights & Biases (W&B) for experiment tracking and visualization.

Before running with W&B, log in once:

```bash
wandb login
```

Then run:

```bash
python main.py
```

A W&B run link should be printed in the terminal, allowing you to inspect metrics, plots, and experiment logs.

The local `wandb/` folder is intentionally ignored by Git and should not be uploaded to GitHub.

## Notes for reproducibility

- The code in `src/` is kept as the original working code.
- Local cache files such as `__pycache__/`, `.DS_Store`, and W&B run folders are excluded from the repository.
- Users running the code on a new machine may need to update dataset paths in `config/dataset/dataset.yaml`.
- If real surveillance data are unavailable, users should use the included synthetic/example data or provide their own data in the same expected format.

## Citation

If you use this repository, please cite the corresponding paper. The citation will be updated after publication.

```bibtex
@software{hashemi_mvts_anomaly_detection,
  author = {Hashemi, Atiye},
  title = {MVTS Anomaly Detection for Online Outbreak Surveillance},
  year = {2026},
  url = {https://github.com/YOUR-USERNAME/YOUR-REPOSITORY}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
