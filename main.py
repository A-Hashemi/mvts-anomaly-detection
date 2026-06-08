import json
import os

import hydra
import pytorch_lightning as pl
import torch
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

import wandb
from src.anomaly_detection.detector_registry import DetectorRegistry
from src.anomaly_generation.generator import (generate_anomalies,
                                              inject_anomalies)
from src.dataset.datamodule import (OnlineTimeSeriesDataModule,
                                    TimeSeriesDataModule)
from src.dataset.loader import load_dataset
from src.evaluation.evaluator import evaluate_detection
from src.models.model_registery import ModelRegistry

@hydra.main(config_path="config", config_name="config")
def run_experiment(cfg):
    """
    Runs an anomaly detection experiment based on the provided configuration.
    
    This function handles the loading of the dataset, generation and injection of anomalies,
    training/testing of the anomaly detection model, and evaluation of the detection results.
    
    Args:
        cfg: The configuration object containing all experiment parameters, loaded by Hydra.
    """
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))

    wandb_logger = WandbLogger(
        name=cfg.experiment.name,
        project=cfg.experiment.project,
    )

    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))


    # Load dataset
    data = load_dataset(cfg.dataset.path, cfg.dataset.translation_path)
    print("Loaded the dataset")

    # Generate anomalies
    anomalies = generate_anomalies(cfg.anomalies, data, logger=wandb_logger)

    print("Generated the anomalies")

    # Inject anomalies into the dataset
    modified_data, anomaly_info = inject_anomalies(data, anomalies)
    print("Injected the anomalies")

    from src.visualization.plot_anomalies import plot_anomalies_interactive

    # Get features to plot from config
    features_to_plot = cfg.features_to_plot

    fig = plot_anomalies_interactive(data, modified_data, anomaly_info, features_to_plot)

    # Save Plotly figure as HTML
    plotly_html_path = 'plotly_figure.html'
    fig.write_html(plotly_html_path, auto_play=False)

    # Create a wandb Table and add the Plotly figure as HTML
    table = wandb.Table(columns=["Plotly Figure"])
    table.add_data(wandb.Html(plotly_html_path))

    # Log the Table to wandb
    wandb_logger.experiment.log({"Anomaly Visualization": table})

    if cfg.model.name in ["stumpy", "other_non_dl_models"]:
        # Existing code for non-deep learning models
        detector = DetectorRegistry.get(cfg.model.name)
        detected_anomalies = detector(modified_data, cfg.model, cfg.anomalies, logger=wandb_logger)

    elif cfg.model.name in ["pca", "isolation_forest", "dbscan"]:
        # Sklearn models
        ModelClass = ModelRegistry.get(cfg.model.name)
        model = ModelClass(cfg.model.model.hparams, logger=wandb_logger)
        model.data = modified_data.values 

        model.fit(model.data)
        detected_anomalies = model.detect_anomalies(model.data)
    else:
        if cfg.experiment.mode == 'online':
            data_module = OnlineTimeSeriesDataModule(
                data=modified_data,
                hparams=cfg.model.hparams,
                sequence_method=cfg.model.sequence_method,
                headstart_size=cfg.model.headstart_size
            )
            data_module.setup()

            ModelClass = ModelRegistry.get(cfg.model.name)
            model = ModelClass(cfg.model.hparams)

            trainer = pl.Trainer(logger=wandb_logger, **cfg.model.trainer)
            trainer.fit(model, datamodule=data_module)

            # Initialize arrays
            N = len(modified_data)
            detected_anomalies_array = np.zeros(N, dtype=int)
            detected_anomalies_scores_array = np.zeros(N, dtype=int)
            training_anomalies_frequency_array = np.zeros(N, dtype=int)

            online_batch = data_module.get_online_batch()

            anomlies_detected_in_training = []
            train_data_len = int(cfg.model.headstart_size) - 1

            while online_batch is not None:
                online_batch_values = online_batch.values.astype('float32')
                online_batch_scaled = data_module.scaler.transform(online_batch_values)
                online_sequences = data_module.create_sequences(online_batch_scaled)
                if not online_sequences.any():
                    break 
                #model.eval()
                with torch.no_grad():
                    # update the threshold:
                    # Test the model
                    trainer.test(model, datamodule=data_module)
                    # Get current data length from data module
                    current_data_length = data_module.get_current_length()
                    
                    # Pass the correct length to inference
                    detected_anomalies_train = model.inference(data_len=current_data_length)
                    anomlies_detected_in_training.append(detected_anomalies_train)

                    predictions = model(online_sequences)
                    anomaly_scores = model.compute_anomaly_scores(online_sequences, predictions)
                    anomaly_indices = model.detect_anomalies_online(anomaly_scores)
                

                for seq_idx, time_step in zip(*anomaly_indices):
                    data_index = train_data_len + seq_idx + time_step
                    if data_index < N:
                        detected_anomalies_array[data_index] = 1

                for time_step, score in enumerate(anomaly_scores[0]):
                    data_index = train_data_len + time_step
                    if data_index < N:
                        detected_anomalies_scores_array[data_index] = score
                
                data_module.update_dataset(online_batch)
                
                # Retrain the model
                trainer = pl.Trainer(logger=wandb_logger, **cfg.model.trainer)
                trainer.fit(model, datamodule=data_module)
                
                # Get the next online batch
                online_batch = data_module.get_online_batch()
                train_data_len += len(online_batch_values)

            wandb_logger.experiment.log({
                'training_anomalies_frequency': wandb.Histogram(training_anomalies_frequency_array)
            })
            
            detected_anomalies = {
                    'multivariate': {
                        'anomaly_indices': np.where(detected_anomalies_array == 1)[0],
                        'anomaly_scores': detected_anomalies_scores_array,
                    }
                }
        else:
            # Deep Learning models
            ModelClass = ModelRegistry.get(cfg.model.name)
            model = ModelClass(cfg.model.hparams)
            data_module = TimeSeriesDataModule(
                data=modified_data,
                hparams=cfg.model.hparams,
                sequence_method=cfg.model.sequence_method  # Ensure this parameter is set in your config
            )
            model.data = modified_data  
            trainer = pl.Trainer(logger=wandb_logger, **cfg.model.trainer)

            # Train the model
            trainer.fit(model, datamodule=data_module)
            print("Model training completed")

            # Test the model
            trainer.test(model, datamodule=data_module)
            print("Model testing completed")

            # Perform inference to detect anomalies
            detected_anomalies = model.inference()

    # Evaluate the detection results
    evaluation_report = evaluate_detection(detected_anomalies, anomalies, modified_data,data, wandb_logger, cfg.experiment.results_path)
    wandb_logger.experiment.log({"Evaluation Report": evaluation_report['multivariate_novelty']})
    print("Evaluation Report:", evaluation_report)
    wandb_logger.experiment.finish()
    wandb.finish()

    os.makedirs("results", exist_ok=True)

    experiment_name = cfg.experiment.name
    # with open(f"results/{experiment_name}_evaluation_report.json", "w") as f:
    #     json.dump(evaluation_report, f)

if __name__ == "__main__":
    run_experiment()
