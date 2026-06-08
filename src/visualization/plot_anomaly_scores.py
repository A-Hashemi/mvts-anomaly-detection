import os
from typing import Union

import numpy as np
import plotly.graph_objs as go
import wandb


def plot_anomaly_scores(anomaly_scores: np.ndarray, anomaly_indices: np.ndarray, logger: Union[wandb.sdk.wandb_run.Run, object]) -> None:
    """
    Plot the anomaly scores and highlight the detected anomalies, then log the plot using Weights & Biases (wandb).

    Args:
        anomaly_scores (np.ndarray): Array of anomaly scores over time.
        anomaly_indices (np.ndarray): Array of indices indicating the detected anomalies.
        logger (Union[wandb.sdk.wandb_run.Run, object]): The logger used for logging plots to wandb. Must have an 'experiment' attribute for logging.

    Returns:
        None
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
        title='Anomaly Scores with Detected Anomalies (STUMPY)',
        xaxis_title='Time',
        yaxis_title='Anomaly Score',
        hovermode='closest',
        legend_title="Legend"
    )

    output_dir = logger.save_dir if hasattr(logger, 'save_dir') else './wandb_logs'
    os.makedirs(output_dir, exist_ok=True)
    plotly_html_path = os.path.join(output_dir, 'stumpy_anomaly_scores_plot.html')
    fig.write_html(plotly_html_path, auto_play=False)

    table = wandb.Table(columns=["Anomaly Scores Plot"])
    table.add_data(wandb.Html(plotly_html_path))

    logger.experiment.log({"STUMPY Anomaly Scores Plot": table})

    data = [[x, y] for (x, y) in zip(time_axis, anomaly_scores)]
    line_table = wandb.Table(data=data, columns=["Time", "Anomaly Score"])

    logger.experiment.log(
        {
            "STUMPY Anomaly Scores Line Plot": wandb.plot.line(
                line_table, "Time", "Anomaly Score", title="STUMPY Anomaly Scores Over Time"
            )
        }
    )