from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go


def plot_anomalies_interactive(original_data: pd.DataFrame,modified_data: pd.DataFrame, anomaly_info: Dict[str, Dict[str, Any]], features: List[str]) -> go.Figure:
    """
    Create an interactive Plotly figure showing the data and injected anomalies for specified features.

    Args:
        modified_data (pd.DataFrame): The data with injected anomalies.
        anomaly_info (Dict[str, Dict[str, Any]]): Information about the injected anomalies.
        features (List[str]): List of features to plot.

    Returns:
        go.Figure: The Plotly figure.
    """
    fig = go.Figure()

    for feature in features:
        if feature in modified_data.columns:
            # Plot the original data
            fig.add_trace(go.Scatter(
                x=original_data.index,
                y=original_data[feature],
                mode='lines',
                name=f'{feature} (Original Data)'
            ))

            fig.add_trace(go.Scatter(
                x=modified_data.index,
                y=modified_data[feature],
                mode='lines',
                name=f'{feature} (Modified Data)'
            ))

            # Check if there are anomalies for this feature
            if feature in anomaly_info:
                for anomaly_type, info in anomaly_info[feature].items():
                    if anomaly_type in ['sir_mpox', 'sir_measles']:
                        indices = info['indices']
                        values = modified_data.loc[indices, feature]

                        fig.add_trace(go.Scatter(
                            x=indices,
                            y=values,
                            mode='markers',
                            marker=dict(size=6),
                            name=f'{feature} ({anomaly_type})',
                            hoverinfo='x+y'
                        ))
    fig.update_layout(
        title='Data with Injected Anomalies',
        xaxis_title='Index',
        yaxis_title='Value',
        hovermode='closest'
    )
    return fig
