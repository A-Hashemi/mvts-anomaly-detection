from typing import Any, Dict

import numpy as np
import pandas as pd

from .registry import AnomalyInjectionRegistry

def generate_anomalies(cfg: Dict[str, Any], data: pd.DataFrame, logger=None) -> Dict[str, Any]:
    """
    Generate anomalies based on the provided configuration and dataset using registered anomaly generators.

    This function generates both symptom-wise and systemic anomalies according to the
    configuration. It uses the AnomalyInjectionRegistry to instantiate the appropriate
    anomaly generators based on the configuration, including scenario-specific SIR models.

    Args:
        cfg (Dict[str, Any]): A dictionary containing the configuration for anomaly generation.
                              It should have 'dataset' as its top-level key, with 'symptom_anomalies'
                              and 'systemic_anomalies' as sub-keys.
        data (pd.DataFrame): The input dataset on which to generate anomalies.

    Returns:
        Dict[str, Any]: A dictionary of generated anomalies. The keys are column names,
                        and the values are dictionaries containing different types of
                        anomalies for each column.

    Raises:
        KeyError: If required keys are missing from the configuration dictionary.
    """
    anomalies: Dict[str, Any] = {}

    if cfg['dataset']['symptom_anomalies']['enabled']:
        for column, column_cfg in cfg['dataset']['symptom_anomalies']['columns'].items():
            column_anomalies: Dict[str, Any] = {}
            
            for anomaly_type, anomaly_cfg in column_cfg.items():
                if anomaly_cfg['enabled']:
                    generator_class = AnomalyInjectionRegistry.get(anomaly_type)
                    if generator_class:
                        generator = generator_class(data[column], anomaly_cfg)
                        column_anomalies[anomaly_type] = generator.generate()
                    else:
                        print(f"Warning: No registered generator found for anomaly type '{anomaly_type}'")
            
            if column_anomalies:
                anomalies[column] = column_anomalies

    if cfg['dataset']['systemic_anomalies']['enabled']:
        systemic_cfg = cfg['dataset']['systemic_anomalies']
        sir_model = systemic_cfg['model']
        sir_params = systemic_cfg['params']
        injection_start_index = systemic_cfg['injection_start_index']

        generator_class = AnomalyInjectionRegistry.get(sir_model)
        if generator_class:
            generator = generator_class(data, sir_params, injection_start_index, logger=logger)
            systemic_anomalies = generator.generate()
            
            # Merge systemic anomalies with existing anomalies
            for column, anomaly_data in systemic_anomalies.items():
                if column in anomalies:
                    anomalies[column][sir_model] = anomaly_data
                else:
                    anomalies[column] = {sir_model: anomaly_data}
        else:
            print(f"Warning: No registered generator found for SIR model '{sir_model}'")

    return anomalies




def inject_anomalies(data: pd.DataFrame, anomalies: Dict[str, Any]) -> pd.DataFrame:
    """
    Inject generated anomalies into the original dataset.

    This function takes the original dataset and the dictionary of generated anomalies,
    and injects the anomalies into the dataset at the appropriate dates or indices.

    Args:
        data (pd.DataFrame): The original dataset.
        anomalies (Dict[str, Any]): A dictionary of generated anomalies. The keys are column names,
                                    and the values are dictionaries containing different types of
                                    anomalies for each column.

    Returns:
        pd.DataFrame: A new DataFrame with the injected anomalies.

    Note:
        This function handles 'sir_mpox', 'point', 'contextual', 'collective', and 'novelty' type anomalies.
    """
    modified_data = data.copy()
    anomaly_info = {}

    for column, anomaly_types in anomalies.items():
        anomaly_info[column] = {}
        for anomaly_type, anomaly_data in anomaly_types.items():
            if anomaly_type in ['sir_mpox', 'sir_measles']:
                indices = anomaly_data['indices']
                values = anomaly_data['values']
                anomaly_info[column][anomaly_type] = {'indices': indices, 'values': values}

                if column not in modified_data.columns:
                    raise ValueError(f"Column '{column}' is not in the data")

                for idx, value in zip(indices, values):
                    if idx in modified_data.index:
                        modified_data.loc[idx, column] += value
                    else:
                        raise ValueError(f"Index {idx} is not in the data")
                    
            elif anomaly_type == 'point':
                for idx, value in anomaly_data:
                    if idx in modified_data.index:
                        modified_data.loc[idx, column] = value

            elif anomaly_type == 'contextual':
                for idx, anomaly_value, _ in anomaly_data:
                    if idx in modified_data.index:
                        modified_data.loc[idx, column] = anomaly_value

            elif anomaly_type == 'collective':
                for start_idx, end_idx, _, modified_subsequence in anomaly_data:
                    if start_idx in modified_data.index and end_idx in modified_data.index:
                        modified_data.loc[start_idx:end_idx, column] = modified_subsequence

            elif anomaly_type == 'novelty':
                start_idx = anomaly_data['novelty_info']['start_idx']
                end_idx = anomaly_data['novelty_info']['end_idx']
                novelty_data = anomaly_data['novelty_data']
                
                if start_idx in modified_data.index and end_idx in modified_data.index:
                    modified_data.loc[start_idx:, column] = novelty_data[start_idx:]

            else:
                print(f"Warning: Unknown anomaly type '{anomaly_type}' for column '{column}'")

    return modified_data, anomaly_info