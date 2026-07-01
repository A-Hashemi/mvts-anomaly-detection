from typing import Any, Dict

import numpy as np
import pandas as pd

from .registry import AnomalyEvaluationRegistry

from src.utils.tools import save_evaluation_data

AnomalyEvaluationRegistry.load_evaluators()

# def evaluate_detection(detected_anomalies: Dict[str, Dict[str, Any]], true_anomalies: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
#     """
#     Evaluate the detected anomalies against  the true anomalies for each column and anomaly type.

#     Args:
#         detected_anomalies (Dict[str, Dict[str, Any]]): Detected anomalies for each column and type.
#         true_anomalies (Dict[str, Dict[str, Any]]): True anomalies for each column and type.

#     Returns:
#         Dict[str, Dict[str, Dict[str, float]]]: Evaluation results for each column and anomaly type.
#     """
#     results = {}

#     for column, column_anomalies in true_anomalies.items():
#         column_results = {}

#         for anomaly_type, true_anomalies_data in column_anomalies.items():
#             detected_anomalies_data = detected_anomalies.get(column, {}).get(anomaly_type, [])
            
#             evaluator = AnomalyEvaluationRegistry.get(anomaly_type)
#             if evaluator:
#                 evaluation = evaluator().evaluate(true_anomalies_data, detected_anomalies_data, verbose=False)
#                 column_results[anomaly_type] = evaluation

#         if column_results:
#             results[column] = column_results

#     return results


def evaluate_detection(
    detected_anomalies: Dict[str, Dict[str, np.ndarray]], 
    true_anomalies: Dict[str, Dict[str, Any]],
    modified_data: pd.DataFrame,
    original_data: pd.DataFrame,
    logger: None,
    results_path: str
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the detected anomalies against the true anomalies for the mpox scenario.

    Args:
        detected_anomalies (Dict[str, Dict[str, np.ndarray]]): Detected anomaly scores and indices.
        true_anomalies (Dict[str, Dict[str, Any]]): True anomalies for each symptom.
        modified_data (pd.DataFrame): The dataset with injected anomalies.
    Returns:
        Dict[str, Dict[str, float]]: Evaluation results for multivariate anomaly detection.
    """
    evaluator = AnomalyEvaluationRegistry.get('sir')()
    results = evaluator.evaluate(
        true_anomalies=true_anomalies,
        detected_anomalies=detected_anomalies,
        modified_data=modified_data,
        eps_values=[10, 20, 30, 40],  
        min_samples=1,
        logger=logger,  
        verbose=False
    )

    save_evaluation_data(
        detected_anomalies=detected_anomalies,
        true_anomalies=true_anomalies,
        evaluation_results=results,
        modified_data=modified_data,
        original_data=original_data,
        save_path=results_path
    )


    return {'multivariate_novelty': results}
