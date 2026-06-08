from typing import Dict, List, Union

def evaluate_detection(true_anomalies: Dict[str, Dict[str, Union[List[Union[int, tuple]], Dict[str, int]]]], 
                       detected_anomalies: Dict[str, Dict[str, List[int]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluates the performance of anomaly detection by comparing true anomalies with detected anomalies.
    Calculates precision, recall, and F1-score for each type of anomaly in each column.

    Args:
        true_anomalies (Dict[str, Dict[str, Union[List[Union[int, tuple]], Dict[str, int]]]]): 
            A dictionary where keys are column names, and values are dictionaries of true anomalies by type.
            - If the anomaly type is 'novelty', the value is a dictionary with 'start_idx' and 'end_idx'.
            - For other types, it is a list of anomalies (either int or tuple with index).
        
        detected_anomalies (Dict[str, Dict[str, List[int]]]): 
            A dictionary where keys are column names, and values are dictionaries of detected anomalies by type.
            Each type contains a list of detected anomaly indices.

    Returns:
        Dict[str, Dict[str, Dict[str, float]]]: 
            A nested dictionary where keys are column names and values are dictionaries of anomaly type.
            Each anomaly type contains a dictionary with 'precision', 'recall', and 'f1_score'.
    """
    results = {}

    for column, column_anomalies in true_anomalies.items():
        column_results = {}

        for anomaly_type, true_anomalies_list in column_anomalies.items():
            detected_anomalies_for_type = detected_anomalies.get(column, {}).get(anomaly_type, [])
            all_true_anomalies = []

            if anomaly_type == 'novelty':
                all_true_anomalies.extend(range(true_anomalies_list['start_idx'], true_anomalies_list['end_idx']))
            else:
                all_true_anomalies = [a[0] if isinstance(a, tuple) else a for a in true_anomalies_list]

            true_positives = set(detected_anomalies_for_type).intersection(set(all_true_anomalies))
            false_positives = set(detected_anomalies_for_type) - set(all_true_anomalies)
            false_negatives = set(all_true_anomalies) - set(detected_anomalies_for_type)

            precision = len(true_positives) / (len(true_positives) + len(false_positives)) if len(detected_anomalies_for_type) > 0 else 0
            recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if len(all_true_anomalies) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            column_results[anomaly_type] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            }

        results[column] = column_results

    return results
