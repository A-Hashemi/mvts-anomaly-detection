import pickle
from pathlib import Path
from typing import Any, Dict


def save_evaluation_data(
    detected_anomalies: Dict,
    true_anomalies: Dict,
    modified_data: Dict,
    evaluation_results: Dict,
    original_data: Dict,
    save_path: str
) -> None:
    """
    Save anomaly detection results using pickle.
    
    Args:
        detected_anomalies: Dictionary containing anomaly scores and indices
        true_anomalies: Dictionary containing ground truth anomalies
        evaluation_results: Dictionary containing evaluation metrics
        save_path: Path to save the pickle file
    """
    data = {
        'detected_anomalies': detected_anomalies,
        'true_anomalies': true_anomalies,
        'evaluation_results': evaluation_results,
        'modified_data': modified_data,
        'original_data': original_data
    }
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
        print("saved")

def load_evaluation_data(load_path: str) -> Dict[str, Any]:
    """
    Load saved anomaly detection results.
    
    Args:
        load_path: Path to the saved pickle file
        
    Returns:
        Dictionary containing the loaded data
    """
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    return data