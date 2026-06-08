from abc import ABC, abstractmethod

class BaseAnomalyEvaluator(ABC):
    @abstractmethod
    def evaluate(self, true_anomalies, pred_anomalies, **kwargs):
        pass