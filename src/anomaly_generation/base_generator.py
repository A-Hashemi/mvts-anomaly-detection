from abc import ABC, abstractmethod

import pandas as pd


class BaseAnomalyGenerator(ABC):
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    @abstractmethod
    def generate(self, column_name: str, **kwargs):
        pass