import importlib
import os
from typing import Any, Callable, Dict


class AnomalyEvaluationRegistry:
    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def decorator(func: Callable) -> Callable:
            cls._registry[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str) -> Callable:
        return cls._registry.get(name)

    @classmethod
    def list_methods(cls) -> Dict[str, Callable]:
        return cls._registry

    @classmethod
    def load_evaluators(cls, package_name: str = 'src.evaluation'):
        package = importlib.import_module(package_name)
        path = os.path.dirname(package.__file__)
        for module in os.listdir(path):
            if module.endswith('.py') and not module.startswith('__'):
                importlib.import_module(f'{package_name}.{module[:-3]}')
