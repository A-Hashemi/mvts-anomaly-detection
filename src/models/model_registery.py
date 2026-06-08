from typing import Dict, Type

from src.models.base_reconstruction_model import ReconstructionAnomalyDetector


class ModelRegistry:
    """
    A registry to manage and access different deep learning models.

    This class provides a way to register and retrieve model classes by name.
    It allows models to be dynamically accessed based on their registration names.

    Attributes:
        _models (Dict[str, Type[ReconstructionAnomalyDetector]]): A dictionary mapping model names to their respective classes.
    """
    
    _models: Dict[str, Type[ReconstructionAnomalyDetector]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Registers a model class in the registry under a specific name.

        Args:
            name (str): The name under which to register the model.

        Returns:
            Callable: A decorator that registers the model.
        """
        def decorator(model: Type[ReconstructionAnomalyDetector]):
            """
            The decorator function that adds the model to the registry.

            Args:
                model (Type[ReconstructionAnomalyDetector]): The model class to register.

            Returns:
                Type[ReconstructionAnomalyDetector]: The registered model class.
            """
            cls._models[name] = model
            return model
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[ReconstructionAnomalyDetector]:
        """
        Retrieves a registered model class by name.

        Args:
            name (str): The name of the registered model to retrieve.

        Returns:
            Type[ReconstructionAnomalyDetector]: The model class corresponding to the given name.

        Raises:
            ValueError: If the model with the given name is not found in the registry.
        """
        model = cls._models.get(name)
        if model is None:
            raise ValueError(f"Model {name} not found in registry")
        return model
