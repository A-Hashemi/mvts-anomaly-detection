from typing import Callable, Dict


class DetectorRegistry:
    """
    A registry to manage and access different detector functions or classes.

    This class allows detectors (functions or classes) to be registered by name 
    and provides a method to retrieve the registered detectors.

    Attributes:
        _detectors (Dict[str, Callable]): A dictionary mapping detector names to their respective callables.
    """
    
    _detectors: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        """
        Registers a detector function or class in the registry under a specific name.

        Args:
            name (str): The name under which to register the detector.

        Returns:
            Callable: A decorator that registers the detector.
        """
        def decorator(detector: Callable):
            """
            The decorator function that adds the detector to the registry.

            Args:
                detector (Callable): The detector function or class to register.

            Returns:
                Callable: The registered detector function or class.
            """
            cls._detectors[name] = detector
            return detector
        return decorator

    @classmethod
    def get(cls, name: str) -> Callable:
        """
        Retrieves a registered detector function or class by name.

        Args:
            name (str): The name of the registered detector to retrieve.

        Returns:
            Callable: The detector function or class corresponding to the given name.

        Raises:
            ValueError: If the detector with the given name is not found in the registry.
        """
        detector = cls._detectors.get(name)
        if detector is None:
            raise ValueError(f"Detector {name} not found in registry")
        return detector
