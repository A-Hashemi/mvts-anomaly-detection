from typing import Type, Dict, Callable, List, Optional

class AnomalyInjectionRegistry:
    """
    A registry to manage different types of anomaly injection classes.
    
    This class uses a registry pattern to keep track of different classes 
    associated with a given name. It allows for dynamic retrieval of registered 
    classes based on their name.
    """
    
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        """
        Registers a class with a given name in the registry.

        Args:
            name (str): The name to register the class under.

        Returns:
            Callable[[Type], Type]: A decorator function that registers the class.
        """
        def decorator(registered_class: Type) -> Type:
            cls._registry[name] = registered_class
            return registered_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """
        Retrieves a class from the registry by name.

        Args:
            name (str): The name of the registered class to retrieve.

        Returns:
            Optional[Type]: The class associated with the given name, or None if not found.
        """
        return cls._registry.get(name)

    @classmethod
    def list(cls) -> List[str]:
        """
        Lists all registered class names.

        Returns:
            List[str]: A list of all registered names in the registry.
        """
        return list(cls._registry.keys())
