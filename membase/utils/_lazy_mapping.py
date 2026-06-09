import importlib
from collections import OrderedDict
from typing import (
    Any, 
    Type, 
    Iterator,
)


class _LazyMapping(OrderedDict):
    """A mapping that lazily loads its values when they are requested.
    
    Its design is inspired by the lazy loading mechanism in Hugging Face Transformers.
    """
    
    def __init__(
        self, 
        mapping: OrderedDict[str, str], 
        module_mapping: dict[str, str],
        package: str,
    ) -> None:
        """Initialize the lazy mapping.
        
        Args:
            mapping (`OrderedDict[str, str]`): 
                A mapping from type keys to class names.
            module_mapping (`dict[str, str]`): 
                A mapping from type keys to relative module names within the
                `package`.  Keys absent from this mapping fall back to 
                `key.lower().replace("-", "_")`.
            package (`str`): 
                The package path passed to `importlib.import_module` for 
                relative imports (typically `__package__` of the caller).
        """
        self._mapping = mapping
        self._module_mapping = module_mapping
        self._package = package
        self._extra_content = {}
        self._modules = {}
    
    def _resolve_module_name(self, key: str) -> str:
        """Return the relative module name for the provided key."""
        return self._module_mapping.get(key, key.lower().replace("-", "_"))

    def __getitem__(self, key: str) -> Type[Any]:
        """Lazily load and return the class associated with the provided key."""
        if key in self._extra_content:
            return self._extra_content[key]
        
        if key not in self._mapping:
            raise KeyError(
                f"'{key}' is not found. Available keys are {list(self._mapping.keys())}."
            )
        
        class_name = self._mapping[key]
        module_name = self._resolve_module_name(key)
        
        # Cache the module if it is not already loaded. 
        if module_name not in self._modules:
            try:
                self._modules[module_name] = importlib.import_module(
                    f".{module_name}", 
                    self._package,
                )
            except ImportError as e:
                raise ImportError(
                    f"Failed to import {module_name} for {key}: {e}"
                )
        
        # Get the class from the module.
        if hasattr(self._modules[module_name], class_name):
            return getattr(self._modules[module_name], class_name)
        
        raise AttributeError(
            f"Module '{module_name}' does not have class '{class_name}'."
        )
    
    def keys(self) -> list[str]:
        """Return all available keys."""
        return list(self._mapping.keys()) + list(self._extra_content.keys())
    
    def values(self) -> list[Type[Any]]:
        """Return all values (load them if necessary)."""
        return [self[k] for k in self.keys()]
    
    def items(self) -> list[tuple[str, Type[Any]]]:
        """Return all key-value pairs."""
        return [(k, self[k]) for k in self.keys()]
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(self.keys())
    
    def __contains__(self, item: object) -> bool:
        """Check if a key exists in the mapping."""
        return item in self._mapping or item in self._extra_content
    
    def __len__(self) -> int:
        """Return the number of items in the mapping."""
        return len(self._mapping) + len(self._extra_content)
    
    def register(self, key: str, value: Type[Any], exist_ok: bool = False) -> None:
        """Register a new class in this mapping.
        
        Args:
            key (`str`): 
                The key to register the class under.
            value (`Type[Any]`): 
                The class to register.
            exist_ok (`bool`, defaults to `False`): 
                If enabled, it allows overwriting existing keys. Otherwise, 
                it raises an error.
        """
        if key in self._mapping and not exist_ok:
            raise ValueError(
                f"'{key}' is already registered. Use exist_ok=True to overwrite."
            )
        self._extra_content[key] = value
