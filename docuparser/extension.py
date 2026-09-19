from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type
from docuparser.context import ParsedDocument
from docuparser.llm.base import BaseLLMClient


class BaseExtension(ABC):
    """Abstract Base Class for all pipeline extensions and plugins."""

    name: str = "base_extension"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def run(self, doc: ParsedDocument, llm: Optional[BaseLLMClient] = None) -> ParsedDocument:
        """Transforms or enriches the ParsedDocument state."""
        pass


class ExtensionRegistry:
    """Registry allowing extensions to be registered and looked up dynamically."""

    _registry: Dict[str, Type[BaseExtension]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(subclass: Type[BaseExtension]):
            cls._registry[name] = subclass
            subclass.name = name
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseExtension]:
        if name not in cls._registry:
            raise KeyError(
                f"Extension '{name}' is not registered. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> List[str]:
        return list(cls._registry.keys())