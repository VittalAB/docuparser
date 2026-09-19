from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMClient(ABC):
    """Abstract interface defining the contract for any LLM client."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates text from a given prompt and optional system instruction."""
        pass