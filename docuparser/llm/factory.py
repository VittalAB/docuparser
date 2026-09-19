from typing import Any, Callable, Dict, Optional, Union
from docuparser.llm.base import BaseLLMClient
from docuparser.llm.generic import OpenAICompatibleClient


class LangChainAdapter(BaseLLMClient):
    """
    Adapts any LangChain BaseChatModel (AzureChatOpenAI, ChatOpenAI, ChatGroq, etc.)
    to the Docuparser BaseLLMClient interface.
    """

    def __init__(self, langchain_model: Any):
        self.model = langchain_model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.model.invoke(messages)
        return str(getattr(response, "content", response))


class CallableAdapter(BaseLLMClient):
    """Adapts raw functions or callables: fn(prompt: str) -> str."""

    def __init__(self, fn: Callable[[str], str]):
        self.fn = fn

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        return str(self.fn(full_prompt))


def create_llm_client(
    client_or_config: Union[BaseLLMClient, Dict[str, Any], Any, Callable]
) -> BaseLLMClient:
    """
    Polymorphic factory resolving:
    1. BaseLLMClient instances
    2. Zero-arg factory functions returning an LLM (e.g. get_llm())
    3. LangChain chat models (AzureChatOpenAI, ChatOpenAI, etc.)
    4. Direct callables: fn(prompt: str) -> str
    5. OpenAI-compatible configuration dicts
    """
    if isinstance(client_or_config, BaseLLMClient):
        return client_or_config

    # Support factory functions that produce an LLM instance when called with 0 args
    if callable(client_or_config) and not hasattr(client_or_config, "invoke"):
        try:
            resolved = client_or_config()
            if callable(resolved) or hasattr(resolved, "invoke") or isinstance(resolved, BaseLLMClient):
                return create_llm_client(resolved)
        except TypeError:
            pass
        return CallableAdapter(client_or_config)

    # LangChain Chat Model Duck Typing (has an invoke method)
    if hasattr(client_or_config, "invoke") and callable(getattr(client_or_config, "invoke")):
        return LangChainAdapter(client_or_config)

    # Dict configuration
    if isinstance(client_or_config, dict):
        return OpenAICompatibleClient(
            base_url=client_or_config.get("base_url", "https://api.groq.com/openai/v1"),
            api_key=client_or_config.get("api_key"),
            model_name=client_or_config.get("model_name", "llama-3.3-70b-versatile"),
            temperature=client_or_config.get("temperature", 0.0),
            extra_headers=client_or_config.get("extra_headers"),
        )

    raise TypeError(
        f"Unsupported LLM type: {type(client_or_config)}. "
        "Expected LangChain model, callable, BaseLLMClient, or dict config."
    )