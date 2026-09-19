import os
import httpx
from typing import Optional, Dict
from docuparser.llm.base import BaseLLMClient


class OpenAICompatibleClient(BaseLLMClient):
    """
    Native client for any OpenAI-compatible endpoint:
    vLLM, Ollama, Groq, LM Studio, Mistral, OpenAI, Together, OpenRouter.
    """

    def __init__(
        self,
        base_url: str = "https://api.groq.com/openai/v1",
        api_key: Optional[str] = None,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY") or "EMPTY"
        self.model_name = model_name
        self.temperature = temperature
        self.extra_headers = extra_headers or {}
        self.timeout = timeout

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]