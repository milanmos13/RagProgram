"""The LLM client adapter: one prompt in, one completion out."""

import os
from typing import Protocol

import httpx


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class OpenAIClient:
    """Real LLM client over the OpenAI chat completions API."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        self._api_key = api_key or os.environ["OPENAI_API_KEY"]
        self._model = model

    def complete(self, prompt: str) -> str:
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
