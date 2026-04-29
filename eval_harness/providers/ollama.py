from __future__ import annotations

import logging

import httpx

from eval_harness.providers.base import Provider

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised when the Ollama backend returns an error or unexpected payload."""


class OllamaProvider(Provider):
    """Ollama `/api/generate` adapter.

    Streaming is disabled (`stream=False`) so a single JSON response carries
    the full completion under the `response` key. Sampling params are passed
    through Ollama's `options` object — defaults here are conservative for
    eval reproducibility (low temperature), but callers can override.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    async def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        logger.debug("ollama.generate", extra={"model": self.model, "url": url})
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                raise OllamaError(f"Ollama request failed: {e}") from e

        data = resp.json()
        if "response" not in data:
            raise OllamaError(f"Ollama response missing 'response' key: {data!r}")
        return data["response"]
