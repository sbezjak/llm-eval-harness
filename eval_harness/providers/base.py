from abc import ABC, abstractmethod


class Provider(ABC):
    """Abstract LLM backend.

    Concrete providers are constructed with their own config (model name,
    base URL, timeouts, sampling params) and expose a single async call
    that maps a prompt to a completion string. Keeping the interface this
    narrow lets scorers and tests treat any backend identically.
    """

    @abstractmethod
    async def generate(self, prompt: str) -> str: ...
