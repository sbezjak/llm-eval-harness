import httpx
import pytest
import respx

from eval_harness.providers.ollama import OllamaProvider


@pytest.mark.mocked
@respx.mock
async def test_ollama_provider_generate_returns_response_field():
    route = respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(200, json={"response": "hello from mock", "done": True})
    )

    provider = OllamaProvider(model="llama3.2")
    out = await provider.generate("ping")

    assert out == "hello from mock"
    assert route.called
    sent = route.calls.last.request
    body = sent.read().decode()
    assert '"model":"llama3.2"' in body
    assert '"stream":false' in body


@pytest.mark.ollama
async def test_ollama_provider_hits_real_backend():
    """Smoke test against a live Ollama. Hard-fails if backend is unreachable —
    selecting `-m ollama` is an explicit assertion that Ollama is running."""
    provider = OllamaProvider(model="llama3.2", timeout=30.0)
    out = await provider.generate("Say the single word: pong")
    assert isinstance(out, str)
    assert out.strip() != ""
