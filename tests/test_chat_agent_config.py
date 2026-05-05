from __future__ import annotations

from unittest.mock import patch

import pytest

from agent import chat_agent
from agent.chat_agent import Provider, ProviderConfig, ProviderConfigurationError


@pytest.fixture(autouse=True)
def clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "AZURE_FOUNDRY_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_API_KEY",
        "AZURE_FOUNDRY_ENDPOINT",
        "AZURE_ENDPOINT",
        "AZURE_API_VERSION",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def test_configured_providers_only_returns_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("AZURE_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_ENDPOINT", "https://example.azure.com")

    providers = chat_agent.configured_providers()

    assert providers == [Provider.OPENAI, Provider.AZURE_FOUNDRY]


def test_providers_for_ui_returns_all_when_none_configured() -> None:
    providers = chat_agent.providers_for_ui()
    assert providers == list(Provider)


def test_resolve_api_key_prefers_explicit_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-value")
    config = ProviderConfig(provider=Provider.OPENAI, model_name="gpt-4o", api_key="explicit")

    assert chat_agent._resolve_api_key(config) == "explicit"


def test_resolve_api_key_raises_when_missing() -> None:
    config = ProviderConfig(provider=Provider.GOOGLE, model_name="gemini-2.5-flash")
    with pytest.raises(ProviderConfigurationError):
        chat_agent._resolve_api_key(config)


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("https://x.cognitiveservices.azure.com", "https://x.cognitiveservices.azure.com/openai/v1"),
        ("https://x.cognitiveservices.azure.com/openai", "https://x.cognitiveservices.azure.com/openai/v1"),
        ("https://x.cognitiveservices.azure.com/openai/v1", "https://x.cognitiveservices.azure.com/openai/v1"),
    ],
)
def test_normalize_azure_base_url(endpoint: str, expected: str) -> None:
    assert chat_agent._normalize_azure_base_url(endpoint) == expected


def test_discover_azure_deployments_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "data": [
                    {"id": "dep-a"},
                    {"name": "dep-b"},
                    {"deployment_name": "dep-c"},
                    {"id": "dep-a"},
                ]
            }

    monkeypatch.setenv("AZURE_API_VERSION", "2025-01-01-preview")

    with patch("httpx.get", return_value=FakeResponse()):
        deployments = chat_agent._discover_azure_deployments("https://example.azure.com", "key")

    assert deployments == ["dep-a", "dep-b", "dep-c"]


def test_discover_models_azure_prefers_deployments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_agent, "_resolve_api_key", lambda _config: "key")
    monkeypatch.setattr(chat_agent, "_resolve_endpoint", lambda _config: "https://example.azure.com")
    monkeypatch.setattr(chat_agent, "_discover_azure_deployments", lambda _e, _k: ["chat-prod", "chat-dev"])

    models = chat_agent.discover_models(Provider.AZURE_FOUNDRY)

    assert set(models) == {"chat-dev", "chat-prod"}


def test_discover_models_fallbacks_to_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_agent, "_resolve_api_key", lambda _config: (_ for _ in ()).throw(RuntimeError("boom")))

    models = chat_agent.discover_models(Provider.OPENAI)

    assert models == chat_agent.PROVIDER_MODEL_CATALOG[Provider.OPENAI]
