from __future__ import annotations

import os

import pytest
from PySide6.QtWidgets import QApplication

from agent.chat_agent import Provider
from gui import chat_window


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_settings_panel_shows_only_configured_providers(monkeypatch: pytest.MonkeyPatch, qapp: QApplication) -> None:
    monkeypatch.setattr(chat_window, "providers_for_ui", lambda: [Provider.AZURE_FOUNDRY])
    monkeypatch.setattr(chat_window, "discover_models", lambda **_kwargs: ["dep-1"])

    panel = chat_window.SettingsPanel()

    assert panel.provider_combo.count() == 1
    assert panel.provider_combo.itemText(0) == Provider.AZURE_FOUNDRY.value


def test_settings_panel_toggles_endpoint_input(monkeypatch: pytest.MonkeyPatch, qapp: QApplication) -> None:
    monkeypatch.setattr(chat_window, "providers_for_ui", lambda: [Provider.OPENAI, Provider.AZURE_FOUNDRY])
    monkeypatch.setattr(chat_window, "discover_models", lambda **_kwargs: ["model-a"])

    panel = chat_window.SettingsPanel()

    panel._on_provider_changed(Provider.OPENAI.value)
    assert panel.endpoint_input.isEnabled() is False

    panel._on_provider_changed(Provider.AZURE_FOUNDRY.value)
    assert panel.endpoint_input.isEnabled() is True
    assert "azure deployments" in panel.hint_label.text().lower()


def test_settings_panel_reads_env_hints(monkeypatch: pytest.MonkeyPatch, qapp: QApplication) -> None:
    monkeypatch.setattr(chat_window, "providers_for_ui", lambda: [Provider.AZURE_FOUNDRY])
    monkeypatch.setattr(chat_window, "discover_models", lambda **_kwargs: ["dep-1"])
    monkeypatch.delenv("AZURE_FOUNDRY_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_ENDPOINT", "https://example.azure.com")
    monkeypatch.setenv("AZURE_API_KEY", "abc")

    panel = chat_window.SettingsPanel()

    assert panel.endpoint_input.text() == "https://example.azure.com"
    assert "Loaded from .env" in panel.api_key_input.placeholderText()


def test_settings_panel_uses_yaml_models_for_azure_when_runtime_empty(
    monkeypatch: pytest.MonkeyPatch, qapp: QApplication
) -> None:
    monkeypatch.setattr(chat_window, "providers_for_ui", lambda: [Provider.AZURE_FOUNDRY])
    monkeypatch.setattr(chat_window, "discover_models", lambda **_kwargs: [])
    monkeypatch.setattr(chat_window, "load_librechat_models", lambda: {Provider.AZURE_FOUNDRY: ["chat-prod"]})

    panel = chat_window.SettingsPanel()

    assert panel.model_combo.count() == 1
    assert panel.model_combo.itemText(0) == "chat-prod"


def test_settings_panel_uses_static_catalog_for_non_azure_when_no_runtime_or_yaml(
    monkeypatch: pytest.MonkeyPatch, qapp: QApplication
) -> None:
    monkeypatch.setattr(chat_window, "providers_for_ui", lambda: [Provider.OPENAI])
    monkeypatch.setattr(chat_window, "discover_models", lambda **_kwargs: [])
    monkeypatch.setattr(chat_window, "load_librechat_models", lambda: {})

    panel = chat_window.SettingsPanel()

    expected = chat_window.PROVIDER_MODEL_CATALOG[Provider.OPENAI]
    actual = [panel.model_combo.itemText(i) for i in range(panel.model_combo.count())]
    assert actual == expected
