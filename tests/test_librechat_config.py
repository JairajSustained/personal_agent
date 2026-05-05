from __future__ import annotations

from pathlib import Path

from agent import Provider, librechat_config


def test_load_librechat_models_extracts_azure_deployments(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "librechat.yaml"
    cfg.write_text(
        """
endpoints:
  azureOpenAI:
    groups:
      - group: foundry
        models:
          gpt-4o:
            deploymentName: chat-prod
          gpt-4o-mini:
            deploymentName: chat-dev
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("LIBRECHAT_CONFIG_PATH", str(cfg))

    models = librechat_config.load_librechat_models()

    assert models[Provider.AZURE_FOUNDRY] == ["chat-dev", "chat-prod"]


def test_load_librechat_models_returns_empty_when_missing(monkeypatch) -> None:
    monkeypatch.setenv("LIBRECHAT_CONFIG_PATH", "/tmp/does-not-exist.yaml")
    assert librechat_config.load_librechat_models() == {}


def test_load_librechat_models_reads_non_azure_providers(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "librechat.yaml"
    cfg.write_text(
        """
endpoints:
  openAI:
    models:
      - gpt-4o
      - gpt-4o-mini
  anthropic:
    models:
      claude-sonnet-4-6: {}
  google:
    models:
      - gemini-2.5-flash
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("LIBRECHAT_CONFIG_PATH", str(cfg))

    models = librechat_config.load_librechat_models()

    assert models[Provider.OPENAI] == ["gpt-4o", "gpt-4o-mini"]
    assert models[Provider.ANTHROPIC] == ["claude-sonnet-4-6"]
    assert models[Provider.GOOGLE] == ["gemini-2.5-flash"]


def test_find_librechat_config_prefers_explicit_env_path(tmp_path: Path, monkeypatch) -> None:
    env_cfg = tmp_path / "env-librechat.yaml"
    env_cfg.write_text("endpoints: {}\n", encoding="utf-8")

    cwd = tmp_path / "project"
    cwd.mkdir()
    (cwd / "librechat.yaml").write_text("endpoints: {openAI: {models: [gpt-5]}}\n", encoding="utf-8")

    monkeypatch.chdir(cwd)
    monkeypatch.setenv("LIBRECHAT_CONFIG_PATH", str(env_cfg))

    selected = librechat_config.find_librechat_config_path()
    assert selected == env_cfg
