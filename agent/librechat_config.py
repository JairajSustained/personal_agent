from __future__ import annotations

import os
from pathlib import Path

import yaml

from .chat_agent import Provider


def _candidate_paths() -> list[Path]:
    explicit = (os.getenv("LIBRECHAT_CONFIG_PATH") or "").strip()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())

    project_local = Path.cwd() / "librechat.yaml"
    candidates.append(project_local)

    return candidates


def find_librechat_config_path() -> Path | None:
    for candidate in _candidate_paths():
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _extract_azure_models(config: dict) -> list[str]:
    endpoints = config.get("endpoints", {}) if isinstance(config, dict) else {}
    azure = endpoints.get("azureOpenAI", {}) if isinstance(endpoints, dict) else {}

    deployment_names: set[str] = set()

    groups = azure.get("groups", []) if isinstance(azure, dict) else []
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, dict):
                continue
            models = group.get("models", {})
            if not isinstance(models, dict):
                continue

            for model_key, model_cfg in models.items():
                if isinstance(model_cfg, dict):
                    deployment_name = model_cfg.get("deploymentName")
                    if isinstance(deployment_name, str) and deployment_name.strip():
                        deployment_names.add(deployment_name.strip())
                        continue

                if isinstance(model_key, str) and model_key.strip():
                    deployment_names.add(model_key.strip())

    return sorted(deployment_names)


def _extract_simple_models(config: dict, key: str) -> list[str]:
    endpoints = config.get("endpoints", {}) if isinstance(config, dict) else {}
    provider = endpoints.get(key, {}) if isinstance(endpoints, dict) else {}
    if not isinstance(provider, dict):
        return []

    models = provider.get("models", [])
    if isinstance(models, list):
        return sorted({m for m in models if isinstance(m, str) and m.strip()})
    if isinstance(models, dict):
        return sorted({m for m in models.keys() if isinstance(m, str) and m.strip()})
    return []


def load_librechat_models() -> dict[Provider, list[str]]:
    path = find_librechat_config_path()
    if path is None:
        return {}

    with path.open("r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle) or {}

    mapping: dict[Provider, list[str]] = {
        Provider.AZURE_FOUNDRY: _extract_azure_models(config),
        Provider.OPENAI: _extract_simple_models(config, "openAI"),
        Provider.ANTHROPIC: _extract_simple_models(config, "anthropic"),
        Provider.GOOGLE: _extract_simple_models(config, "google"),
    }

    return {provider: models for provider, models in mapping.items() if models}
