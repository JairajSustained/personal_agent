"""Personal Agent - Multi-provider chat agent using pydantic-ai."""

from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator
from enum import Enum
from threading import RLock

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

load_dotenv()


class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_FOUNDRY = "azure_foundry"


class ProviderConfig(BaseModel):
    """Runtime config for selecting and authenticating a model provider."""

    model_config = ConfigDict(use_enum_values=False)

    provider: Provider
    model_name: str = Field(min_length=1)
    api_key: str | None = None
    endpoint: str | None = None


PROVIDER_MODEL_CATALOG: dict[Provider, list[str]] = {
    Provider.OPENAI: ["gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-5-mini"],
    Provider.ANTHROPIC: ["claude-sonnet-4-6", "claude-opus-4-20250514", "claude-haiku-4-5-20251001"],
    Provider.GOOGLE: ["gemini-2.5-flash", "gemini-2.5-pro"],
    Provider.AZURE_FOUNDRY: ["gpt-4o-mini"],
}


class ProviderConfigurationError(ValueError):
    """Raised when a provider is missing required configuration."""


def _env_has_value(*names: str) -> bool:
    for name in names:
        if (os.getenv(name) or "").strip():
            return True
    return False


def configured_providers() -> list[Provider]:
    """Return providers that are currently configured via environment variables."""
    configured: list[Provider] = []

    if _env_has_value("OPENAI_API_KEY"):
        configured.append(Provider.OPENAI)
    if _env_has_value("ANTHROPIC_API_KEY"):
        configured.append(Provider.ANTHROPIC)
    if _env_has_value("GOOGLE_API_KEY"):
        configured.append(Provider.GOOGLE)
    if _env_has_value("AZURE_FOUNDRY_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_API_KEY") and _env_has_value(
        "AZURE_FOUNDRY_ENDPOINT", "AZURE_ENDPOINT"
    ):
        configured.append(Provider.AZURE_FOUNDRY)

    return configured


def providers_for_ui() -> list[Provider]:
    """Return configured providers, or all providers if none are configured."""
    configured = configured_providers()
    return configured if configured else list(Provider)


def _resolve_api_key(config: ProviderConfig) -> str:
    explicit_key = (config.api_key or "").strip()
    if explicit_key:
        return explicit_key

    env_lookup: dict[Provider, tuple[str, ...]] = {
        Provider.OPENAI: ("OPENAI_API_KEY",),
        Provider.ANTHROPIC: ("ANTHROPIC_API_KEY",),
        Provider.GOOGLE: ("GOOGLE_API_KEY",),
        Provider.AZURE_FOUNDRY: ("AZURE_FOUNDRY_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_API_KEY"),
    }
    for env_var in env_lookup[config.provider]:
        value = (os.getenv(env_var) or "").strip()
        if value:
            return value

    required_vars = " or ".join(env_lookup[config.provider])
    raise ProviderConfigurationError(
        f"Missing API key for provider '{config.provider.value}'. Set {required_vars} or provide api_key."
    )


def _resolve_endpoint(config: ProviderConfig) -> str | None:
    explicit_endpoint = (config.endpoint or "").strip()
    if explicit_endpoint:
        return explicit_endpoint

    if config.provider == Provider.AZURE_FOUNDRY:
        env_endpoint = (os.getenv("AZURE_FOUNDRY_ENDPOINT") or os.getenv("AZURE_ENDPOINT") or "").strip()
        if env_endpoint:
            return env_endpoint

        raise ProviderConfigurationError(
            "Missing Azure Foundry endpoint. Set AZURE_FOUNDRY_ENDPOINT or AZURE_ENDPOINT, or provide endpoint."
        )

    return None


def _normalize_azure_base_url(endpoint: str) -> str:
    cleaned = endpoint.rstrip("/")
    if cleaned.endswith("/openai/v1"):
        return cleaned
    if cleaned.endswith("/openai"):
        return f"{cleaned}/v1"
    return f"{cleaned}/openai/v1"


def _discover_azure_deployments(endpoint: str, api_key: str) -> list[str]:
    import httpx

    api_version = (os.getenv("AZURE_API_VERSION") or "2025-01-01-preview").strip()
    response = httpx.get(
        f"{endpoint.rstrip('/')}/openai/deployments",
        params={"api-version": api_version},
        headers={"api-key": api_key},
        timeout=3.0,
    )
    response.raise_for_status()

    payload = response.json()
    items = payload.get("data", []) if isinstance(payload, dict) else []

    deployment_names: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        for candidate in ("id", "name", "deployment_name"):
            value = item.get(candidate)
            if isinstance(value, str) and value.strip():
                deployment_names.add(value.strip())
                break

    return sorted(deployment_names)


def _build_model(config: ProviderConfig):
    """Build a pydantic-ai model for the given provider."""
    key = _resolve_api_key(config)
    endpoint = _resolve_endpoint(config)

    match config.provider:
        case Provider.OPENAI:
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider

            return OpenAIChatModel(config.model_name, provider=OpenAIProvider(api_key=key))

        case Provider.ANTHROPIC:
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider

            return AnthropicModel(config.model_name, provider=AnthropicProvider(api_key=key))

        case Provider.GOOGLE:
            from pydantic_ai.models.google import GoogleModel
            from pydantic_ai.providers.google import GoogleProvider

            return GoogleModel(config.model_name, provider=GoogleProvider(api_key=key))

        case Provider.AZURE_FOUNDRY:
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider

            return OpenAIChatModel(
                config.model_name,
                provider=OpenAIProvider(api_key=key, base_url=_normalize_azure_base_url(endpoint)),
            )

    raise ProviderConfigurationError(f"Unsupported provider: {config.provider.value}")


def discover_models(
    provider: Provider,
    api_key: str | None = None,
    endpoint: str | None = None,
) -> list[str]:
    """Best-effort runtime model discovery per provider.

    Falls back to static catalog on API errors or missing credentials.
    """
    config = ProviderConfig(
        provider=provider,
        model_name=PROVIDER_MODEL_CATALOG.get(provider, ["placeholder"])[0],
        api_key=api_key,
        endpoint=endpoint,
    )

    try:
        key = _resolve_api_key(config)

        if provider == Provider.OPENAI:
            from openai import OpenAI

            client = OpenAI(api_key=key)
            return sorted({m.id for m in client.models.list().data if isinstance(m.id, str) and m.id})

        if provider == Provider.ANTHROPIC:
            from anthropic import Anthropic

            client = Anthropic(api_key=key)
            models = client.models.list()
            return sorted(
                {
                    model.id
                    for model in getattr(models, "data", [])
                    if hasattr(model, "id") and isinstance(model.id, str) and model.id
                }
            )

        if provider == Provider.GOOGLE:
            from google import genai

            client = genai.Client(api_key=key)
            discovered: list[str] = []
            for model in client.models.list():
                model_name = getattr(model, "name", "")
                if isinstance(model_name, str) and model_name.startswith("models/"):
                    discovered.append(model_name.removeprefix("models/"))
            return sorted(set(discovered))

        if provider == Provider.AZURE_FOUNDRY:
            from openai import OpenAI

            raw_endpoint = _resolve_endpoint(config)
            deployments = _discover_azure_deployments(raw_endpoint, key)
            if deployments:
                return deployments

            normalized = _normalize_azure_base_url(raw_endpoint)
            try:
                client = OpenAI(api_key=key, base_url=normalized)
                models = sorted({m.id for m in client.models.list().data if isinstance(m.id, str) and m.id})
                if models:
                    return models
            except Exception:
                pass

            try:
                import httpx

                api_version = (os.getenv("AZURE_API_VERSION") or "2025-01-01-preview").strip()
                response = httpx.get(
                    f"{raw_endpoint.rstrip('/')}/openai/models",
                    params={"api-version": api_version},
                    headers={"api-key": key},
                    timeout=3.0,
                )
                response.raise_for_status()
                payload = response.json()
                items = payload.get("data", []) if isinstance(payload, dict) else []
                models = sorted(
                    {
                        item.get("id")
                        for item in items
                        if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("id")
                    }
                )
                if models:
                    return models
            except Exception:
                pass

    except Exception:
        if provider == Provider.AZURE_FOUNDRY:
            return []
        return PROVIDER_MODEL_CATALOG.get(provider, [])

    if provider == Provider.AZURE_FOUNDRY:
        return []
    return PROVIDER_MODEL_CATALOG.get(provider, [])


class ChatAgent:
    """Chat agent supporting multiple providers via pydantic-ai."""

    def __init__(
        self,
        provider: Provider = Provider.OPENAI,
        model_name: str | None = None,
        instructions: str = "You are a helpful personal assistant.",
        api_key: str | None = None,
        endpoint: str | None = None,
    ):
        catalog = PROVIDER_MODEL_CATALOG.get(provider, [])
        default_model = catalog[0] if catalog else ""
        configured_model = model_name or default_model
        if not configured_model:
            raise ProviderConfigurationError(
                f"No model configured for provider '{provider.value}'. Please set model_name."
            )

        self.provider = provider
        self.model_name = configured_model
        self.instructions = instructions
        self._api_key = api_key
        self._endpoint = endpoint
        self._messages: list[ModelMessage] = []
        self._history_lock = RLock()

        self._agent = self._create_agent(
            provider=provider,
            model_name=self.model_name,
            api_key=api_key,
            endpoint=endpoint,
            instructions=instructions,
        )

    def _create_agent(
        self,
        provider: Provider,
        model_name: str,
        api_key: str | None,
        endpoint: str | None,
        instructions: str,
    ) -> Agent:
        config = ProviderConfig(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            endpoint=endpoint,
        )
        return Agent(model=_build_model(config), instructions=instructions)

    @staticmethod
    def _result_text(result) -> str:
        """Extract plain text from pydantic-ai run result."""
        response = getattr(result, "response", None)
        if response is None:
            return ""

        text_value = getattr(response, "text", "")
        if callable(text_value):
            text_value = text_value()
        return str(text_value or "")

    @staticmethod
    def _fallback_memory_update(current_memory: str, user_message: str) -> str:
        """Deterministic memory fallback when model-based update is unavailable."""
        existing = ChatAgent._normalize_memory_lines(current_memory)
        extracted = ChatAgent._extract_memory_candidates(user_message)
        merged = ChatAgent._merge_memory_lines(existing, extracted)
        if not merged:
            return ""
        return "\n".join(f"- {line}" for line in merged) + "\n"

    @staticmethod
    def _normalize_memory_lines(memory_text: str) -> list[str]:
        lines: list[str] = []
        for raw in memory_text.splitlines():
            line = raw.strip().lstrip("-").strip()
            if not line:
                continue

            lowered = line.lower()
            if "personal assistant" in lowered and ("chatgpt" in lowered or "you" in lowered):
                line = "Assistant Role: personal assistant"

            maybe_name = re.fullmatch(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})", line)
            if maybe_name:
                line = f"Name: {maybe_name.group(1)}"

            if ChatAgent._is_noise_memory_line(line):
                continue
            lines.append(line[:220])

        return ChatAgent._dedupe_lines(lines)

    @staticmethod
    def _is_noise_memory_line(line: str) -> bool:
        lowered = line.strip().lower()
        if not lowered:
            return True
        if lowered.endswith("?"):
            return True
        noise_prefixes = (
            "what is",
            "who is",
            "can you",
            "please remember",
            "remember this",
            "my name?",
            "okay, chatgpt",
        )
        return any(lowered.startswith(prefix) for prefix in noise_prefixes)

    @staticmethod
    def _extract_memory_candidates(user_message: str) -> list[str]:
        cleaned = " ".join(user_message.strip().split())
        if not cleaned:
            return []

        lowered = cleaned.lower()
        candidates: list[str] = []

        name_patterns = (
            r"\bmy name is\s+([A-Za-z][A-Za-z\s\-']{1,50})",
            r"\bi am\s+([A-Za-z][A-Za-z\s\-']{1,50})",
            r"\bcall me\s+([A-Za-z][A-Za-z\s\-']{1,50})",
        )
        for pattern in name_patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                raw_name = re.sub(r"\s+", " ", match.group(1)).strip(" .,!?")
                if raw_name:
                    title_name = " ".join(part.capitalize() for part in raw_name.split())
                    candidates.append(f"Name: {title_name}")
                break

        preference_markers = ("i prefer", "please use", "i like", "i don't like", "i do not like")
        if any(marker in lowered for marker in preference_markers):
            candidates.append(f"Preference: {cleaned[:180]}")

        if "you are" in lowered and "assistant" in lowered:
            candidates.append("Assistant Role: personal assistant")

        return ChatAgent._dedupe_lines([line for line in candidates if not ChatAgent._is_noise_memory_line(line)])

    @staticmethod
    def _dedupe_lines(lines: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for line in lines:
            key = line.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(line.strip())
        return deduped

    @staticmethod
    def _merge_memory_lines(existing: list[str], candidates: list[str]) -> list[str]:
        merged = list(existing)

        for candidate in candidates:
            if ChatAgent._is_noise_memory_line(candidate):
                continue

            if ":" in candidate:
                key = candidate.split(":", 1)[0].strip().lower()
                replaced = False
                for idx, line in enumerate(merged):
                    if ":" in line and line.split(":", 1)[0].strip().lower() == key:
                        merged[idx] = candidate
                        replaced = True
                        break
                if not replaced:
                    merged.append(candidate)
            elif candidate not in merged:
                merged.append(candidate)

        merged = ChatAgent._dedupe_lines(merged)
        if len(merged) > 80:
            merged = merged[-80:]
        return merged

    @staticmethod
    def _parse_model_memory_text(model_output: str) -> list[str]:
        raw_lines = [ln.strip().lstrip("-").strip() for ln in model_output.splitlines() if ln.strip()]
        parsed = [ln[:220] for ln in raw_lines if not ChatAgent._is_noise_memory_line(ln)]
        return ChatAgent._dedupe_lines(parsed)

    async def chat(self, message: str) -> str:
        """Send a message and get a response."""
        cleaned_message = message.strip()
        if not cleaned_message:
            raise ValueError("Message cannot be empty.")

        with self._history_lock:
            message_history = list(self._messages)

        result = await self._agent.run(cleaned_message, message_history=message_history)

        with self._history_lock:
            self._messages = result.all_messages()

        return self._result_text(result)

    async def chat_stream(self, message: str) -> AsyncIterator[str]:
        """Send a message and stream the response."""
        cleaned_message = message.strip()
        if not cleaned_message:
            raise ValueError("Message cannot be empty.")

        with self._history_lock:
            message_history = list(self._messages)

        async with self._agent.run_stream(cleaned_message, message_history=message_history) as result:
            async for chunk in result.stream_text(delta=True):
                yield chunk

            with self._history_lock:
                self._messages = result.all_messages()

    def reconfigure(
        self,
        provider: Provider,
        model_name: str,
        api_key: str | None = None,
        endpoint: str | None = None,
        instructions: str | None = None,
        validate_model: bool = True,
    ) -> None:
        """Reinitialize the backing model for a new provider/model selection."""
        self.provider = provider
        self.model_name = model_name
        self.instructions = instructions or self.instructions
        self._api_key = api_key
        self._endpoint = endpoint

        if validate_model and provider == Provider.AZURE_FOUNDRY:
            discovered = discover_models(provider=provider, api_key=api_key, endpoint=endpoint)
            if discovered and model_name not in discovered:
                raise ProviderConfigurationError(
                    "Selected Azure model is not a deployment in this resource. "
                    "Pick one from the dropdown refresh list."
                )

        self._agent = self._create_agent(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            endpoint=endpoint,
            instructions=self.instructions,
        )
        self.clear_history()

    def update_instructions(self, instructions: str) -> None:
        """Update instructions without clearing conversation history."""
        self.instructions = instructions
        self._agent = self._create_agent(
            provider=self.provider,
            model_name=self.model_name,
            api_key=self._api_key,
            endpoint=self._endpoint,
            instructions=instructions,
        )

    async def generate_title(self, first_user_message: str) -> str:
        """Generate a concise chat title from the first user message."""
        prompt = first_user_message.strip()
        if not prompt:
            return "New Chat"

        title_agent = self._create_agent(
            provider=self.provider,
            model_name=self.model_name,
            api_key=self._api_key,
            endpoint=self._endpoint,
            instructions=(
                "Generate a short chat title. Return only the title text. "
                "Max 8 words. No punctuation at start or end."
            ),
        )
        try:
            result = await title_agent.run(f"Create a title for this request: {prompt}")
            title = self._result_text(result).strip().strip('"').strip("'")
            return title[:64] if title else "New Chat"
        except Exception:
            return "New Chat"

    async def generate_memory_update(
        self,
        current_memory: str,
        user_message: str,
        assistant_message: str,
    ) -> str:
        """Generate an updated long-term memory text from a new chat turn."""
        base_existing = self._normalize_memory_lines(current_memory)
        extracted = self._extract_memory_candidates(user_message)

        memory_agent = self._create_agent(
            provider=self.provider,
            model_name=self.model_name,
            api_key=self._api_key,
            endpoint=self._endpoint,
            instructions=(
                "Extract durable memory only. Return one fact per line. "
                "Keep stable identity and preferences, avoid transient questions. "
                "Do not include prompts, retries, or generic chatter."
            ),
        )

        prompt = (
            "Current memory:\n"
            f"{current_memory or '(empty)'}\n\n"
            "Latest user message:\n"
            f"{user_message}\n\n"
            "Latest assistant response:\n"
            f"{assistant_message}\n\n"
            "Return updated memory text only."
        )
        try:
            result = await memory_agent.run(prompt)
            updated = self._result_text(result).strip()
            parsed_model = self._parse_model_memory_text(updated)
            merged = self._merge_memory_lines(base_existing, [*extracted, *parsed_model])
            if not merged:
                return ""
            return "\n".join(f"- {line}" for line in merged) + "\n"
        except Exception:
            return self._fallback_memory_update(current_memory, user_message)

    def get_available_models(self) -> list[str]:
        """Get available models for current provider."""
        return PROVIDER_MODEL_CATALOG.get(self.provider, [])

    def get_history_size(self) -> int:
        """Return number of messages currently cached in session history."""
        with self._history_lock:
            return len(self._messages)

    def export_history_json(self) -> str:
        """Serialize the current model message history for persistence."""
        with self._history_lock:
            return ModelMessagesTypeAdapter.dump_json(self._messages).decode("utf-8")

    def import_history_json(self, history_json: str) -> None:
        """Restore model message history from persisted JSON."""
        restored = ModelMessagesTypeAdapter.validate_json(history_json)
        with self._history_lock:
            self._messages = list(restored)

    def clear_history(self):
        """Clear conversation history."""
        with self._history_lock:
            self._messages = []
