"""Personal Agent - Multi-provider chat agent using pydantic-ai."""

from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

load_dotenv()


@dataclass
class AgentDeps:
    """Runtime dependencies injected into pydantic-ai tool calls."""

    memory_store: Any  # MemoryStore | None


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
        memory_store: Any = None,
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
        self._memory_store = memory_store
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
        include_tools: bool = True,
    ) -> Agent:
        config = ProviderConfig(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            endpoint=endpoint,
        )
        if include_tools:
            tool_instructions = (
                "\n\nYou have access to these tools:\n"
                "- save_memory(fact): Call this when the user shares personal info, preferences, or durable facts to remember across sessions. Always call this when the user says 'remember that', 'my name is', or shares preferences explicitly.\n"
                "- get_memory(query): Call this to look up specific things the user has shared previously.\n"
                "- search_web(query): Call this when the user asks about recent events, current data, prices, news, or anything that may have changed since your training cutoff.\n"
                "Call save_memory proactively whenever the user shares information worth remembering long-term."
            )
            return Agent(
                model=_build_model(config),
                instructions=f"{instructions}{tool_instructions}",
                tools=[save_memory, get_memory, search_web],
                deps_type=AgentDeps,
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
    def _clean_name_candidate(raw_name: str) -> str:
        """Normalize a captured name while rejecting obvious non-name phrases."""
        name = re.sub(r"\s+", " ", raw_name).strip(" .,!?;:")
        if not name:
            return ""

        # Stop at common clause boundaries so "my name is John Doe and I like X" stores only John Doe.
        name = re.split(
            r"\s+(?:and|but|because|so)\s+\b(?:i|my|you|we|they|he|she|it|use|prefer|like)\b",
            name,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" .,!?;:")

        words = name.split()
        if not words or len(words) > 4:
            return ""

        rejected_first_words = {
            "a",
            "an",
            "at",
            "from",
            "in",
            "into",
            "on",
            "the",
            "to",
            "with",
            "working",
            "using",
        }
        if words[0].lower() in rejected_first_words:
            return ""

        if any(any(char.isdigit() for char in word) for word in words):
            return ""

        return " ".join(part.capitalize() for part in words)

    @staticmethod
    def _extract_memory_candidates(user_message: str) -> list[str]:
        cleaned = " ".join(user_message.strip().split())
        if not cleaned:
            return []

        lowered = cleaned.lower()
        candidates: list[str] = []

        name_patterns = (
            r"\bmy name is\s+([^.!?;,]{1,80})",
            r"\bi am\s+([^.!?;,]{1,80})",
            r"\bcall me\s+([^.!?;,]{1,80})",
        )
        for pattern in name_patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                title_name = ChatAgent._clean_name_candidate(match.group(1))
                if title_name:
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

    async def chat(self, message: str) -> str:
        """Send a message and get a response."""
        cleaned_message = message.strip()
        if not cleaned_message:
            raise ValueError("Message cannot be empty.")

        with self._history_lock:
            message_history = list(self._messages)

        result = await self._agent.run(
            cleaned_message,
            message_history=message_history,
            deps=AgentDeps(memory_store=getattr(self, "_memory_store", None)),
        )

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

        async with self._agent.run_stream(
            cleaned_message,
            message_history=message_history,
            deps=AgentDeps(memory_store=getattr(self, "_memory_store", None)),
        ) as result:
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
            include_tools=False,
        )
        try:
            result = await title_agent.run(f"Create a title for this request: {prompt}")
            title = self._result_text(result).strip().strip('"').strip("'")
            return title[:64] if title else "New Chat"
        except Exception:
            return "New Chat"

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


async def save_memory(ctx: RunContext[AgentDeps], fact: str) -> str:
    """Save a durable fact about the user to long-term memory.

    Call this when the user shares personal information, preferences, or any fact
    that should persist across future conversations.
    """
    if ctx.deps.memory_store is None:
        return "Memory not available."
    current = ctx.deps.memory_store.load_text()
    existing = ChatAgent._normalize_memory_lines(current)
    new_lines = ChatAgent._normalize_memory_lines(fact)
    if not new_lines:
        return "Nothing to save — fact contained no usable memory lines."
    merged = ChatAgent._merge_memory_lines(existing, new_lines)
    text = "\n".join(f"- {line}" for line in merged) + "\n" if merged else ""
    ctx.deps.memory_store.save_text(text)
    return f"Saved: {new_lines[0][:60]}"


async def get_memory(ctx: RunContext[AgentDeps], query: str) -> str:
    """Retrieve relevant memory facts for the given query.

    Call this to look up specific things the user has shared in previous sessions.
    """
    if ctx.deps.memory_store is None:
        return "(memory not available)"
    result = ctx.deps.memory_store.load_relevant_text(query)
    return result.strip() or "(no relevant memory found)"


async def search_web(query: str) -> str:
    """Search the web for current information about a topic or question.

    Call this when the user asks about recent events, current data, or anything
    that may be outside the model's training knowledge.
    """
    import httpx

    try:
        params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
        async with httpx.AsyncClient(timeout=6.0) as client:
            response = await client.get("https://api.duckduckgo.com/", params=params)
            response.raise_for_status()
            data = response.json()

        parts: list[str] = []
        abstract = (data.get("AbstractText") or "").strip()
        if abstract:
            source = (data.get("AbstractSource") or "").strip()
            parts.append(f"{abstract} [{source}]" if source else abstract)

        for item in data.get("RelatedTopics", [])[:4]:
            if not isinstance(item, dict):
                continue
            text = (item.get("Text") or "").strip()
            url = (item.get("FirstURL") or "").strip()
            if text:
                parts.append(f"- {text}" + (f" ({url})" if url else ""))

        return "\n".join(parts) if parts else f"No results found for: {query}"
    except Exception as exc:  # noqa: BLE001
        return f"Search unavailable: {exc}"
