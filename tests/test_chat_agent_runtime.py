from __future__ import annotations

import asyncio
from pathlib import Path
from threading import RLock

import pytest

from agent import MemoryStore
from agent.chat_agent import (
    AgentDeps,
    ChatAgent,
    Provider,
    ProviderConfigurationError,
    get_memory,
    save_memory,
)


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeResult:
    def __init__(self, text, messages=None):
        self.response = _FakeResponse(text)
        self._messages = messages or []

    def all_messages(self):
        return self._messages


class _FakeStreamResult:
    def __init__(self, chunks, messages=None):
        self._chunks = list(chunks)
        self._messages = messages or []

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _tb):
        return False

    async def stream_text(self, *, delta):
        assert delta is True
        for chunk in self._chunks:
            yield chunk

    def all_messages(self):
        return self._messages


def test_result_text_supports_string_and_callable() -> None:
    result_str = _FakeResult("hello")
    assert ChatAgent._result_text(result_str) == "hello"

    result_callable = _FakeResult(lambda: "world")
    assert ChatAgent._result_text(result_callable) == "world"


def test_chat_uses_response_text_contract() -> None:
    class FakeAgentRuntime:
        async def run(self, _message, message_history, **_kwargs):
            assert message_history == []
            return _FakeResult("ok from response.text", messages=["m1"])

    agent = ChatAgent.__new__(ChatAgent)
    agent._agent = FakeAgentRuntime()
    agent._messages = []
    agent._history_lock = RLock()

    output = asyncio.run(ChatAgent.chat(agent, "hi"))
    assert output == "ok from response.text"
    assert agent._messages == ["m1"]


def test_chat_stream_yields_chunks_and_updates_messages() -> None:
    class FakeAgentRuntime:
        def run_stream(self, _message, message_history, **_kwargs):
            assert _message == "hi"
            assert message_history == []
            return _FakeStreamResult(["a", "b", "c"], messages=["m2"])

    async def _collect(agent: ChatAgent) -> list[str]:
        return [chunk async for chunk in ChatAgent.chat_stream(agent, "hi")]

    agent = ChatAgent.__new__(ChatAgent)
    agent._agent = FakeAgentRuntime()
    agent._messages = []
    agent._history_lock = RLock()

    chunks = asyncio.run(_collect(agent))
    assert chunks == ["a", "b", "c"]
    assert agent._messages == ["m2"]


def test_chat_and_chat_stream_raise_on_empty_message() -> None:
    class FakeAgentRuntime:
        async def run(self, _message, message_history, **_kwargs):
            assert message_history == []
            return _FakeResult("ignored")

        def run_stream(self, _message, message_history, **_kwargs):
            assert message_history == []
            return _FakeStreamResult([])

    async def _consume_empty_stream(agent: ChatAgent) -> list[str]:
        return [chunk async for chunk in ChatAgent.chat_stream(agent, "  ")]

    agent = ChatAgent.__new__(ChatAgent)
    agent._agent = FakeAgentRuntime()
    agent._messages = []
    agent._history_lock = RLock()

    with pytest.raises(ValueError):
        asyncio.run(ChatAgent.chat(agent, "   "))

    with pytest.raises(ValueError):
        asyncio.run(_consume_empty_stream(agent))


def test_generate_title_falls_back_on_provider_error() -> None:
    class FailingTitleAgent:
        async def run(self, _prompt):
            raise RuntimeError("content_filter")

    agent = ChatAgent.__new__(ChatAgent)
    agent.provider = Provider.OPENAI
    agent.model_name = "gpt-4o"
    agent._api_key = None
    agent._endpoint = None
    agent._create_agent = lambda **_kwargs: FailingTitleAgent()

    title = asyncio.run(ChatAgent.generate_title(agent, "my first message"))
    assert title == "New Chat"


class _FakeContext:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps


def test_save_memory_tool_writes_fact_to_store(tmp_path: Path) -> None:
    store = MemoryStore(file_path=tmp_path / "memory.txt")
    ctx = _FakeContext(deps=AgentDeps(memory_store=store))

    result = asyncio.run(save_memory(ctx, "Name: Alice"))

    assert "Alice" in result
    assert "Name: Alice" in store.load_text()


def test_save_memory_tool_merges_with_existing_facts(tmp_path: Path) -> None:
    store = MemoryStore(file_path=tmp_path / "memory.txt")
    store.save_text("- Preference: concise answers\n")
    ctx = _FakeContext(deps=AgentDeps(memory_store=store))

    asyncio.run(save_memory(ctx, "Name: Bob"))

    text = store.load_text()
    assert "Preference: concise answers" in text
    assert "Name: Bob" in text


def test_save_memory_tool_returns_message_when_no_store() -> None:
    ctx = _FakeContext(deps=AgentDeps(memory_store=None))

    result = asyncio.run(save_memory(ctx, "Name: Alice"))

    assert "not available" in result.lower()


def test_get_memory_tool_retrieves_relevant_text(tmp_path: Path) -> None:
    store = MemoryStore(file_path=tmp_path / "memory.txt")
    store.save_text("- Name: Alice\n- Preference: concise answers\n")
    ctx = _FakeContext(deps=AgentDeps(memory_store=store))

    result = asyncio.run(get_memory(ctx, "user's name"))

    assert "Alice" in result


def test_get_memory_tool_returns_message_when_no_store() -> None:
    ctx = _FakeContext(deps=AgentDeps(memory_store=None))

    result = asyncio.run(get_memory(ctx, "anything"))

    assert "not available" in result.lower()


def test_memory_extraction_keeps_name_separate_from_preferences() -> None:
    extracted = ChatAgent._extract_memory_candidates("My name is Alice and I like concise answers")

    assert "Name: Alice" in extracted
    assert "Name: Alice And I Like Concise Answers" not in extracted
    assert "Preference: My name is Alice and I like concise answers" in extracted


def test_memory_extraction_does_not_treat_location_as_name() -> None:
    extracted = ChatAgent._extract_memory_candidates("I am from Delhi and I prefer Hindi examples")

    assert all(not line.startswith("Name:") for line in extracted)
    assert "Preference: I am from Delhi and I prefer Hindi examples" in extracted


def test_normalize_memory_lines_filters_questions_and_keeps_name() -> None:
    memory = """
- what is my name?
- Alice Smith
- Please remember my name
"""
    normalized = ChatAgent._normalize_memory_lines(memory)
    assert "Name: Alice Smith" in normalized
    assert all("?" not in line for line in normalized)


def test_reconfigure_raises_for_unknown_azure_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = ChatAgent.__new__(ChatAgent)
    agent.provider = Provider.OPENAI
    agent.model_name = "gpt-4o"
    agent.instructions = "base"
    agent._api_key = None
    agent._endpoint = None
    agent._messages = ["m"]

    monkeypatch.setattr("agent.chat_agent.discover_models", lambda **_kwargs: ["chat-prod"])
    monkeypatch.setattr(agent, "_create_agent", lambda **_kwargs: object())

    with pytest.raises(ProviderConfigurationError):
        ChatAgent.reconfigure(
            agent,
            provider=Provider.AZURE_FOUNDRY,
            model_name="chat-dev",
            endpoint="https://example.azure.com",
        )
