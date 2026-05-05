from __future__ import annotations

import asyncio
from threading import RLock

import pytest

from agent.chat_agent import ChatAgent, Provider, ProviderConfigurationError


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
        async def run(self, _message, message_history):
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
        def run_stream(self, _message, message_history):
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
        async def run(self, _message, message_history):
            assert message_history == []
            return _FakeResult("ignored")

        def run_stream(self, _message, message_history):
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


def test_generate_memory_update_falls_back_to_existing_memory() -> None:
    class FailingMemoryAgent:
        async def run(self, _prompt):
            raise RuntimeError("content_filter")

    agent = ChatAgent.__new__(ChatAgent)
    agent.provider = Provider.OPENAI
    agent.model_name = "gpt-4o"
    agent._api_key = None
    agent._endpoint = None
    agent._create_agent = lambda **_kwargs: FailingMemoryAgent()

    current = "- Preference: prefers concise answers\n"
    updated = asyncio.run(
        ChatAgent.generate_memory_update(
            agent,
            current,
            "Please call me Om and use bullet points",
            "hello",
        )
    )
    assert "Preference: prefers concise answers" in updated
    assert "Name: Om" in updated


def test_generate_memory_update_merges_model_output_and_filters_noise() -> None:
    class SuccessfulMemoryAgent:
        async def run(self, _prompt):
            return _FakeResult(
                """
- Preference: prefers concise answers
- Location: Bangalore
- what is my name?
"""
            )

    agent = ChatAgent.__new__(ChatAgent)
    agent.provider = Provider.OPENAI
    agent.model_name = "gpt-4o"
    agent._api_key = None
    agent._endpoint = None
    agent._create_agent = lambda **_kwargs: SuccessfulMemoryAgent()

    updated = asyncio.run(
        ChatAgent.generate_memory_update(
            agent,
            "- Preference: prefers detailed answers\n",
            "Please call me Om",
            "Sure",
        )
    )

    assert "Name: Om" in updated
    assert "Preference: prefers concise answers" in updated
    assert "Preference: prefers detailed answers" not in updated
    assert "what is my name" not in updated.lower()


def test_fallback_memory_update_dedupes_lines() -> None:
    current = "- Preference: Keep responses concise\n"
    updated = ChatAgent._fallback_memory_update(current, "Please use concise responses")
    assert updated.count("Preference:") == 1


def test_normalize_memory_lines_filters_questions_and_keeps_name() -> None:
    memory = """
- what is my name?
- Jairaj Sahgal
- Please remember my name
"""
    normalized = ChatAgent._normalize_memory_lines(memory)
    assert "Name: Jairaj Sahgal" in normalized
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
