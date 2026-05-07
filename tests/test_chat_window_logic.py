from __future__ import annotations

from collections.abc import Callable

import pytest

from gui import chat_window


class _StubMemoryStore:
    def __init__(self, text: str) -> None:
        self._text = text
        self.saved: list[str] = []

    def load_text(self) -> str:
        return self._text

    def save_text(self, text: str) -> None:
        self.saved.append(text)
        self._text = text


class _StubStack:
    def __init__(self) -> None:
        self.current_index: int | None = None

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802
        self.current_index = index


class _StubButton:
    def __init__(self) -> None:
        self.object_name: str | None = None
        self._style = _StubStyle()

    def setObjectName(self, name: str) -> None:  # noqa: N802
        self.object_name = name

    def style(self) -> _StubStyle:
        return self._style


class _StubStyle:
    def __init__(self) -> None:
        self.polish_calls = 0
        self.unpolish_calls = 0

    def polish(self, _widget) -> None:
        self.polish_calls += 1

    def unpolish(self, _widget) -> None:
        self.unpolish_calls += 1


@pytest.fixture
def render_assistant_block() -> Callable[[str], str]:
    window = chat_window.ChatWindow.__new__(chat_window.ChatWindow)

    def _render(text: str) -> str:
        block = chat_window.ChatWindow._message_html(window, "Assistant", text)
        return " ".join(block.split())

    return _render


def test_build_agent_instructions_includes_persistent_memory_block() -> None:
    window = chat_window.ChatWindow.__new__(chat_window.ChatWindow)
    window._memory_store = _StubMemoryStore("- Name: John Doe\n")

    instructions = chat_window.ChatWindow._build_agent_instructions(window)

    assert "Persistent Memory:" in instructions
    assert "- Name: John Doe" in instructions


def test_build_agent_instructions_uses_base_when_memory_empty() -> None:
    window = chat_window.ChatWindow.__new__(chat_window.ChatWindow)
    window._memory_store = _StubMemoryStore("  \n")

    instructions = chat_window.ChatWindow._build_agent_instructions(window)

    assert instructions == chat_window.ChatWindow.BASE_INSTRUCTIONS


def test_normalize_memory_file_rewrites_noisy_memory_lines() -> None:
    window = chat_window.ChatWindow.__new__(chat_window.ChatWindow)
    store = _StubMemoryStore("- what is my name?\n- Alice Smith\n")
    window._memory_store = store

    chat_window.ChatWindow._normalize_memory_file(window)

    assert store.saved == ["- Name: Alice Smith\n"]


def test_set_view_mode_toggles_between_chat_and_graph() -> None:
    window = chat_window.ChatWindow.__new__(chat_window.ChatWindow)
    window.view_stack = _StubStack()
    window.chat_view_button = _StubButton()
    window.graph_view_button = _StubButton()

    chat_window.ChatWindow._set_view_mode(window, "graph")
    assert window.view_stack.current_index == 1
    assert window.chat_view_button.object_name == "secondaryButton"
    assert window.graph_view_button.object_name == "primaryButton"
    assert window.chat_view_button.style().polish_calls == 1
    assert window.graph_view_button.style().polish_calls == 1

    chat_window.ChatWindow._set_view_mode(window, "chat")
    assert window.view_stack.current_index == 0
    assert window.chat_view_button.object_name == "primaryButton"
    assert window.graph_view_button.object_name == "secondaryButton"


def test_assistant_markdown_to_html_renders_basic_markdown() -> None:
    rendered = chat_window.ChatWindow._assistant_markdown_to_html("**Bold**\n\n- one\n- two")

    assert "Bold" in rendered
    assert "font-weight" in rendered
    assert "one" in rendered
    assert "two" in rendered


def test_message_html_uses_markdown_rendering_for_assistant() -> None:
    window = chat_window.ChatWindow.__new__(chat_window.ChatWindow)
    block = chat_window.ChatWindow._message_html(window, "Assistant", "**Hello**")

    assert "background:#171c26" in block
    assert "font-weight" in block
    assert "Hello" in block


def test_latex_math_to_plain_converts_fraction_symbols_and_power() -> None:
    converted = chat_window.ChatWindow._latex_math_to_plain(r"\pi = \frac{C}{d} + r^{2}")

    assert converted == "pi = (C)/(d) + r^2"


def test_normalize_math_markup_handles_inline_and_display_latex() -> None:
    source = r"Pi (\pi) and equation \[ C = 2\pi r \]"

    normalized = chat_window.ChatWindow._normalize_math_markup(source)

    assert "\\pi" not in normalized
    assert "Pi (pi)" in normalized
    assert "C = 2pi r" in normalized


def test_normalize_math_markup_converts_bracket_equation_line() -> None:
    source = """
where:
[ A = \\pi r^2 ]
and this should stay:
[not latex]
"""

    normalized = chat_window.ChatWindow._normalize_math_markup(source)

    assert "A = pi r^2" in normalized
    assert "[not latex]" in normalized


def test_assistant_markdown_to_html_normalizes_user_reported_math_snippet() -> None:
    snippet = r"""Pi ((\pi)) is most basically defined as:
[ \pi = \frac{C}{d} ]
where:
(C) = circumference of a circle
(d) = diameter of the circle
[ C = \pi d ]
[ A = \pi r^2 ]
[ \pi \approx 3.14159 ]
"""

    rendered = chat_window.ChatWindow._assistant_markdown_to_html(snippet)

    assert "\\pi" not in rendered
    assert "\\frac" not in rendered
    assert "(C)/(d)" in rendered
    assert "3.14159" in rendered


@pytest.mark.parametrize(
    ("text", "expected", "unexpected"),
    [
        (
            "**Bold**\n\n- one\n- two",
            ["Bold", "one", "two", "display:inline-block;background:#171c26"],
            ["\\pi", "\\frac"],
        ),
        (
            "Pi ((\\pi)) is most basically defined as:\n[ \\pi = \\frac{C}{d} ]\n[ \\pi \\approx 3.14159 ]",
            ["(C)/(d)", "3.14159"],
            [r"\pi", r"\frac", r"\approx"],
        ),
        (
            r"Area is \(\pi r^2\) and circumference is [ C = 2\pi r ]",
            ["pi r^2", "C = 2pi r"],
            [r"\pi", r"\("],
        ),
    ],
    ids=["markdown-list", "math-snippet", "inline-and-bracket-math"],
)
def test_assistant_bubble_html_snapshot_cases(
    render_assistant_block: Callable[[str], str],
    text: str,
    expected: list[str],
    unexpected: list[str],
) -> None:
    rendered = render_assistant_block(text)

    for token in expected:
        assert token in rendered
    for token in unexpected:
        assert token not in rendered


def test_apply_template_text_prefixes_only_once() -> None:
    template = "Create a practical step-by-step plan for:\n"

    first = chat_window.ChatWindow._apply_template_text(template, "Ship v1")
    second = chat_window.ChatWindow._apply_template_text(template, first)

    assert first == "Create a practical step-by-step plan for:\nShip v1"
    assert second == first


def test_extract_last_message_by_role() -> None:
    transcript = [
        {"role": "You", "content": "hello"},
        {"role": "Assistant", "content": "hi"},
        {"role": "You", "content": "plan this"},
    ]

    assert chat_window.ChatWindow._extract_last_message(transcript, "You") == "plan this"
    assert chat_window.ChatWindow._extract_last_message(transcript, "Assistant") == "hi"
    assert chat_window.ChatWindow._extract_last_message(transcript, "System") == ""


def test_normalize_memory_editor_text_dedupes_and_formats() -> None:
    raw = "Name: John Doe\n- Name: John Doe\nwhat is my name?\nPreference: concise answers\n"

    normalized = chat_window.ChatWindow._normalize_memory_editor_text(raw)

    assert normalized == "- Name: John Doe\n- Preference: concise answers\n"
