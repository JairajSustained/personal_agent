from __future__ import annotations

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

    def setObjectName(self, name: str) -> None:  # noqa: N802
        self.object_name = name


class _StubTranscript:
    def __init__(self) -> None:
        self.blocks: list[str] = []

    def append(self, block: str) -> None:
        self.blocks.append(block)


def test_build_agent_instructions_includes_persistent_memory_block() -> None:
    window = chat_window.ChatWindow.__new__(chat_window.ChatWindow)
    window._memory_store = _StubMemoryStore("- Name: Om\n")

    instructions = chat_window.ChatWindow._build_agent_instructions(window)

    assert "Persistent Memory:" in instructions
    assert "- Name: Om" in instructions


def test_build_agent_instructions_uses_base_when_memory_empty() -> None:
    window = chat_window.ChatWindow.__new__(chat_window.ChatWindow)
    window._memory_store = _StubMemoryStore("  \n")

    instructions = chat_window.ChatWindow._build_agent_instructions(window)

    assert instructions == chat_window.ChatWindow.BASE_INSTRUCTIONS


def test_normalize_memory_file_rewrites_noisy_memory_lines() -> None:
    window = chat_window.ChatWindow.__new__(chat_window.ChatWindow)
    store = _StubMemoryStore("- what is my name?\n- Jairaj Sahgal\n")
    window._memory_store = store

    chat_window.ChatWindow._normalize_memory_file(window)

    assert store.saved == ["- Name: Jairaj Sahgal\n"]


def test_set_view_mode_toggles_between_chat_and_graph() -> None:
    window = chat_window.ChatWindow.__new__(chat_window.ChatWindow)
    window.view_stack = _StubStack()
    window.chat_view_button = _StubButton()
    window.graph_view_button = _StubButton()
    window._apply_theme = lambda: None

    chat_window.ChatWindow._set_view_mode(window, "graph")
    assert window.view_stack.current_index == 1
    assert window.chat_view_button.object_name == "secondaryButton"
    assert window.graph_view_button.object_name == "primaryButton"

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


def test_append_message_uses_markdown_rendering_for_assistant() -> None:
    window = chat_window.ChatWindow.__new__(chat_window.ChatWindow)
    window.transcript = _StubTranscript()

    chat_window.ChatWindow._append_message(window, "Assistant", "**Hello**")

    assert len(window.transcript.blocks) == 1
    block = window.transcript.blocks[0]
    assert "Assistant" in block
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
