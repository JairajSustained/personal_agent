"""Desktop GUI for the personal chat agent."""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import textwrap
from datetime import UTC, datetime, timedelta

from PySide6.QtCore import QObject, QPointF, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPalette,
    QPen,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from agent import (
    PROVIDER_MODEL_CATALOG,
    ChatAgent,
    ConversationStore,
    MemoryStore,
    Provider,
    ProviderConfigurationError,
    configured_providers,
    discover_models,
    load_librechat_models,
    providers_for_ui,
)

LOGGER = logging.getLogger(__name__)


class ChatWorkerSignals(QObject):
    """Signals emitted by background chat streaming worker."""

    chunk = Signal(str)
    done = Signal()
    failed = Signal(str)


class ChatWorker(QRunnable):
    """Run a streaming chat request in a dedicated event loop thread."""

    def __init__(self, agent: ChatAgent, message: str) -> None:
        super().__init__()
        self._agent = agent
        self._message = message
        self.signals = ChatWorkerSignals()

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            task = loop.create_task(self._run_stream())
            loop.run_until_complete(task)
            self._safe_emit(self.signals.done)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Chat worker failed")
            self._safe_emit(self.signals.failed, str(exc))
        finally:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            asyncio.set_event_loop(None)
            loop.close()

    async def _run_stream(self) -> None:
        async for chunk in self._agent.chat_stream(self._message):
            self._safe_emit(self.signals.chunk, chunk)

    @staticmethod
    def _safe_emit(signal, *args) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:
            pass


class TitleWorkerSignals(QObject):
    """Signals emitted by async title generation worker."""

    done = Signal(str)
    failed = Signal(str)


class TitleWorker(QRunnable):
    """Generate conversation titles without blocking the UI."""

    def __init__(self, agent: ChatAgent, first_user_message: str) -> None:
        super().__init__()
        self._agent = agent
        self._first_user_message = first_user_message
        self.signals = TitleWorkerSignals()

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            task = loop.create_task(self._generate())
            title = loop.run_until_complete(task)
            self._safe_emit(self.signals.done, title)
        except Exception as exc:  # noqa: BLE001
            self._safe_emit(self.signals.failed, str(exc))
        finally:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            asyncio.set_event_loop(None)
            loop.close()

    async def _generate(self) -> str:
        return await self._agent.generate_title(self._first_user_message)

    @staticmethod
    def _safe_emit(signal, *args) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:
            pass


class MemoryGraphWidget(QWidget):
    """Simple memory graph view inspired by Dost-style relationship map."""

    _NODE_COLORS = {
        "person": "#c4a882",
        "topic": "#8aab9b",
        "preference": "#a889c4",
        "event": "#c48a8a",
        "concept": "#89a8c4",
    }
    _PREFERENCE_KW = {"like", "prefer", "love", "enjoy", "want", "dislike", "hate", "favorite", "favourite"}
    _EVENT_KW = {"meeting", "event", "appointment", "yesterday", "tomorrow", "last week", "next week",
                 "january", "february", "march", "april", "may", "june", "july", "august",
                 "september", "october", "november", "december"}
    _PERSON_KW = {"name is", "named", "friend", "colleague", "family", "mother", "father",
                  "brother", "sister", "manager", "boss"}
    _CONCEPT_KW = {"because", "therefore", "which", "that", "based on", "generally", "typically", "usually"}

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._nodes: list[dict] = []
        self._edges: list[tuple[int, int, str]] = []
        self._selected_idx: int | None = None
        self.setMinimumHeight(320)
        self.setObjectName("memoryGraph")

    @classmethod
    def _classify(cls, text: str) -> str:
        lower = text.lower()
        if any(kw in lower for kw in cls._PREFERENCE_KW):
            return "preference"
        if any(kw in lower for kw in cls._EVENT_KW):
            return "event"
        if any(kw in lower for kw in cls._PERSON_KW):
            return "person"
        if any(kw in lower for kw in cls._CONCEPT_KW):
            return "concept"
        return "topic"

    def set_graph_data(self, memory_text: str, transcript: list[dict[str, str]] = ()) -> None:  # noqa: ARG002
        facts: list[str] = []
        for line in memory_text.splitlines():
            cleaned = line.strip().lstrip("-").strip()
            if cleaned:
                facts.append(cleaned[:40])

        unique_facts: list[str] = []
        seen: set[str] = set()
        for fact in facts:
            key = fact.lower()
            if key not in seen:
                seen.add(key)
                unique_facts.append(fact)
            if len(unique_facts) >= 20:
                break

        self._nodes = [{"label": "You", "type": "person", "x": 0.5, "y": 0.5}]
        self._edges = []

        if unique_facts:
            import math

            total = len(unique_facts)
            for idx, fact in enumerate(unique_facts):
                angle = (2 * math.pi * idx) / max(1, total)
                r = 0.36 if total <= 8 else 0.40
                x = 0.5 + r * math.cos(angle)
                y = 0.5 + r * math.sin(angle)
                node_type = self._classify(fact)
                self._nodes.append({"label": fact, "type": node_type, "x": x, "y": y})
                self._edges.append((0, len(self._nodes) - 1, "remembers"))

        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        background = QLinearGradient(0, 0, rect.width(), rect.height())
        background.setColorAt(0, QColor("#10131a"))
        background.setColorAt(1, QColor("#171b24"))
        painter.fillRect(rect, background)

        painter.setPen(QPen(QColor(255, 255, 255, 10), 1))
        grid_size = 34
        for x in range(0, rect.width(), grid_size):
            painter.drawLine(x, 0, x, rect.height())
        for y in range(0, rect.height(), grid_size):
            painter.drawLine(0, y, rect.width(), y)

        if not self._nodes or len(self._nodes) == 1:
            painter.setPen(QColor("#8d96a8"))
            painter.setFont(QFont("IBM Plex Sans", 11, QFont.Medium))
            painter.drawText(rect, Qt.AlignCenter, "No memories yet — ask Dost to remember something")
            return

        points: list[QPointF] = []
        for node in self._nodes:
            points.append(QPointF(rect.width() * node["x"], rect.height() * node["y"]))

        painter.setPen(QPen(QColor("#3d4657"), 1.3))
        for a, b, label in self._edges:
            painter.drawLine(points[a], points[b])
            mid_x = (points[a].x() + points[b].x()) / 2
            mid_y = (points[a].y() + points[b].y()) / 2
            painter.setFont(QFont("IBM Plex Sans", 7, QFont.Medium))
            painter.setPen(QColor("#8d96a8"))
            painter.drawText(QPointF(mid_x + 3, mid_y - 3), label)
            painter.setPen(QPen(QColor("#3d4657"), 1.3))

        for idx, node in enumerate(self._nodes):
            is_person = node["type"] == "person"
            radius = 18 if is_person else 12
            hex_color = self._NODE_COLORS.get(node["type"], "#8aab9b")
            color = QColor(hex_color)
            if self._selected_idx == idx:
                color = color.lighter(130)

            painter.setPen(QPen(QColor("#0b0d12"), 1.8))
            painter.setBrush(QBrush(color))
            p = points[idx]
            painter.drawEllipse(p, radius, radius)

            painter.setPen(QColor("#eef2f7"))
            painter.setFont(QFont("IBM Plex Sans", 9, QFont.Medium))
            painter.drawText(p.x() + radius + 6, p.y() + 4, node["label"])

    def mousePressEvent(self, event) -> None:  # noqa: N802
        rect = self.rect()
        click = QPointF(event.position())
        self._selected_idx = None
        for idx, node in enumerate(self._nodes):
            p = QPointF(rect.width() * node["x"], rect.height() * node["y"])
            if (p - click).manhattanLength() < 22:
                self._selected_idx = idx
                break
        self.update()


class SettingsPanel(QWidget):
    """Provider/model configuration controls for the active session."""

    settings_applied = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        root_layout = QVBoxLayout(self)
        group = QGroupBox("Model Setup")
        form = QFormLayout(group)

        self.provider_combo = QComboBox(self)
        for provider in providers_for_ui():
            self.provider_combo.addItem(provider.value)

        self.model_combo = QComboBox(self)
        self.model_combo.setEditable(True)

        self.api_key_input = QLineEdit(self)
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Optional override")

        self.endpoint_input = QLineEdit(self)
        self.endpoint_input.setPlaceholderText("Optional override")

        self.refresh_button = QPushButton("Refresh Models")
        self.refresh_button.setObjectName("secondaryButton")

        self.apply_button = QPushButton("Apply")
        self.apply_button.setObjectName("primaryButton")

        self.hint_label = QLabel(self)
        self.hint_label.setWordWrap(True)

        form.addRow("Provider", self.provider_combo)
        form.addRow("Model", self.model_combo)
        form.addRow("API Key", self.api_key_input)
        form.addRow("Endpoint", self.endpoint_input)
        form.addRow(self.refresh_button)
        form.addRow(self.apply_button)
        form.addRow(self.hint_label)

        root_layout.addWidget(group)
        root_layout.addStretch()

        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        self.api_key_input.editingFinished.connect(
            lambda: self._on_provider_changed(self.provider_combo.currentText())
        )
        self.endpoint_input.editingFinished.connect(
            lambda: self._on_provider_changed(self.provider_combo.currentText())
        )
        self.refresh_button.clicked.connect(
            lambda: self._on_provider_changed(self.provider_combo.currentText())
        )
        self.apply_button.clicked.connect(self._emit_settings)

        self._apply_env_hints()
        self._on_provider_changed(self.provider_combo.currentText())

    def _apply_env_hints(self) -> None:
        """Pre-fill or hint fields when credentials are available via env vars."""
        configured = configured_providers()
        if configured:
            self.api_key_input.setPlaceholderText("Loaded from .env (optional override)")

        endpoint_value = (os.getenv("AZURE_FOUNDRY_ENDPOINT") or os.getenv("AZURE_ENDPOINT") or "").strip()
        if endpoint_value:
            self.endpoint_input.setText(endpoint_value)
            self.endpoint_input.setPlaceholderText("Loaded from .env (optional override)")

    def _yaml_models(self, provider: Provider) -> list[str]:
        """Load provider model hints from optional LibreChat-style config."""
        return load_librechat_models().get(provider, [])

    def _on_provider_changed(self, provider_text: str) -> None:
        """Refresh model options for the selected provider."""
        if not provider_text:
            return

        provider = Provider(provider_text)
        selected_model = self.model_combo.currentText().strip()

        runtime_models = discover_models(
            provider=provider,
            api_key=self.api_key_input.text().strip() or None,
            endpoint=self.endpoint_input.text().strip() or None,
        )
        yaml_models = self._yaml_models(provider)
        static_models = PROVIDER_MODEL_CATALOG.get(provider, [])

        if provider == Provider.AZURE_FOUNDRY:
            models = runtime_models or yaml_models
        else:
            models = runtime_models or yaml_models or static_models

        self.model_combo.clear()
        self.model_combo.addItems(models)

        if selected_model and selected_model in models:
            self.model_combo.setCurrentText(selected_model)

        self.endpoint_input.setEnabled(provider == Provider.AZURE_FOUNDRY)
        if provider == Provider.AZURE_FOUNDRY:
            if models:
                self.hint_label.setText("Azure deployments loaded. Select one and click Apply.")
            else:
                self.hint_label.setText("No Azure deployments found. Check .env then click Refresh Models.")
        else:
            self.hint_label.setText("Using .env keys by default. Fields are optional overrides.")

    def _emit_settings(self) -> None:
        """Emit current settings payload to parent window."""
        self.settings_applied.emit(self.get_config())

    def get_config(self) -> dict:
        """Return normalized provider configuration from UI fields."""
        return {
            "provider": Provider(self.provider_combo.currentText()),
            "model_name": self.model_combo.currentText().strip(),
            "api_key": self.api_key_input.text().strip() or None,
            "endpoint": self.endpoint_input.text().strip() or None,
        }


class ChatWindow(QMainWindow):
    """Main desktop window managing chat UI, memory, and persistence."""

    BASE_INSTRUCTIONS = "You are a helpful personal assistant."
    PROMPT_TEMPLATES = {
        "Summarize": "Summarize this clearly with key takeaways and next actions:\n",
        "Plan": "Create a practical step-by-step plan for:\n",
        "Debug": "Help me debug this. Start with likely causes, then give verification steps:\n",
        "Review": "Review this for correctness, security, performance, and maintainability:\n",
        "Remember": "Please remember this durable preference or fact:\n",
    }

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Dost — Personal Agent")
        self.resize(1220, 800)
        self.setMinimumSize(980, 680)

        self._thread_pool = QThreadPool.globalInstance()
        self._agent: ChatAgent | None = None
        self._streaming = False
        self._assistant_buffer = ""
        self._pending_user_message = ""
        self._title_generation_in_progress = False
        self._title_generated_for_conversations: set[str] = set()
        self._suppress_conversation_selection = False

        self._store = ConversationStore()
        self._memory_store = MemoryStore()
        self._current_conversation_id: str | None = None
        self._transcript_records: list[dict[str, str]] = []

        self._setup_ui()
        self._setup_shortcuts()
        self._load_initial_conversation()
        self._apply_theme()

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0f1117;
                color: #eef2f7;
                font-family: 'IBM Plex Sans', 'SF Pro Text', 'Segoe UI', sans-serif;
            }
            QWidget#sidebarPanel, QWidget#chatPanel, QWidget#composerBar {
                background: #151923;
                border: 1px solid #252b38;
                border-radius: 18px;
            }
            QWidget#topBar {
                background: #11151d;
                border: 1px solid #252b38;
                border-radius: 16px;
            }
            QLabel#brandTitle {
                color: #ffffff;
                font-size: 22px;
                font-weight: 800;
                letter-spacing: 0.2px;
            }
            QLabel#brandSubtitle, QLabel#sectionHint, QLabel#chatSubtitle {
                color: #8d96a8;
                font-size: 12px;
            }
            QLabel#chatTitle {
                color: #ffffff;
                font-size: 18px;
                font-weight: 800;
            }
            QGroupBox {
                border: 1px solid #252b38;
                border-radius: 16px;
                margin-top: 14px;
                padding: 18px 12px 12px 12px;
                background: #151923;
                font-weight: 700;
                color: #eef2f7;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                color: #c9d2e3;
            }
            QListWidget, QLineEdit, QComboBox {
                border: 1px solid #2a3140;
                border-radius: 12px;
                padding: 9px 11px;
                background: #10131a;
                color: #eef2f7;
                selection-background-color: #2f6fec;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #5b8def;
                background: #111722;
            }
            QLineEdit:disabled, QComboBox:disabled {
                color: #626b7b;
                background: #11141b;
                border-color: #202633;
            }
            QListWidget {
                padding: 6px;
                outline: 0;
            }
            QListWidget::item {
                border-radius: 10px;
                padding: 9px 10px;
                margin: 2px;
                color: #dbe3ef;
            }
            QListWidget::item:selected {
                background: #203150;
                color: #ffffff;
                border-left: 3px solid #6ea8fe;
            }
            QListWidget::item:hover { background: #1a2230; }
            QTextBrowser {
                border: 1px solid #252b38;
                border-radius: 16px;
                background: #0f1117;
                color: #eef2f7;
                font-family: 'IBM Plex Sans', 'SF Pro Text', sans-serif;
                font-size: 13px;
                padding: 6px;
            }
            QPushButton {
                min-height: 18px;
            }
            QPushButton#primaryButton {
                border: none;
                border-radius: 11px;
                padding: 9px 15px;
                background: #2f6fec;
                color: white;
                font-weight: 700;
            }
            QPushButton#primaryButton:hover { background: #3f7df1; }
            QPushButton#primaryButton:pressed { background: #265ec8; }
            QPushButton#primaryButton:disabled { background: #253454; color: #7b8798; }
            QPushButton#secondaryButton {
                border: 1px solid #2a3140;
                border-radius: 11px;
                padding: 9px 14px;
                background: #151a24;
                color: #dbe3ef;
                font-weight: 700;
            }
            QPushButton#secondaryButton:hover { background: #1b2432; border-color: #3b465a; }
            QPushButton#secondaryButton:pressed { background: #111722; }
            QPushButton#dangerButton {
                border: 1px solid #4b2a2f;
                border-radius: 11px;
                padding: 9px 14px;
                background: #21171b;
                color: #ffb4bd;
                font-weight: 700;
            }
            QPushButton#dangerButton:hover { background: #321d24; border-color: #7c3f49; }
            QStackedWidget {
                border: none;
                background: transparent;
            }
            QLabel { color: #eef2f7; }
            QSplitter::handle { background: transparent; width: 10px; }
            QStatusBar {
                background: #0f1117;
                color: #8d96a8;
                border-top: 1px solid #252b38;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                border-radius: 5px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: #303848;
                border-radius: 5px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover { background: #465165; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QComboBox::drop-down { border: none; width: 24px; }
            QComboBox QAbstractItemView {
                background: #151923;
                border: 1px solid #2a3140;
                border-radius: 10px;
                color: #eef2f7;
                selection-background-color: #203150;
                padding: 4px;
            }
            QWidget#memoryGraph {
                border: 1px solid #252b38;
                border-radius: 16px;
            }
            """
        )

    def _setup_ui(self) -> None:
        """Build the full split-pane interface and wire UI events."""
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 10)
        root_layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter)

        sidebar = QWidget(self)
        sidebar.setObjectName("sidebarPanel")
        sidebar.setMinimumWidth(280)
        sidebar.setMaximumWidth(340)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(14, 14, 14, 14)
        sidebar_layout.setSpacing(10)

        brand_label = QLabel("Dost")
        brand_label.setObjectName("brandTitle")
        brand_subtitle = QLabel("Personal memory assistant")
        brand_subtitle.setObjectName("brandSubtitle")

        self.new_chat_button = QPushButton("New chat")
        self.new_chat_button.setObjectName("primaryButton")

        self.search_input = QLineEdit(sidebar)
        self.search_input.setPlaceholderText("Search")

        self.conversation_list = QListWidget(sidebar)

        quiet_actions = QHBoxLayout()
        quiet_actions.setSpacing(8)
        self.pin_chat_button = QPushButton("Pin")
        self.pin_chat_button.setObjectName("secondaryButton")
        self.export_chat_button = QPushButton("Export")
        self.export_chat_button.setObjectName("secondaryButton")
        self.delete_chat_button = QPushButton("Delete")
        self.delete_chat_button.setObjectName("dangerButton")
        self.settings_toggle_button = QPushButton("Settings")
        self.settings_toggle_button.setObjectName("secondaryButton")
        quiet_actions.addWidget(self.pin_chat_button)
        quiet_actions.addWidget(self.export_chat_button)
        quiet_actions.addWidget(self.delete_chat_button)
        quiet_actions.addWidget(self.settings_toggle_button)

        self.settings_panel = SettingsPanel(self)
        self.settings_panel.setVisible(False)
        self.settings_panel.settings_applied.connect(self._apply_settings)

        sidebar_layout.addWidget(brand_label)
        sidebar_layout.addWidget(brand_subtitle)
        sidebar_layout.addSpacing(4)
        sidebar_layout.addWidget(self.new_chat_button)
        sidebar_layout.addWidget(self.search_input)
        sidebar_layout.addWidget(self.conversation_list, 1)
        sidebar_layout.addLayout(quiet_actions)
        sidebar_layout.addWidget(self.settings_panel)

        chat_container = QWidget(self)
        chat_container.setObjectName("chatPanel")
        chat_layout = QVBoxLayout(chat_container)
        chat_layout.setContentsMargins(14, 14, 14, 14)
        chat_layout.setSpacing(12)

        top_bar = QWidget(chat_container)
        top_bar.setObjectName("topBar")
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(14, 10, 10, 10)
        title_stack = QVBoxLayout()
        title_stack.setSpacing(1)
        self.chat_title_label = QLabel("Chat")
        self.chat_title_label.setObjectName("chatTitle")
        chat_subtitle = QLabel("Ask, plan, remember — without clutter")
        chat_subtitle.setObjectName("chatSubtitle")
        title_stack.addWidget(self.chat_title_label)
        title_stack.addWidget(chat_subtitle)
        self.chat_view_button = QPushButton("Chat")
        self.chat_view_button.setObjectName("primaryButton")
        self.graph_view_button = QPushButton("Memory")
        self.graph_view_button.setObjectName("secondaryButton")
        top_bar_layout.addLayout(title_stack)
        top_bar_layout.addStretch(1)
        top_bar_layout.addWidget(self.chat_view_button)
        top_bar_layout.addWidget(self.graph_view_button)

        self.view_stack = QStackedWidget(chat_container)

        chat_page = QWidget()
        chat_page_layout = QVBoxLayout(chat_page)

        self.transcript = QTextBrowser(chat_page)
        self.transcript.setOpenExternalLinks(True)

        composer = QWidget(chat_page)
        composer.setObjectName("composerBar")
        input_row = QHBoxLayout(composer)
        input_row.setContentsMargins(10, 10, 10, 10)
        input_row.setSpacing(8)
        self.input_field = QLineEdit(chat_page)
        self.input_field.setPlaceholderText("Message Dost...")
        self.input_field.returnPressed.connect(self._send_message)

        self.template_combo = QComboBox(chat_page)
        self.template_combo.addItems(list(self.PROMPT_TEMPLATES.keys()))
        self.template_combo.setMinimumWidth(128)

        self.template_apply_button = QPushButton("Use", chat_page)
        self.template_apply_button.setObjectName("secondaryButton")
        self.template_apply_button.clicked.connect(self._apply_selected_template)

        self.retry_button = QPushButton("Retry", chat_page)
        self.retry_button.setObjectName("secondaryButton")
        self.retry_button.clicked.connect(self._retry_last_user_message)

        self.copy_last_button = QPushButton("Copy Last", chat_page)
        self.copy_last_button.setObjectName("secondaryButton")
        self.copy_last_button.clicked.connect(self._copy_last_assistant_response)

        self.attach_button = QPushButton("Attach", chat_page)
        self.attach_button.setObjectName("secondaryButton")
        self.attach_button.clicked.connect(self._attach_file)

        self.memory_edit_button = QPushButton("Edit Memory", chat_page)
        self.memory_edit_button.setObjectName("secondaryButton")
        self.memory_edit_button.clicked.connect(self._edit_memory)

        self.send_button = QPushButton("Send", chat_page)
        self.send_button.setObjectName("primaryButton")
        self.send_button.clicked.connect(self._send_message)

        self.clear_button = QPushButton("Clear", chat_page)
        self.clear_button.setObjectName("secondaryButton")
        self.clear_button.clicked.connect(self._clear_chat)

        input_row.addWidget(self.input_field, 1)
        input_row.addWidget(self.template_combo)
        input_row.addWidget(self.template_apply_button)
        input_row.addWidget(self.retry_button)
        input_row.addWidget(self.copy_last_button)
        input_row.addWidget(self.attach_button)
        input_row.addWidget(self.memory_edit_button)
        input_row.addWidget(self.clear_button)
        input_row.addWidget(self.send_button)

        chat_page_layout.addWidget(self.transcript, 1)
        chat_page_layout.addWidget(composer)

        graph_page = QWidget()
        graph_layout = QVBoxLayout(graph_page)
        self.graph_hint = QLabel("Remembered context map")
        self.graph_hint.setObjectName("sectionHint")
        self.memory_graph = MemoryGraphWidget(graph_page)
        graph_layout.addWidget(self.graph_hint)
        graph_layout.addWidget(self.memory_graph)

        self.view_stack.addWidget(chat_page)
        self.view_stack.addWidget(graph_page)

        chat_layout.addWidget(top_bar)
        chat_layout.addWidget(self.view_stack, 1)

        splitter.addWidget(sidebar)
        splitter.addWidget(chat_container)
        splitter.setSizes([360, 840])

        self.new_chat_button.clicked.connect(self._new_chat)
        self.pin_chat_button.clicked.connect(self._toggle_pin_current_chat)
        self.export_chat_button.clicked.connect(self._export_current_chat_markdown)
        self.delete_chat_button.clicked.connect(self._delete_current_chat)
        self.settings_toggle_button.clicked.connect(self._toggle_settings_panel)
        self.conversation_list.itemSelectionChanged.connect(self._on_conversation_selected)
        self.search_input.textChanged.connect(lambda _text: self._refresh_conversation_list())
        self.chat_view_button.clicked.connect(lambda: self._set_view_mode("chat"))
        self.graph_view_button.clicked.connect(lambda: self._set_view_mode("graph"))
        self._set_view_mode("chat")

    def _setup_shortcuts(self) -> None:
        """Register keyboard shortcuts for common desktop actions."""
        self._shortcuts = [
            QShortcut(QKeySequence("Ctrl+N"), self),
            QShortcut(QKeySequence("Ctrl+F"), self),
            QShortcut(QKeySequence("Ctrl+E"), self),
            QShortcut(QKeySequence("Ctrl+R"), self),
            QShortcut(QKeySequence("Ctrl+Shift+C"), self),
            QShortcut(QKeySequence("Ctrl+M"), self),
            QShortcut(QKeySequence("Ctrl+Shift+A"), self),
        ]
        self._shortcuts[0].activated.connect(self._new_chat)
        self._shortcuts[1].activated.connect(self.search_input.setFocus)
        self._shortcuts[2].activated.connect(self._export_current_chat_markdown)
        self._shortcuts[3].activated.connect(self._retry_last_user_message)
        self._shortcuts[4].activated.connect(self._copy_last_assistant_response)
        self._shortcuts[5].activated.connect(self._edit_memory)
        self._shortcuts[6].activated.connect(self._attach_file)

    @staticmethod
    def _apply_template_text(template: str, current: str) -> str:
        body = current.lstrip()
        if body.startswith(template):
            return current
        return f"{template}{current}" if current else template

    @staticmethod
    def _extract_last_message(transcript: list[dict[str, str]], role: str) -> str:
        for message in reversed(transcript):
            if message.get("role") == role:
                return str(message.get("content", "")).strip()
        return ""

    @staticmethod
    def _normalize_memory_editor_text(text: str) -> str:
        normalized_lines = ChatAgent._normalize_memory_lines(text)
        normalized = "\n".join(f"- {line}" for line in normalized_lines).strip()
        return f"{normalized}\n" if normalized else ""

    def _toggle_settings_panel(self) -> None:
        """Keep advanced model settings out of the main view until needed."""
        visible = not self.settings_panel.isVisible()
        self.settings_panel.setVisible(visible)
        self.settings_toggle_button.setText("Hide" if visible else "Settings")

    def _refresh_conversation_list(self) -> None:
        """Render conversations in sidebar grouped by date, honoring search and pin state."""
        current_id = self._current_conversation_id
        self._suppress_conversation_selection = True
        self.conversation_list.blockSignals(True)
        try:
            self.conversation_list.clear()
            query = self.search_input.text().strip()
            records = self._store.search_conversations(query)

            today = datetime.now(UTC).date()
            yesterday = today - timedelta(days=1)

            def _date_group(record) -> str:
                if record.pinned:
                    return "Pinned"
                try:
                    dt = datetime.fromisoformat(record.updated_at)
                    d = dt.date()
                    if d == today:
                        return "Today"
                    if d == yesterday:
                        return "Yesterday"
                except Exception:
                    pass
                return "Earlier"

            current_group: str | None = None
            for record in records:
                group = _date_group(record)
                if group != current_group:
                    current_group = group
                    header = QListWidgetItem(group)
                    header.setFlags(Qt.NoItemFlags)
                    header.setForeground(QColor("#9196a1"))
                    header.setFont(QFont("IBM Plex Sans", 8, QFont.Bold))
                    self.conversation_list.addItem(header)

                item = QListWidgetItem(record.title)
                item.setData(Qt.UserRole, record.id)
                item.setToolTip(f"{record.provider} • {record.model_name}")
                self.conversation_list.addItem(item)
                if record.id == current_id:
                    self.conversation_list.setCurrentItem(item)
        finally:
            self.conversation_list.blockSignals(False)
            self._suppress_conversation_selection = False

        active = self._store.get_conversation(current_id) if current_id else None
        self.pin_chat_button.setText("Unpin" if active and active.pinned else "Pin")

    def _default_provider_and_model(self) -> tuple[Provider, str]:
        config = self.settings_panel.get_config()
        provider = config["provider"]
        model = config["model_name"]
        if not model:
            models = discover_models(provider, config["api_key"], config["endpoint"]) or PROVIDER_MODEL_CATALOG.get(
                provider, []
            )
            model = models[0] if models else ""
        return provider, model

    def _load_initial_conversation(self) -> None:
        """Load memory, restore active conversation, or bootstrap a new one."""
        self._normalize_memory_file()

        records = self._store.list_conversations()
        active_id = self._store.get_active_id()

        if not records:
            provider, model = self._default_provider_and_model()
            if not model:
                self.statusBar().showMessage("Select a provider and refresh models.")
                return
            record = self._store.create_conversation(provider, model)
            records = [record]
            active_id = record.id

        self._refresh_conversation_list()
        target = next((record for record in records if record.id == active_id), records[0])
        self._activate_conversation(target.id)

    def _activate_conversation(self, conversation_id: str) -> None:
        """Switch UI/agent context to a selected persisted conversation."""
        record = self._store.get_conversation(conversation_id)
        if record is None:
            return

        self._current_conversation_id = record.id
        self._transcript_records = list(record.transcript)

        self._render_transcript()
        self._set_provider_and_model(record.provider, record.model_name)
        self._apply_settings(self.settings_panel.get_config(), show_dialogs=False)

        if self._agent is not None and record.history_json:
            try:
                self._agent.import_history_json(record.history_json)
            except Exception:
                LOGGER.warning("Could not restore chat history for conversation %s", record.id)

        self._title_generation_in_progress = False
        self._memory_update_in_progress = False

    def _set_provider_and_model(self, provider: str, model_name: str) -> None:
        idx = self.settings_panel.provider_combo.findText(provider)
        if idx < 0:
            self.settings_panel.provider_combo.addItem(provider)
            idx = self.settings_panel.provider_combo.findText(provider)
        self.settings_panel.provider_combo.setCurrentIndex(idx)

        self.settings_panel._on_provider_changed(provider)
        if model_name:
            if self.settings_panel.model_combo.findText(model_name) < 0:
                self.settings_panel.model_combo.addItem(model_name)
            self.settings_panel.model_combo.setCurrentText(model_name)

    def _render_transcript(self) -> None:
        self.transcript.setHtml(self._build_full_html())
        self._refresh_graph_view()

    def _set_view_mode(self, mode: str) -> None:
        """Switch between chat and memory-graph views."""
        if mode == "graph":
            self.view_stack.setCurrentIndex(1)
            self.chat_view_button.setObjectName("secondaryButton")
            self.graph_view_button.setObjectName("primaryButton")
            if hasattr(self, "chat_title_label"):
                self.chat_title_label.setText("Memory")
        else:
            self.view_stack.setCurrentIndex(0)
            self.chat_view_button.setObjectName("primaryButton")
            self.graph_view_button.setObjectName("secondaryButton")
            if hasattr(self, "chat_title_label"):
                self.chat_title_label.setText("Chat")
        self.chat_view_button.style().unpolish(self.chat_view_button)
        self.chat_view_button.style().polish(self.chat_view_button)
        self.graph_view_button.style().unpolish(self.graph_view_button)
        self.graph_view_button.style().polish(self.graph_view_button)

    def _refresh_graph_view(self) -> None:
        """Refresh memory graph using stored memory and current transcript."""
        memory_text = self._memory_store.load_text()
        self.memory_graph.set_graph_data(memory_text=memory_text, transcript=self._transcript_records)

    def _message_html(self, role: str, text: str, timestamp: str | None = None) -> str:
        """Return HTML for a single chat message."""
        safe_text = html.escape(text).replace("\n", "<br>")
        ts_bit = f"<br><font size='1' color='#7f8998'>{html.escape(timestamp)}</font>" if timestamp else ""
        role_lower = role.lower()
        if role_lower == "you":
            return (
                f"<p align='right' style='margin:10px 0;'>"
                f"<span style='display:inline-block;background:#2f6fec;color:#ffffff;"
                f"padding:10px 14px;border-radius:16px;'>{safe_text}</span>"
                f"{ts_bit}</p>"
            )
        if role_lower == "assistant":
            rendered = self._assistant_markdown_to_html(text)
            return (
                f"<p align='left' style='margin:10px 0;'>"
                f"<span style='display:inline-block;background:#171c26;color:#eef2f7;"
                f"border:1px solid #252b38;padding:10px 14px;border-radius:16px;'>{rendered}</span>"
                f"{ts_bit}</p>"
            )
        return f"<p><font color='#ffb4bd'><b>{html.escape(role)}:</b> {safe_text}</font></p>"

    def _build_full_html(self, extra: str = "") -> str:
        """Build a complete HTML document for setHtml(), avoiding Qt append() state leaks."""
        parts = [
            self._message_html(m.get("role", "System"), m.get("content", ""), m.get("ts"))
            for m in self._transcript_records
        ]
        if extra:
            parts.append(extra)
        body = "".join(parts)
        if not body:
            body = (
                "<div style='height:520px;display:flex;align-items:center;justify-content:center;'>"
                "<div style='text-align:center;color:#8d96a8;'>"
                "<div style='font-size:28px;color:#eef2f7;font-weight:700;margin-bottom:8px;'>What do you want to do?</div>"
                "<div>Start with a question, a plan, or something you want Dost to remember.</div>"
                "</div></div>"
            )
        return (
            "<html><head></head>"
            "<body style='background-color:#0f1117;color:#eef2f7;"
            "font-family:\"IBM Plex Sans\",\"SF Pro Text\",sans-serif;font-size:13px;margin:10px;'>"
            f"{body}</body></html>"
        )

    @staticmethod
    def _assistant_markdown_to_html(text: str) -> str:
        from PySide6.QtGui import QTextDocument

        normalized_text = ChatWindow._normalize_math_markup(text)
        document = QTextDocument()
        document.setMarkdown(normalized_text)
        rendered = document.toHtml()

        body_match = re.search(r"<body[^>]*>(?P<body>.*)</body>", rendered, flags=re.IGNORECASE | re.DOTALL)
        if body_match:
            return body_match.group("body")
        return html.escape(normalized_text).replace("\n", "<br>")

    @staticmethod
    def _normalize_math_markup(text: str) -> str:
        normalized = text

        normalized = re.sub(
            r"\\\[(.*?)\\\]",
            lambda match: ChatWindow._latex_math_to_plain(match.group(1)),
            normalized,
            flags=re.DOTALL,
        )
        normalized = re.sub(
            r"\\\((.*?)\\\)",
            lambda match: ChatWindow._latex_math_to_plain(match.group(1)),
            normalized,
            flags=re.DOTALL,
        )

        normalized = re.sub(
            r"\(([^()\n]*\\[A-Za-z][^()\n]*)\)",
            lambda match: f"({ChatWindow._latex_math_to_plain(match.group(1))})",
            normalized,
        )

        def _bracket_math(match: re.Match[str]) -> str:
            inner = match.group(1)
            if "\\" in inner or "^" in inner or "_" in inner:
                return ChatWindow._latex_math_to_plain(inner)
            return match.group(0)

        normalized = re.sub(r"(?m)^\[\s*(.+?)\s*\]$", _bracket_math, normalized)
        normalized = re.sub(
            r"\[\s*([^\]\n]*\\[A-Za-z][^\]\n]*)\s*\]",
            lambda match: ChatWindow._latex_math_to_plain(match.group(1)),
            normalized,
        )
        return normalized

    @staticmethod
    def _latex_math_to_plain(expression: str) -> str:
        plain = expression.strip()

        frac_pattern = r"\\frac\{([^{}]+)\}\{([^{}]+)\}"
        while re.search(frac_pattern, plain):
            plain = re.sub(frac_pattern, r"(\1)/(\2)", plain)

        plain = re.sub(r"\\text\{([^{}]+)\}", r"\1", plain)

        symbol_map = {
            r"\pi": "pi",
            r"\theta": "theta",
            r"\alpha": "alpha",
            r"\beta": "beta",
            r"\gamma": "gamma",
            r"\delta": "delta",
            r"\approx": "~=",
            r"\times": "*",
            r"\cdot": "*",
            r"\leq": "<=",
            r"\geq": ">=",
            r"\neq": "!=",
        }
        for latex_symbol, plain_symbol in symbol_map.items():
            plain = plain.replace(latex_symbol, plain_symbol)

        plain = re.sub(
            r"\^\{([^{}]+)\}",
            lambda match: f"^{match.group(1)}",
            plain,
        )
        plain = re.sub(r"_\{([^{}]+)\}", r"_\1", plain)

        plain = plain.replace("{", "").replace("}", "")
        plain = re.sub(r"\s+", " ", plain)
        return plain.strip()

    def _append_assistant_chunk(self, chunk: str) -> None:
        self._assistant_buffer += chunk
        self._render_transcript_preview()

    def _render_transcript_preview(self) -> None:
        if self._assistant_buffer:
            extra = self._message_html("Assistant", self._assistant_buffer)
        else:
            extra = (
                "<p align='left' style='margin:10px 0;'>"
                "<span style='display:inline-block;background:#171c26;color:#8d96a8;"
                "border:1px solid #252b38;padding:10px 14px;border-radius:16px;'>&#9679; &#9679; &#9679;</span></p>"
            )
        self.transcript.setHtml(self._build_full_html(extra))
        self.transcript.verticalScrollBar().setValue(self.transcript.verticalScrollBar().maximum())

    def _set_chat_enabled(self, enabled: bool) -> None:
        self.input_field.setEnabled(enabled)
        self.send_button.setEnabled(enabled)
        self.settings_panel.apply_button.setEnabled(enabled)
        self.settings_panel.refresh_button.setEnabled(enabled)

    def _build_agent_instructions(self, query: str | None = None) -> str:
        """Compose runtime agent instructions including persisted/retrieved memory."""
        if query and hasattr(self._memory_store, "load_relevant_text"):
            memory_text = self._memory_store.load_relevant_text(query).strip()
            if not memory_text:
                memory_text = self._memory_store.load_text().strip()
        else:
            memory_text = self._memory_store.load_text().strip()

        if not memory_text:
            return self.BASE_INSTRUCTIONS

        heading = "Relevant Persistent Memory" if query else "Persistent Memory"
        return (
            f"{self.BASE_INSTRUCTIONS}\n\n"
            f"{heading}:\n"
            "Use this memory as long-term context for this response.\n"
            "If the user asks to override memory in this chat, follow the user.\n"
            f"---\n{memory_text}\n---"
        )

    def _normalize_memory_file(self) -> None:
        """Normalize existing memory file to reduce noisy/duplicate lines."""
        current = self._memory_store.load_text()
        normalized_lines = ChatAgent._normalize_memory_lines(current)
        normalized = "\n".join(f"- {line}" for line in normalized_lines).strip()
        normalized = f"{normalized}\n" if normalized else ""
        if normalized != current:
            self._memory_store.save_text(normalized)

    def _apply_settings(self, config: dict, show_dialogs: bool = True) -> None:
        """Apply provider/model settings and reconfigure the active agent."""
        try:
            if self._agent is None:
                self._agent = ChatAgent(**config, instructions=self._build_agent_instructions(), memory_store=self._memory_store)
            else:
                self._agent.reconfigure(
                    **config,
                    instructions=self._build_agent_instructions(),
                    validate_model=show_dialogs,
                )

            provider = config["provider"].value
            model_name = config["model_name"]
            self.statusBar().showMessage(f"Connected: {provider} / {model_name}")

            if self._current_conversation_id:
                self._store.upsert_conversation(
                    conversation_id=self._current_conversation_id,
                    provider=config["provider"],
                    model_name=model_name,
                    transcript=self._transcript_records,
                    history_json=self._agent.export_history_json(),
                )
                self._refresh_conversation_list()
        except ProviderConfigurationError as exc:
            self.statusBar().showMessage("Check provider credentials/model and click Apply again")
            if show_dialogs:
                QMessageBox.warning(self, "Configuration Error", str(exc))
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to apply provider settings")
            self.statusBar().showMessage("Initialization failed")
            if show_dialogs:
                QMessageBox.critical(self, "Initialization Error", str(exc))

    def _conversation_title(self) -> str:
        for message in self._transcript_records:
            if message.get("role") == "You":
                title = message.get("content", "").strip()
                if title:
                    return textwrap.shorten(title, width=48, placeholder="...")
        return "New Chat"

    def _apply_selected_template(self) -> None:
        template_name = self.template_combo.currentText().strip()
        template = self.PROMPT_TEMPLATES.get(template_name)
        if not template:
            return
        self.input_field.setText(self._apply_template_text(template, self.input_field.text()))
        self.input_field.setFocus()

    def _copy_last_assistant_response(self) -> None:
        content = self._extract_last_message(self._transcript_records, "Assistant")
        if not content:
            self.statusBar().showMessage("No assistant response to copy")
            return
        QApplication.clipboard().setText(content)
        self.statusBar().showMessage("Copied last assistant response")

    def _retry_last_user_message(self) -> None:
        if self._streaming or self._agent is None:
            return
        message = self._extract_last_message(self._transcript_records, "You")
        if not message:
            self.statusBar().showMessage("No previous user message to retry")
            return
        self.input_field.setText(message)
        self._send_message()

    def _edit_memory(self) -> None:
        current = self._memory_store.load_text()
        updated, accepted = QInputDialog.getMultiLineText(self, "Edit Memory", "Persistent memory", current)
        if not accepted:
            return

        normalized = self._normalize_memory_editor_text(updated)
        self._memory_store.save_text(normalized)
        if self._agent is not None:
            self._agent.update_instructions(self._build_agent_instructions())
        self._refresh_graph_view()
        self.statusBar().showMessage("Memory updated")

    def _attach_file(self) -> None:
        """Open a file and prepend its content as context to the input field."""
        file_filter = "Text files (*.txt *.md *.py *.js *.ts *.json *.yaml *.yml *.csv *.html *.css *.sh *.toml *.ini *.log)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Attach File", "", file_filter)
        if not file_path:
            return

        try:
            with open(file_path, encoding="utf-8", errors="replace") as fh:
                content = fh.read()

            if len(content) > 12_000:
                content = content[:12_000] + "\n...(truncated)"

            filename = os.path.basename(file_path)
            ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
            prefix = f"[File: {filename}]\n```{ext}\n{content}\n```\n\n"
            self.input_field.setText(prefix + self.input_field.text())
            self.input_field.setFocus()
            self.statusBar().showMessage(f"Attached: {filename} ({len(content):,} chars)")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Attach File", f"Could not read file:\n{exc}")

    def _send_message(self) -> None:
        """Queue a user message for streaming response generation."""
        if self._streaming or self._agent is None:
            return

        message = self.input_field.text().strip()
        if not message:
            return

        if self._agent is not None:
            self._agent.update_instructions(self._build_agent_instructions(query=message))

        self._streaming = True
        self._assistant_buffer = ""
        self._pending_user_message = message
        self._set_chat_enabled(False)
        self.input_field.clear()

        self._transcript_records.append({"role": "You", "content": message, "ts": datetime.now().strftime("%H:%M")})
        self._render_transcript_preview()

        worker = ChatWorker(self._agent, message)
        worker.signals.chunk.connect(self._append_assistant_chunk)
        worker.signals.done.connect(self._on_response_done)
        worker.signals.failed.connect(self._on_response_failed)
        self._thread_pool.start(worker)

    def _on_response_done(self) -> None:
        self._streaming = False
        self._set_chat_enabled(True)
        self.input_field.setFocus()

        self._transcript_records.append({"role": "Assistant", "content": self._assistant_buffer, "ts": datetime.now().strftime("%H:%M")})
        self._assistant_buffer = ""
        self._render_transcript()
        self._persist_active_conversation()
        self._maybe_generate_model_title()
        self._refresh_graph_view()

    def _on_response_failed(self, error_message: str) -> None:
        self._streaming = False
        self._set_chat_enabled(True)
        self._assistant_buffer = ""
        self._transcript_records.append({"role": "System", "content": f"Error: {error_message}"})
        self._render_transcript()
        self.statusBar().showMessage("Request failed")
        self.input_field.setFocus()
        self._persist_active_conversation()

    def _persist_active_conversation(self) -> None:
        """Persist transcript, metadata, and model history for active chat."""
        if self._current_conversation_id is None or self._agent is None:
            return

        config = self.settings_panel.get_config()
        self._store.upsert_conversation(
            conversation_id=self._current_conversation_id,
            provider=config["provider"],
            model_name=config["model_name"],
            transcript=self._transcript_records,
            history_json=self._agent.export_history_json(),
            title=self._conversation_title(),
        )
        self._refresh_conversation_list()

    def _first_user_message(self) -> str:
        for message in self._transcript_records:
            if message.get("role") == "You":
                return str(message.get("content", "")).strip()
        return ""

    def _maybe_generate_model_title(self) -> None:
        """Generate a refined title once a conversation has first exchange."""
        if self._agent is None or self._current_conversation_id is None:
            return
        if self._title_generation_in_progress:
            return
        if self._current_conversation_id in self._title_generated_for_conversations:
            return

        first_user_message = self._first_user_message()
        has_assistant = any(message.get("role") == "Assistant" for message in self._transcript_records)
        if not first_user_message or not has_assistant:
            return

        self._title_generation_in_progress = True
        worker = TitleWorker(self._agent, first_user_message)
        worker.signals.done.connect(self._on_title_generated)
        worker.signals.failed.connect(self._on_title_generation_failed)
        self._thread_pool.start(worker)

    def _on_title_generated(self, title: str) -> None:
        self._title_generation_in_progress = False
        if self._current_conversation_id is None:
            return

        clean_title = title.strip() or self._conversation_title()
        config = self.settings_panel.get_config()
        if self._agent is None:
            return

        self._store.upsert_conversation(
            conversation_id=self._current_conversation_id,
            provider=config["provider"],
            model_name=config["model_name"],
            transcript=self._transcript_records,
            history_json=self._agent.export_history_json(),
            title=clean_title,
        )
        self._title_generated_for_conversations.add(self._current_conversation_id)
        self._refresh_conversation_list()

    def _on_title_generation_failed(self, _error_message: str) -> None:
        self._title_generation_in_progress = False

    def _clear_chat(self) -> None:
        self._transcript_records = []
        self.transcript.clear()
        if self._agent is not None:
            self._agent.clear_history()
        self._persist_active_conversation()
        self._refresh_graph_view()
        self.statusBar().showMessage("Conversation cleared")

    def _new_chat(self) -> None:
        """Create and activate a fresh conversation with current settings."""
        provider, model = self._default_provider_and_model()
        if not model:
            QMessageBox.warning(self, "No Model", "Select a provider and refresh models first.")
            return

        record = self._store.create_conversation(provider, model)
        self._refresh_conversation_list()
        self._activate_conversation(record.id)

    def _delete_current_chat(self) -> None:
        if not self._current_conversation_id:
            return
        self._store.delete_conversation(self._current_conversation_id)
        self._current_conversation_id = None
        self._load_initial_conversation()

    def _toggle_pin_current_chat(self) -> None:
        """Toggle pin state for the currently selected conversation."""
        if not self._current_conversation_id:
            return
        is_pinned = self._store.toggle_pin(self._current_conversation_id)
        self.statusBar().showMessage("Pinned" if is_pinned else "Unpinned")
        self._refresh_conversation_list()

    def _export_current_chat_markdown(self) -> None:
        """Export active conversation transcript to a markdown file."""
        if not self._current_conversation_id:
            return

        record = self._store.get_conversation(self._current_conversation_id)
        if record is None:
            return

        suggested = f"{record.title.replace('/', '-').replace(' ', '_')}.md"
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Conversation", suggested, "Markdown (*.md)")
        if not file_path:
            return

        markdown = self._store.export_markdown(self._current_conversation_id)
        if not markdown:
            QMessageBox.warning(self, "Export", "No conversation selected to export.")
            return

        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(markdown)
        self.statusBar().showMessage(f"Exported markdown: {file_path}")

    def _on_conversation_selected(self) -> None:
        if self._suppress_conversation_selection:
            return
        selected = self.conversation_list.selectedItems()
        if not selected:
            return
        item = selected[0]
        conversation_id = item.data(Qt.UserRole)
        if isinstance(conversation_id, str):
            self._activate_conversation(conversation_id)


def main() -> None:
    import sys

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("IBM Plex Sans", 10))

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#0f1117"))
    palette.setColor(QPalette.WindowText, QColor("#eef2f7"))
    palette.setColor(QPalette.Base, QColor("#10131a"))
    palette.setColor(QPalette.AlternateBase, QColor("#151923"))
    palette.setColor(QPalette.Text, QColor("#eef2f7"))
    palette.setColor(QPalette.Button, QColor("#151a24"))
    palette.setColor(QPalette.ButtonText, QColor("#eef2f7"))
    palette.setColor(QPalette.Highlight, QColor("#2f6fec"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ToolTipBase, QColor("#151923"))
    palette.setColor(QPalette.ToolTipText, QColor("#eef2f7"))
    app.setPalette(palette)

    window = ChatWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
