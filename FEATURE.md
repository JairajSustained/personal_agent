# Suggested Features

Features ordered by user value and implementation effort.

## 1. Memory-only graph view
**Why:** The current graph mixes memory facts with raw transcript messages, making it noisy and confusing. The graph should show only persisted memory facts, organized by category.

**Status:** Implemented.

## 2. Web search tool
**Why:** The model can answer factual questions from training data but cannot look up current events. A `search_web` pydantic-ai tool gives the agent live context without requiring the user to copy-paste from a browser.

**Status:** Implemented.

## 3. File attachment
**Why:** Users often need to share code, documents, or logs as context. An "Attach" button in the composer reads a file and prepends its content to the message.

**Status:** Implemented.

## 4. Multiline composer
**Why:** Shift+Enter newlines make it practical to write multi-paragraph prompts, pastes, and structured instructions. The current single-line input truncates context.

**Status:** Not started.

## 5. Context length indicator
**Why:** Long conversations consume model context. A live message count in the status bar tells the user when to clear and start fresh to avoid degraded responses.

**Status:** Not started.

## 6. Conversation rename via double-click
**Why:** Double-clicking a title to rename inline is a standard desktop UX pattern. Reduces friction vs. navigating to a settings action.

**Status:** Not started.

## 7. Memory export / import
**Why:** Lets users back up, share, or migrate their memory file independently of conversations. Complements the existing JSON backup story.

**Status:** Not started.

## 8. Model-tagged assistant messages
**Why:** When switching providers mid-session, users lose track of which model produced which answer. A small badge (provider / model) on each assistant bubble adds accountability.

**Status:** Not started.

## 9. Font size control
**Why:** Ctrl+= / Ctrl+- to increase and decrease the chat transcript font size. Accessibility basic; also useful for screen sharing or presentation.

**Status:** Not started.

## 10. Pinned memory facts sidebar
**Why:** A read-only panel showing the top few memory facts at a glance. Removes the need to open the graph or edit dialog to see what Dost knows.

**Status:** Not started.

---

## Verification

```bash
uv run pytest -q
uv run ruff check .
```
