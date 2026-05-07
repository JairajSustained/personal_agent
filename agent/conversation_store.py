from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .chat_agent import Provider


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ConversationRecord:
    id: str
    title: str
    provider: str
    model_name: str
    created_at: str
    updated_at: str
    transcript: list[dict[str, str]]
    history_json: str
    pinned: bool = False


class ConversationStore:
    """JSON-backed conversation persistence and retrieval service."""

    def __init__(self, file_path: Path | None = None) -> None:
        base_dir = Path.home() / ".personal_agent"
        base_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = file_path or (base_dir / "conversations.json")

    def _default_state(self) -> dict:
        return {"active_id": None, "conversations": []}

    def load(self) -> dict:
        """Load raw store state from disk with safe defaults."""
        if not self.file_path.exists():
            return self._default_state()

        try:
            payload = json.loads(self.file_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return self._default_state()
            payload.setdefault("active_id", None)
            payload.setdefault("conversations", [])
            return payload
        except Exception:
            return self._default_state()

    def save(self, state: dict) -> None:
        """Persist the full store state to disk."""
        self.file_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def list_conversations(self) -> list[ConversationRecord]:
        """Return conversations sorted by pin status and update time."""
        state = self.load()
        items = state.get("conversations", [])
        records: list[ConversationRecord] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            records.append(
                ConversationRecord(
                    id=item.get("id", ""),
                    title=item.get("title", "New Chat"),
                    provider=item.get("provider", Provider.OPENAI.value),
                    model_name=item.get("model_name", ""),
                    created_at=item.get("created_at", _now_iso()),
                    updated_at=item.get("updated_at", _now_iso()),
                    transcript=item.get("transcript", []),
                    history_json=item.get("history_json", "[]"),
                    pinned=bool(item.get("pinned", False)),
                )
            )
        return sorted(records, key=lambda record: (record.pinned, record.updated_at), reverse=True)

    def create_conversation(self, provider: Provider, model_name: str) -> ConversationRecord:
        """Create and persist an empty conversation record."""
        now = _now_iso()
        record = ConversationRecord(
            id=uuid.uuid4().hex,
            title="New Chat",
            provider=provider.value,
            model_name=model_name,
            created_at=now,
            updated_at=now,
            transcript=[],
            history_json="[]",
        )

        state = self.load()
        state["conversations"].append(record.__dict__)
        state["active_id"] = record.id
        self.save(state)
        return record

    def upsert_conversation(
        self,
        conversation_id: str,
        provider: Provider,
        model_name: str,
        transcript: list[dict[str, str]],
        history_json: str,
        title: str | None = None,
    ) -> None:
        """Update an existing conversation or create it if missing."""
        state = self.load()
        conversations = state.get("conversations", [])
        updated = False
        for item in conversations:
            if not isinstance(item, dict) or item.get("id") != conversation_id:
                continue
            item["provider"] = provider.value
            item["model_name"] = model_name
            item["updated_at"] = _now_iso()
            item["transcript"] = transcript
            item["history_json"] = history_json
            item["pinned"] = bool(item.get("pinned", False))
            if title:
                item["title"] = title
            updated = True
            break

        if not updated:
            conversations.append(
                {
                    "id": conversation_id,
                    "title": title or "New Chat",
                    "provider": provider.value,
                    "model_name": model_name,
                    "created_at": _now_iso(),
                    "updated_at": _now_iso(),
                    "transcript": transcript,
                    "history_json": history_json,
                    "pinned": False,
                }
            )

        state["active_id"] = conversation_id
        self.save(state)

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and rebalance active selection."""
        state = self.load()
        conversations = state.get("conversations", [])
        conversations = [item for item in conversations if isinstance(item, dict) and item.get("id") != conversation_id]
        state["conversations"] = conversations
        if state.get("active_id") == conversation_id:
            state["active_id"] = conversations[0].get("id") if conversations else None
        self.save(state)

    def get_active_id(self) -> str | None:
        """Return the currently active conversation id, if any."""
        return self.load().get("active_id")

    def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        """Fetch one conversation record by id."""
        for record in self.list_conversations():
            if record.id == conversation_id:
                return record
        return None

    def toggle_pin(self, conversation_id: str) -> bool:
        """Toggle pinned status for a conversation and return new status."""
        state = self.load()
        conversations = state.get("conversations", [])
        for item in conversations:
            if not isinstance(item, dict) or item.get("id") != conversation_id:
                continue
            next_value = not bool(item.get("pinned", False))
            item["pinned"] = next_value
            item["updated_at"] = _now_iso()
            self.save(state)
            return next_value
        return False

    def export_markdown(self, conversation_id: str) -> str:
        """Export one conversation transcript as Markdown text."""
        record = self.get_conversation(conversation_id)
        if record is None:
            return ""

        lines = [
            f"# {record.title}",
            "",
            f"- Provider: `{record.provider}`",
            f"- Model: `{record.model_name}`",
            f"- Updated: `{record.updated_at}`",
            "",
            "---",
            "",
        ]

        for msg in record.transcript:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "System"))
            content = str(msg.get("content", ""))
            lines.append(f"## {role}")
            lines.append("")
            lines.append(content)
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        """Rename a conversation and return whether it was found."""
        clean_title = title.strip()
        if not clean_title:
            return False

        state = self.load()
        for item in state.get("conversations", []):
            if not isinstance(item, dict) or item.get("id") != conversation_id:
                continue
            item["title"] = clean_title[:120]
            item["updated_at"] = _now_iso()
            self.save(state)
            return True
        return False

    def duplicate_conversation(self, conversation_id: str) -> ConversationRecord | None:
        """Create a copy of an existing conversation and make the copy active."""
        state = self.load()
        conversations = state.get("conversations", [])
        source = next(
            (item for item in conversations if isinstance(item, dict) and item.get("id") == conversation_id),
            None,
        )
        if source is None:
            return None

        now = _now_iso()
        copied = dict(source)
        copied["id"] = uuid.uuid4().hex
        copied["title"] = f"{source.get('title', 'New Chat')} (copy)"
        copied["created_at"] = now
        copied["updated_at"] = now
        copied["transcript"] = list(source.get("transcript", []))
        copied["pinned"] = False
        conversations.append(copied)
        state["active_id"] = copied["id"]
        self.save(state)
        return self.get_conversation(copied["id"])

    def export_all_json(self) -> str:
        """Export the complete conversation store as formatted JSON."""
        return json.dumps(self.load(), indent=2, ensure_ascii=False)

    def import_from_json(self, payload: str) -> int:
        """Merge conversations from an exported JSON payload and return import count."""
        try:
            imported_state = json.loads(payload)
        except json.JSONDecodeError:
            return 0
        if not isinstance(imported_state, dict):
            return 0

        incoming = imported_state.get("conversations", [])
        if not isinstance(incoming, list):
            return 0

        state = self.load()
        conversations = state.get("conversations", [])
        existing_ids = {item.get("id") for item in conversations if isinstance(item, dict)}
        imported_count = 0
        for item in incoming:
            if not isinstance(item, dict):
                continue
            copied = dict(item)
            copied_id = str(copied.get("id") or uuid.uuid4().hex)
            if copied_id in existing_ids:
                copied_id = uuid.uuid4().hex
            copied["id"] = copied_id
            copied.setdefault("title", "Imported Chat")
            copied.setdefault("provider", Provider.OPENAI.value)
            copied.setdefault("model_name", "")
            copied.setdefault("created_at", _now_iso())
            copied.setdefault("updated_at", _now_iso())
            copied.setdefault("transcript", [])
            copied.setdefault("history_json", "[]")
            copied["pinned"] = bool(copied.get("pinned", False))
            conversations.append(copied)
            existing_ids.add(copied_id)
            imported_count += 1

        if imported_count:
            state["active_id"] = conversations[-1].get("id")
            self.save(state)
        return imported_count

    def conversation_stats(self, conversation_id: str) -> dict[str, int]:
        """Return basic message and word counts for one conversation."""
        record = self.get_conversation(conversation_id)
        if record is None:
            return {"messages": 0, "user_messages": 0, "assistant_messages": 0, "words": 0}

        user_messages = 0
        assistant_messages = 0
        words = 0
        for msg in record.transcript:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", ""))
            if role == "You":
                user_messages += 1
            elif role == "Assistant":
                assistant_messages += 1
            words += len(str(msg.get("content", "")).split())
        return {
            "messages": len(record.transcript),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "words": words,
        }

    def search_conversations(self, query: str) -> list[ConversationRecord]:
        """Search conversations by title, metadata, and transcript text."""
        term = query.strip().lower()
        if not term:
            return self.list_conversations()

        matches: list[ConversationRecord] = []
        for record in self.list_conversations():
            if term in record.title.lower():
                matches.append(record)
                continue

            if term in record.provider.lower() or term in record.model_name.lower():
                matches.append(record)
                continue

            transcript_text = " ".join(
                msg.get("content", "") for msg in record.transcript if isinstance(msg, dict)
            ).lower()
            if term in transcript_text:
                matches.append(record)

        return matches
