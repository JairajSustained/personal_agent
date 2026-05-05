from __future__ import annotations

import os
from pathlib import Path


class MemoryStore:
    """File-based memory store shared across conversations."""

    def __init__(self, file_path: Path | None = None) -> None:
        default_path = Path.home() / ".personal_agent" / "memory.txt"
        env_path = (os.getenv("MEMORY_FILE_PATH") or "").strip()
        selected = Path(env_path).expanduser() if env_path else default_path
        self.file_path = file_path or selected
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def load_text(self) -> str:
        if not self.file_path.exists():
            return ""
        try:
            return self.file_path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def save_text(self, text: str) -> None:
        self.file_path.write_text(text, encoding="utf-8")
