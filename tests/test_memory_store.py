from __future__ import annotations

from pathlib import Path

from agent import MemoryStore


def test_memory_store_save_and_load(tmp_path: Path) -> None:
    store = MemoryStore(file_path=tmp_path / "memory.txt")
    assert store.load_text() == ""

    store.save_text("User prefers concise answers")
    assert store.load_text() == "User prefers concise answers"


def test_memory_store_uses_env_override_path(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / "custom-memory.txt"
    monkeypatch.setenv("MEMORY_FILE_PATH", str(env_path))

    store = MemoryStore()
    assert store.file_path == env_path

    store.save_text("Name: Om")
    assert env_path.read_text(encoding="utf-8") == "Name: Om"
