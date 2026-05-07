from __future__ import annotations

from pathlib import Path

from agent import MemoryStore


class _FakeNeo4jSession:
    def __init__(self, driver: _FakeNeo4jDriver) -> None:
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        return False

    def run(self, query: str, **params):
        self.driver.queries.append(query)
        self.driver.params.append(params)
        if self.driver.fail_on_schema and "CONSTRAINT" in query:
            raise RuntimeError("schema failed")

        user_id = params.get("user_id", "default")
        if "RETURN m.text AS text" in query:
            return [{"text": fact["text"]} for fact in self.driver.facts_by_user.get(user_id, [])]
        if "UNWIND $facts AS fact" in query:
            self.driver.facts_by_user[user_id] = list(params["facts"])
        if "WHERE NOT old.key IN $keys" in query:
            keys = set(params["keys"])
            self.driver.facts_by_user[user_id] = [
                fact for fact in self.driver.facts_by_user.get(user_id, []) if fact["key"] in keys
            ]
        return []


class _FakeNeo4jDriver:
    def __init__(self, fail_on_schema: bool = False) -> None:
        self.facts_by_user: dict[str, list[dict[str, str]]] = {}
        self.queries: list[str] = []
        self.params: list[dict] = []
        self.closed = False
        self.fail_on_schema = fail_on_schema

    @property
    def facts(self) -> list[dict[str, str]]:
        return self.facts_by_user.get("default", [])

    @facts.setter
    def facts(self, value: list[dict[str, str]]) -> None:
        self.facts_by_user["default"] = value

    def verify_connectivity(self) -> None:
        return None

    def session(self, **_kwargs) -> _FakeNeo4jSession:
        return _FakeNeo4jSession(self)

    def close(self) -> None:
        self.closed = True


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

    store.save_text("Name: Alice")
    assert env_path.read_text(encoding="utf-8") == "Name: Alice"


def test_file_memory_store_returns_relevant_facts(tmp_path: Path) -> None:
    store = MemoryStore(file_path=tmp_path / "memory.txt")
    store.save_text("- Name: Alice\n- Preference: likes concise answers\n- Location: Delhi\n")

    relevant = store.load_relevant_text("please answer concise", limit=2)

    assert "Preference: likes concise answers" in relevant
    assert "Location: Delhi" not in relevant


def test_neo4j_memory_store_saves_graph_facts_and_keeps_file_mirror(
    monkeypatch, tmp_path: Path
) -> None:
    fake_driver = _FakeNeo4jDriver()
    monkeypatch.setenv("MEMORY_BACKEND", "neo4j")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("MEMORY_FILE_PATH", str(tmp_path / "memory.txt"))
    monkeypatch.setattr(MemoryStore, "_create_neo4j_driver", lambda *_args: fake_driver)

    store = MemoryStore()
    store.save_text("- Name: Alice\n- Preference: likes concise answers\n")

    assert store.backend_name == "neo4j"
    assert fake_driver.facts == [
        {"key": "name: alice", "text": "Name: Alice", "category": "name"},
        {
            "key": "preference: likes concise answers",
            "text": "Preference: likes concise answers",
            "category": "preference",
        },
    ]
    all_queries = "\n".join(fake_driver.queries)
    assert "MemoryCategory" in all_queries
    assert "REQUIRE (m.user_id, m.key) IS UNIQUE" in all_queries
    assert "MERGE (m:MemoryFact {user_id: $user_id, key: fact.key})" in all_queries
    assert (tmp_path / "memory.txt").read_text(encoding="utf-8") == (
        "- Name: Alice\n- Preference: likes concise answers\n"
    )


def test_neo4j_memory_store_loads_graph_facts(monkeypatch, tmp_path: Path) -> None:
    fake_driver = _FakeNeo4jDriver()
    fake_driver.facts = [{"text": "Name: Alice"}, {"text": "Preference: likes concise answers"}]
    monkeypatch.setenv("MEMORY_BACKEND", "neo4j")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("MEMORY_FILE_PATH", str(tmp_path / "memory.txt"))
    monkeypatch.setattr(MemoryStore, "_create_neo4j_driver", lambda *_args: fake_driver)

    store = MemoryStore()

    assert store.load_text() == "- Name: Alice\n- Preference: likes concise answers\n"


def test_neo4j_memory_store_scopes_facts_by_user(monkeypatch, tmp_path: Path) -> None:
    fake_driver = _FakeNeo4jDriver()
    monkeypatch.setenv("MEMORY_BACKEND", "neo4j")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("MEMORY_FILE_PATH", str(tmp_path / "memory.txt"))
    monkeypatch.setattr(MemoryStore, "_create_neo4j_driver", lambda *_args: fake_driver)

    monkeypatch.setenv("NEO4J_USER_ID", "user-a")
    first_store = MemoryStore()
    first_store.save_text("- Name: Alice\n")

    monkeypatch.setenv("NEO4J_USER_ID", "user-b")
    second_store = MemoryStore()
    second_store.save_text("- Name: Alice\n- Preference: detailed answers\n")

    assert fake_driver.facts_by_user["user-a"] == [
        {"key": "name: alice", "text": "Name: Alice", "category": "name"}
    ]
    assert fake_driver.facts_by_user["user-b"] == [
        {"key": "name: alice", "text": "Name: Alice", "category": "name"},
        {
            "key": "preference: detailed answers",
            "text": "Preference: detailed answers",
            "category": "preference",
        },
    ]


def test_explicit_file_path_can_still_use_neo4j_when_configured(monkeypatch, tmp_path: Path) -> None:
    fake_driver = _FakeNeo4jDriver()
    monkeypatch.setenv("MEMORY_BACKEND", "neo4j")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setattr(MemoryStore, "_create_neo4j_driver", lambda *_args: fake_driver)

    store = MemoryStore(file_path=tmp_path / "memory.txt")

    assert store.backend_name == "neo4j"


def test_neo4j_memory_store_close_closes_driver(monkeypatch, tmp_path: Path) -> None:
    fake_driver = _FakeNeo4jDriver()
    monkeypatch.setenv("MEMORY_BACKEND", "neo4j")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("MEMORY_FILE_PATH", str(tmp_path / "memory.txt"))
    monkeypatch.setattr(MemoryStore, "_create_neo4j_driver", lambda *_args: fake_driver)

    store = MemoryStore()
    store.close()

    assert fake_driver.closed is True
    assert store.backend_name == "file"


def test_neo4j_driver_is_closed_when_schema_setup_fails(monkeypatch, tmp_path: Path) -> None:
    fake_driver = _FakeNeo4jDriver(fail_on_schema=True)
    monkeypatch.setenv("MEMORY_BACKEND", "neo4j")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("MEMORY_FILE_PATH", str(tmp_path / "memory.txt"))
    monkeypatch.setattr(MemoryStore, "_create_neo4j_driver", lambda *_args: fake_driver)

    store = MemoryStore()

    assert store.backend_name == "file"
    assert fake_driver.closed is True
    assert store.neo4j_unavailable_reason == "schema failed"
