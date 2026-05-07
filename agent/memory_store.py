from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


class MemoryStore:
    """Persistent memory store with file fallback and optional Neo4j backing.

    By default memory stays in ``~/.personal_agent/memory.txt``. If Neo4j is
    explicitly configured with ``MEMORY_BACKEND=neo4j`` or ``NEO4J_URI``, facts
    are also stored as graph nodes connected to a single user node:

    ``(:PersonalAgentUser {id})-[:REMEMBERS]->(:MemoryFact {user_id, key, text, category})``
    """

    def __init__(self, file_path: Path | None = None, enable_neo4j: bool | None = None) -> None:
        default_path = Path.home() / ".personal_agent" / "memory.txt"
        env_path = (os.getenv("MEMORY_FILE_PATH") or "").strip()
        selected = Path(env_path).expanduser() if env_path else default_path
        self.file_path = file_path or selected
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        self.backend_name = "file"
        self.neo4j_unavailable_reason = ""
        self._neo4j_driver: Any | None = None
        self._neo4j_database = (os.getenv("NEO4J_DATABASE") or "").strip() or None
        self._neo4j_user_id = (os.getenv("NEO4J_USER_ID") or "default").strip() or "default"

        wants_neo4j = (os.getenv("MEMORY_BACKEND") or "").strip().lower() == "neo4j" or bool(
            (os.getenv("NEO4J_URI") or "").strip()
        )
        should_enable_neo4j = wants_neo4j if enable_neo4j is None else enable_neo4j
        if should_enable_neo4j:
            self._enable_neo4j_if_available()

    def _enable_neo4j_if_available(self) -> None:
        uri = (os.getenv("NEO4J_URI") or "").strip()
        user = (os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or "neo4j").strip()
        password = (os.getenv("NEO4J_PASSWORD") or "").strip()

        if not uri:
            self.neo4j_unavailable_reason = "NEO4J_URI is not set"
            return
        if not password:
            self.neo4j_unavailable_reason = "NEO4J_PASSWORD is not set"
            return

        try:
            self._neo4j_driver = self._create_neo4j_driver(uri, user, password)
            verify = getattr(self._neo4j_driver, "verify_connectivity", None)
            if callable(verify):
                verify()
            self._ensure_neo4j_schema()
            self.backend_name = "neo4j"
        except Exception as exc:  # noqa: BLE001
            self.neo4j_unavailable_reason = str(exc)
            self.close()

    @staticmethod
    def _create_neo4j_driver(uri: str, user: str, password: str):
        from neo4j import GraphDatabase

        return GraphDatabase.driver(uri, auth=(user, password), connection_timeout=2.0)

    def _session(self):
        if self._neo4j_driver is None:
            raise RuntimeError("Neo4j is not enabled")
        if self._neo4j_database:
            return self._neo4j_driver.session(database=self._neo4j_database)
        return self._neo4j_driver.session()

    def _ensure_neo4j_schema(self) -> None:
        queries = (
            "CREATE CONSTRAINT personal_agent_user_id IF NOT EXISTS "
            "FOR (u:PersonalAgentUser) REQUIRE u.id IS UNIQUE",
            "DROP CONSTRAINT memory_fact_key IF EXISTS",
            "CREATE CONSTRAINT memory_fact_user_key IF NOT EXISTS "
            "FOR (m:MemoryFact) REQUIRE (m.user_id, m.key) IS UNIQUE",
        )
        with self._session() as session:
            for query in queries:
                session.run(query)

    def close(self) -> None:
        if self._neo4j_driver is not None:
            close = getattr(self._neo4j_driver, "close", None)
            if callable(close):
                close()
        self._neo4j_driver = None
        self.backend_name = "file"

    @staticmethod
    def _parse_memory_facts(text: str) -> list[dict[str, str]]:
        facts: list[dict[str, str]] = []
        seen: set[str] = set()
        for raw in text.splitlines():
            fact_text = raw.strip().lstrip("-").strip()
            if not fact_text:
                continue
            key = re.sub(r"\s+", " ", fact_text).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            category = fact_text.split(":", 1)[0].strip().lower() if ":" in fact_text else "note"
            facts.append({"key": key[:240], "text": fact_text[:240], "category": category[:48]})
        return facts

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "do",
            "for",
            "i",
            "in",
            "is",
            "it",
            "me",
            "my",
            "of",
            "on",
            "or",
            "the",
            "to",
            "use",
            "what",
            "you",
        }
        return {token for token in re.findall(r"[a-z0-9']+", text.lower()) if token not in stop_words}

    @classmethod
    def _rank_facts(cls, facts: list[str], query: str, limit: int) -> list[str]:
        query_tokens = cls._tokenize(query)
        if not query_tokens:
            return facts[:limit]

        scored: list[tuple[int, int, str]] = []
        token_counts = Counter(query_tokens)
        for idx, fact in enumerate(facts):
            fact_tokens = cls._tokenize(fact)
            overlap = sum(token_counts[token] for token in fact_tokens & query_tokens)
            exact_bonus = 2 if query.lower() in fact.lower() else 0
            score = overlap + exact_bonus
            if score > 0:
                scored.append((score, -idx, fact))

        return [fact for _score, _idx, fact in sorted(scored, reverse=True)[:limit]]

    def _load_text_from_file(self) -> str:
        if not self.file_path.exists():
            return ""
        try:
            return self.file_path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def load_text(self) -> str:
        if self.backend_name == "neo4j" and self._neo4j_driver is not None:
            try:
                with self._session() as session:
                    result = session.run(
                        """
                        MATCH (:PersonalAgentUser {id: $user_id})-[:REMEMBERS]->(m:MemoryFact)
                        RETURN m.text AS text
                        ORDER BY coalesce(m.updated_at, datetime({epochMillis: 0})), m.text
                        """,
                        user_id=self._neo4j_user_id,
                    )
                    lines = [record["text"] for record in result if record.get("text")]
                return "" if not lines else "\n".join(f"- {line}" for line in lines) + "\n"
            except Exception as exc:  # noqa: BLE001
                self.neo4j_unavailable_reason = str(exc)

        return self._load_text_from_file()

    def _save_text_to_file(self, text: str) -> None:
        self.file_path.write_text(text, encoding="utf-8")

    def load_relevant_text(self, query: str, limit: int = 12) -> str:
        """Return the most relevant memory facts for a prompt.

        This is intentionally lexical and local: Neo4j provides graph persistence,
        while deterministic token overlap keeps retrieval dependency-free and
        testable. If no facts match, callers can fall back to full memory.
        """
        query = query.strip()
        if not query:
            return self.load_text()

        if self.backend_name == "neo4j" and self._neo4j_driver is not None:
            try:
                query_tokens = sorted(self._tokenize(query))
                with self._session() as session:
                    result = session.run(
                        """
                        MATCH (:PersonalAgentUser {id: $user_id})-[:REMEMBERS]->(m:MemoryFact)
                        WITH m, [token IN $tokens WHERE toLower(m.text) CONTAINS token] AS hits
                        WITH m, size(hits) AS score
                        WHERE score > 0
                        RETURN m.text AS text
                        ORDER BY score DESC, coalesce(m.updated_at, datetime({epochMillis: 0})) DESC
                        LIMIT $limit
                        """,
                        user_id=self._neo4j_user_id,
                        tokens=query_tokens,
                        limit=limit,
                    )
                    lines = [record["text"] for record in result if record.get("text")]
                return "" if not lines else "\n".join(f"- {line}" for line in lines) + "\n"
            except Exception as exc:  # noqa: BLE001
                self.neo4j_unavailable_reason = str(exc)

        all_facts = [fact["text"] for fact in self._parse_memory_facts(self._load_text_from_file())]
        ranked = self._rank_facts(all_facts, query, limit)
        return "" if not ranked else "\n".join(f"- {line}" for line in ranked) + "\n"

    def save_text(self, text: str) -> None:
        if self.backend_name == "neo4j" and self._neo4j_driver is not None:
            facts = self._parse_memory_facts(text)
            try:
                with self._session() as session:
                    session.run(
                        """
                        MERGE (u:PersonalAgentUser {id: $user_id})
                        SET u.updated_at = datetime()
                        WITH u
                        OPTIONAL MATCH (u)-[r:REMEMBERS]->(old:MemoryFact)
                        WHERE old.user_id IS NULL OR old.user_id <> $user_id OR NOT old.key IN $keys
                        DELETE r
                        WITH collect(old) AS old_facts
                        UNWIND old_facts AS old_fact
                        WITH old_fact
                        WHERE old_fact IS NOT NULL AND NOT (old_fact)<-[:REMEMBERS]-(:PersonalAgentUser)
                        DETACH DELETE old_fact
                        """,
                        user_id=self._neo4j_user_id,
                        keys=[fact["key"] for fact in facts],
                    )
                    session.run(
                        """
                        MERGE (u:PersonalAgentUser {id: $user_id})
                        WITH u
                        UNWIND $facts AS fact
                        MERGE (m:MemoryFact {user_id: $user_id, key: fact.key})
                        SET m.text = fact.text,
                            m.category = fact.category,
                            m.updated_at = datetime()
                        MERGE (c:MemoryCategory {name: fact.category})
                        MERGE (u)-[:REMEMBERS]->(m)
                        MERGE (m)-[:IN_CATEGORY]->(c)
                        """,
                        user_id=self._neo4j_user_id,
                        facts=facts,
                    )
            except Exception as exc:  # noqa: BLE001
                self.neo4j_unavailable_reason = str(exc)

        # Always keep a plain-text mirror so the app remains usable if Neo4j is unavailable later.
        self._save_text_to_file(text)
