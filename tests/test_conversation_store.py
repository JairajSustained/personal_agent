from __future__ import annotations

from pathlib import Path

from agent import ConversationStore, Provider


def test_create_and_upsert_conversation(tmp_path: Path) -> None:
    store = ConversationStore(file_path=tmp_path / "conversations.json")

    created = store.create_conversation(Provider.OPENAI, "gpt-4o")
    assert created.model_name == "gpt-4o"

    store.upsert_conversation(
        conversation_id=created.id,
        provider=Provider.OPENAI,
        model_name="gpt-4o",
        transcript=[{"role": "You", "content": "hello"}],
        history_json="[]",
        title="Hello",
    )

    records = store.list_conversations()
    assert len(records) == 1
    assert records[0].title == "Hello"
    assert records[0].transcript[0]["content"] == "hello"


def test_delete_conversation(tmp_path: Path) -> None:
    store = ConversationStore(file_path=tmp_path / "conversations.json")
    first = store.create_conversation(Provider.OPENAI, "gpt-4o")
    second = store.create_conversation(Provider.OPENAI, "gpt-4o-mini")

    store.delete_conversation(second.id)

    records = store.list_conversations()
    assert len(records) == 1
    assert records[0].id == first.id


def test_search_conversations_matches_title_and_content(tmp_path: Path) -> None:
    store = ConversationStore(file_path=tmp_path / "conversations.json")
    first = store.create_conversation(Provider.OPENAI, "gpt-4o")
    second = store.create_conversation(Provider.ANTHROPIC, "claude-sonnet-4-6")

    store.upsert_conversation(
        conversation_id=first.id,
        provider=Provider.OPENAI,
        model_name="gpt-4o",
        transcript=[{"role": "You", "content": "Need deployment checklist"}],
        history_json="[]",
        title="Azure rollout",
    )
    store.upsert_conversation(
        conversation_id=second.id,
        provider=Provider.ANTHROPIC,
        model_name="claude-sonnet-4-6",
        transcript=[{"role": "You", "content": "Draft product brief"}],
        history_json="[]",
        title="Product planning",
    )

    by_title = store.search_conversations("azure")
    assert len(by_title) == 1
    assert by_title[0].id == first.id

    by_content = store.search_conversations("brief")
    assert len(by_content) == 1
    assert by_content[0].id == second.id


def test_search_conversations_matches_provider_and_model_metadata(tmp_path: Path) -> None:
    store = ConversationStore(file_path=tmp_path / "conversations.json")
    first = store.create_conversation(Provider.OPENAI, "gpt-4o")
    second = store.create_conversation(Provider.ANTHROPIC, "claude-sonnet-4-6")

    by_provider = store.search_conversations("anthropic")
    assert len(by_provider) == 1
    assert by_provider[0].id == second.id

    by_model = store.search_conversations("gpt-4o")
    assert len(by_model) == 1
    assert by_model[0].id == first.id


def test_toggle_pin_and_sort_order(tmp_path: Path) -> None:
    store = ConversationStore(file_path=tmp_path / "conversations.json")
    first = store.create_conversation(Provider.OPENAI, "gpt-4o")
    second = store.create_conversation(Provider.OPENAI, "gpt-4o-mini")

    now_pinned = store.toggle_pin(first.id)
    assert now_pinned is True

    listed = store.list_conversations()
    assert listed[0].id == first.id
    assert listed[0].pinned is True

    now_unpinned = store.toggle_pin(first.id)
    assert now_unpinned is False
    listed_again = store.list_conversations()
    assert {listed_again[0].id, listed_again[1].id} == {first.id, second.id}
    assert listed_again[0].pinned is False


def test_export_markdown(tmp_path: Path) -> None:
    store = ConversationStore(file_path=tmp_path / "conversations.json")
    record = store.create_conversation(Provider.OPENAI, "gpt-4o")
    store.upsert_conversation(
        conversation_id=record.id,
        provider=Provider.OPENAI,
        model_name="gpt-4o",
        transcript=[
            {"role": "You", "content": "Hello"},
            {"role": "Assistant", "content": "Hi there"},
        ],
        history_json="[]",
        title="Greeting",
    )

    markdown = store.export_markdown(record.id)

    assert "# Greeting" in markdown
    assert "## You" in markdown
    assert "Hello" in markdown


def test_delete_active_conversation_rebalances_active_id(tmp_path: Path) -> None:
    store = ConversationStore(file_path=tmp_path / "conversations.json")
    first = store.create_conversation(Provider.OPENAI, "gpt-4o")
    second = store.create_conversation(Provider.OPENAI, "gpt-4o-mini")
    assert store.get_active_id() == second.id

    store.delete_conversation(second.id)

    assert store.get_active_id() == first.id


def test_export_markdown_returns_empty_for_unknown_conversation(tmp_path: Path) -> None:
    store = ConversationStore(file_path=tmp_path / "conversations.json")
    assert store.export_markdown("missing-id") == ""


def test_rename_and_duplicate_conversation(tmp_path: Path) -> None:
    store = ConversationStore(file_path=tmp_path / "conversations.json")
    record = store.create_conversation(Provider.OPENAI, "gpt-4o")
    store.upsert_conversation(
        conversation_id=record.id,
        provider=Provider.OPENAI,
        model_name="gpt-4o",
        transcript=[{"role": "You", "content": "Original prompt"}],
        history_json="[]",
        title="Original",
    )

    assert store.rename_conversation(record.id, "Renamed") is True
    duplicate = store.duplicate_conversation(record.id)

    assert store.get_conversation(record.id).title == "Renamed"
    assert duplicate is not None
    assert duplicate.id != record.id
    assert duplicate.title == "Renamed (copy)"
    assert duplicate.transcript == [{"role": "You", "content": "Original prompt"}]


def test_export_all_json_and_import_merge(tmp_path: Path) -> None:
    source = ConversationStore(file_path=tmp_path / "source.json")
    record = source.create_conversation(Provider.OPENAI, "gpt-4o")
    source.upsert_conversation(
        conversation_id=record.id,
        provider=Provider.OPENAI,
        model_name="gpt-4o",
        transcript=[{"role": "You", "content": "Archive this"}],
        history_json="[]",
        title="Archive",
    )

    exported = source.export_all_json()
    target = ConversationStore(file_path=tmp_path / "target.json")
    imported = target.import_from_json(exported)

    assert imported == 1
    imported_records = target.list_conversations()
    assert len(imported_records) == 1
    assert imported_records[0].title == "Archive"
    assert imported_records[0].transcript[0]["content"] == "Archive this"


def test_conversation_stats_counts_messages_and_words(tmp_path: Path) -> None:
    store = ConversationStore(file_path=tmp_path / "conversations.json")
    record = store.create_conversation(Provider.OPENAI, "gpt-4o")
    store.upsert_conversation(
        conversation_id=record.id,
        provider=Provider.OPENAI,
        model_name="gpt-4o",
        transcript=[
            {"role": "You", "content": "hello there"},
            {"role": "Assistant", "content": "general kenobi"},
        ],
        history_json="[]",
        title="Stats",
    )

    stats = store.conversation_stats(record.id)

    assert stats == {"messages": 2, "user_messages": 1, "assistant_messages": 1, "words": 4}
