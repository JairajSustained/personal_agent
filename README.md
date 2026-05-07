# Personal Agent

Production-oriented desktop chat application built with Python, PySide6, and pydantic-ai.

![Personal Agent screenshot](docs/screenshot.png)

## Supported providers

- OpenAI
- Anthropic
- Google (Gemini)
- Azure Foundry (OpenAI-compatible model endpoint)

## Quick start

1. Install dependencies:

```bash
uv sync
```

2. Create environment file:

```bash
cp .env.example .env
```

3. Populate `.env` with provider credentials.

4. Run the app:

```bash
uv run personal-agent
```

## Production notes

- Configuration validation fails fast with clear messages for missing keys/endpoints.
- Chat requests run in a background worker to keep the UI responsive.
- Background event loops are shut down cleanly to avoid pending-task warnings.
- Session history is protected with a lock for safe multi-threaded access.
- Conversations are persisted to `~/.personal_agent/conversations.json` and can be reopened.
- Conversation titles start from first prompt, then get model-refined automatically.
- Conversation list supports full-text search across titles and message content.
- Chats can be pinned/favorited and exported as Markdown.
- Persistent memory is stored in a local text file and injected into each new conversation.
- Provider/model can be switched at runtime from the settings panel.

## Memory

- Default memory file: `~/.personal_agent/memory.txt`
- Optional override: `MEMORY_FILE_PATH`
- Memory is updated automatically by the agent from conversation turns.
- Chat turns retrieve relevant memory facts for the current prompt using deterministic lexical ranking.
- New chats inherit memory context automatically.

### Optional Neo4j graph memory

By default the app uses the plain-text memory file. To persist memory facts in Neo4j as graph-backed memory, set:

```bash
MEMORY_BACKEND=neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>
# Optional:
NEO4J_DATABASE=neo4j
NEO4J_USER_ID=default
```

Neo4j stores facts like:

```cypher
(:PersonalAgentUser {id})-[:REMEMBERS]->(:MemoryFact {user_id, key})-[:IN_CATEGORY]->(:MemoryCategory)
```

A plain-text mirror is still written to `MEMORY_FILE_PATH` so memory survives if Neo4j is unavailable. `NEO4J_USER_ID` scopes fact nodes per user.

## Documentation

- Architecture: `docs/ARCHITECTURE.md`
- Features: `docs/FEATURES.md`

## LibreChat-style model config

If you want model lists from a LibreChat config, set:

- `LIBRECHAT_CONFIG_PATH=/absolute/path/to/librechat.yaml`

Azure deployment names defined in that file are used as model options when runtime discovery is unavailable.

## Azure Foundry configuration

Set these variables for Azure Foundry models:

- `AZURE_FOUNDRY_API_KEY` (or `AZURE_OPENAI_API_KEY` or `AZURE_API_KEY`)
- `AZURE_FOUNDRY_ENDPOINT` (or `AZURE_ENDPOINT`)
- Optional: `AZURE_API_VERSION`

Example endpoint:

```text
https://<resource>.services.ai.azure.com/models
```
