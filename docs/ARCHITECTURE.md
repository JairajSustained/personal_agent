# Architecture

## High-level structure

- `main.py`: app entrypoint and logging setup.
- `gui/chat_window.py`: Qt desktop UI, user interactions, background workers.
  - includes `MemoryGraphWidget` for a visual context graph.
- `agent/chat_agent.py`: provider abstraction, model discovery, request/stream handling.
- `agent/conversation_store.py`: JSON persistence for conversations/messages/history.
- `agent/memory_store.py`: text-file persistence for long-term memory.
- `agent/librechat_config.py`: optional LibreChat-style yaml model/deployment parsing.

## Runtime flow

1. App starts and loads saved memory (`~/.personal_agent/memory.txt`).
2. App loads conversation list from `~/.personal_agent/conversations.json`.
3. UI selects active conversation, rehydrates transcript + model history.
4. User sends a message; UI worker streams response in a background event loop.
5. Transcript + model history are persisted after each response.
6. First exchange triggers async auto-title generation.

## Provider/model resolution

- Providers shown in UI are filtered by available env credentials.
- Models are resolved in this order:
  1. Live provider discovery
  2. LibreChat yaml fallback (if configured)
  3. Static catalog (non-Azure)
- Azure prefers deployment discovery and uses deployment names for requests.

## Storage files

- Conversations: `~/.personal_agent/conversations.json`
- Memory: `~/.personal_agent/memory.txt` (or `MEMORY_FILE_PATH`)
