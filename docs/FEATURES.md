# Features

## Chat and providers

- Multi-provider support: OpenAI, Anthropic, Google, Azure Foundry.
- Streaming assistant responses.
- Runtime provider/model switching from settings panel.

## Azure behavior

- Uses deployment names for chat requests.
- Supports env aliases:
  - `AZURE_API_KEY`, `AZURE_FOUNDRY_API_KEY`, `AZURE_OPENAI_API_KEY`
  - `AZURE_ENDPOINT`, `AZURE_FOUNDRY_ENDPOINT`
- `Refresh Models` refetches deployments/models.

## LibreChat-style configuration

- Optional `LIBRECHAT_CONFIG_PATH` points to a `librechat.yaml` file.
- Azure deployment/model preferences can be sourced from yaml when live lookup is unavailable.

## Conversation management

- Multiple conversations with persisted state.
- Search across conversation titles, model/provider metadata, and transcript content.
- Pin/unpin conversations to keep important threads at top.
- Export current conversation as Markdown.
- Toggle between chat view and a lightweight memory graph view.

## Memory

- File-based persistent memory in plain text.
- Automatically updated by the agent after chat turns.
- New conversations automatically include full memory context in agent instructions.

## UX

- Sidebar for conversations and settings.
- macOS-inspired frosted/glass visual language for panels.
- Background workers prevent UI blocking.
- Graceful async shutdown avoids pending-task warnings.
