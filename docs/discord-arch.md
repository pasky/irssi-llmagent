# Discord Frontend Design

## Status
Implemented with full IRC feature parity. The Discord frontend is now in **beta**, purely because it needs more real-world usage testing.

## Decisions (Approved)
- **Triggering**: Only on **@mentions** (DMs count as mentions).
- **Library**: **discord.py** (2.x) for long‑term maintenance stability.
- **Arc naming**: Use **human‑readable** identifiers (e.g., `discord:{guild_name}#{channel_name}`).
- **Channel scoping**: All channels by default (no allowlist required initially).
- **Replies**: Respond as **replies** to the original Discord message.
- **Thread isolation**: Thread content (not starting messages) is isolated context for all purposes except the chronicle.
- **Prompts/models**: Reuse the **IRC command prompt/model configuration verbatim**.

## Scope
- **Agentic actor trigger** on Discord @mention (including DMs).
- Store messages in `ChatHistory` with readable Discord arcs.
- Send responses back to the channel as replies to the original message.
- Reuse the IRC command prompt/model configuration verbatim (including modes).
- Feature parity for IRC commands (parsing, mode classification, overrides, help).
- Proactive interjection, debouncing, auto-chronicling, cost followups.
- Inline images/attachments processing.

## Future Considerations
- Introduce a shared `RoomMessageEvent` abstraction to support Slack.
