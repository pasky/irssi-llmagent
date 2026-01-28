# Discord Frontend Plan (Phase 1 - Beta)

## Status
Phase 1 is implemented and the Discord frontend is now in **beta**.

## Goal
Add a Discord frontend in parallel to the IRC/irssi frontend with minimal feature parity. The initial scope is **agentic actor on @mention** only, while keeping the design extensible for future Slack integration and Discord threads/replies/inline images.

## Decisions (Approved)
- **Triggering**: Only on **@mentions** (DMs count as mentions).
- **Library**: **discord.py** (2.x) for long‑term maintenance stability.
- **Arc naming**: Use **human‑readable** identifiers (e.g., `discord:{guild_name}#{channel_name}`).
- **Channel scoping**: All channels by default (no allowlist required initially).
- **Replies**: Respond as **replies** to the original Discord message (thread behavior later).
- **Prompts/models**: Reuse the **IRC command prompt/model configuration verbatim**.

## Phase 1 Scope
- **Agentic actor trigger** on Discord @mention (including DMs).
- Store messages in `ChatHistory` with readable Discord arcs.
- Send responses back to the channel as replies to the original message.
- Reuse the IRC command prompt/model configuration verbatim (including modes).
- Feature parity for IRC commands (parsing, mode classification, overrides, help).
- Proactive interjection, debouncing, auto-chronicling, cost followups.

## Non‑Goals (Phase 1)
- No thread/reply context awareness beyond sending a reply.
- No inline images/attachments processing.

## Architecture Sketch
- Add `muaddib/rooms/discord/monitor.py` with a `DiscordRoomMonitor`.
- Update `MuaddibAgent.run()` to start IRC + Discord monitors concurrently.
- Minimal configuration under `rooms.discord`, but reuse IRC `command.modes.serious` for prompt/model.

## Future Considerations
- Support threads/replies as first‑class context in `ChatHistory`.
- Inline image handling by attaching metadata/URLs to context.
- Introduce a shared `RoomMessageEvent` abstraction to support Slack.

## TODO: Missing Features vs IRC
- Thread/reply context awareness (beyond sending a reply).
- Inline images/attachments processing.
- DM-specific handling differences (DMs are logged but no distinct behavior beyond reply).
