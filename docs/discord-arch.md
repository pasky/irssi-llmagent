# Discord Frontend Plan (Phase 1)

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
- Reuse the IRC serious agentic mode prompt/model; no extra modes.

## Non‑Goals (Phase 1)
- No proactive interjecting, mode classification, or Perplexity.
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
- Command parsing (`!s`, `!a`, `!d`, `!u`, `!h`, `!p`, `!c`, model overrides).
- Automatic mode classification (sarcastic vs serious vs unsafe).
- Channel-level mode overrides and per-channel defaults.
- Proactive interjection (debouncer + validation models).
- Command debouncing/aggregation for rapid follow-ups.
- Perplexity integration (`!p` command).
- Unsafe mode support and explicit unsafe routing.
- Progress messages (tool progress callbacks are not surfaced to Discord).
- Tool persistence summaries stored separately (no persistence callback wiring).
- Auto-chronicling integration (no `AutoChronicler` usage).
- Cost follow-up messages for expensive requests.
- Daily cost milestone announcements.
- Ignore list support for users.
- Context reduction toggles (no `!c` no-context option).
- Model override per request (e.g., `@provider:model`).
- IRC-style help message for available modes and models.
- Artifact creation on long responses is IRC-only (no Discord-specific artifact behavior).
- DM-specific handling differences (DMs are logged but no distinct behavior beyond reply).
