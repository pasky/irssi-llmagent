# Slack Frontend Design

## Status
Proposed (design phase).

## Decisions (Proposed)
- **Triggering**: Respond only on **@mentions** and **DMs** (DMs count as mentions). Non‑mentioned messages can still be scanned for proactive interjection when enabled.
- **Library**: **slack_bolt** async app + **slack_sdk** (Socket Mode) for a long‑running, server‑less process.
- **Arc naming**: Use **human‑readable** identifiers: `slack:{workspace_name}#{channel_name}`. DMs use `slack:_DM#{normalized_user}_{user_id}`.
- **Channel scoping**: All channels by default (no allowlist required initially).
- **Replies**: Respond in the **same thread** as the triggering message (use `thread_ts`). If the message is not in a thread, reply in channel without creating a new thread.
- **Thread isolation**: Thread content (not the thread starter) is isolated context for all purposes except the chronicle, matching Discord behavior.
- **Prompts/models**: Reuse the **IRC command prompt/model configuration verbatim**.

## Scope
- **Agentic actor trigger** on Slack app mentions and DMs.
- Store messages in `ChatHistory` with readable Slack arcs.
- Send responses back to the channel or thread as replies to the original Slack message.
- Reuse the IRC command prompt/model configuration verbatim (including modes and overrides).
- Feature parity for IRC commands (parsing, mode classification, overrides, help).
- Proactive interjection, debouncing, auto‑chronicling, and cost followups.
- Inline image/file attachment processing.

## Architecture Overview
- **Room monitor**: `SlackRoomMonitor` under `muaddib/rooms/slack/monitor.py`, mirroring Discord/IRC monitors.
- **Client**: `slack_bolt.async_app.AsyncApp` with `AsyncSocketModeHandler`.
- **Events**:
  - `app_mention` → direct command handling.
  - `message` events in DMs (`channel_type == "im"`) → direct command handling.
  - `message` events in channels without mention → passive processing (proactive interjection path).
- **Mention stripping**: Remove the leading `<@BOTID>` token (and optional nick aliases) before command parsing.
- **Message identity**:
  - `platform_id`: Slack `ts`.
  - `thread_id`: Slack `thread_ts` when present.
  - `thread_starter_id`: resolved via `ChatHistory.get_message_id_by_platform_id` for `thread_ts`.
- **Reply sending**: Use `chat.postMessage` with `thread_ts` (when applicable). Support **reply edit debouncing** by tracking the last response and using `chat.update` when within `reply_edit_debounce_seconds`.

## Attachments & Message Normalization
- **Text**: Normalize Slack markup (`<@U123>` → `@name`, `<#C123|channel>` → `#channel`, `<http://...|label>` → `http://...`).
- **Files**: Append a `[Attachments]` block including filename, mime type, size, and `url_private`. If feasible, also include image URLs for vision model use.
- **Bot/self messages**: Ignore events from the bot user or `subtype == "bot_message"`.

## Configuration
Add a new `rooms.slack` block to `config.json.example` (mirrors `discord`/`irc`):

```json
"slack": {
  "enabled": false,
  "bot_token": "xoxb-...",
  "app_token": "xapp-...",
  "reply_edit_debounce_seconds": 15.0,
  "prompt_note": " You are now connected to Slack, and can use light Slack markdown.",
  "command": {
    "history_size": 20,
    "response_max_chars": 1600,
    "debounce": 3,
    "ignore_users": []
  },
  "proactive": {
    "debounce_seconds": 40.0,
    "interjecting": ["workspace#general"],
    "interjecting_test": []
  }
}
```

## Permissions (Slack App)
Required OAuth scopes (minimum):
- `app_mentions:read`
- `chat:write`
- `channels:history`, `groups:history`, `im:history`, `mpim:history`
- `channels:read`, `groups:read`, `im:read`, `mpim:read`
- `files:read` (for attachments)
- `users:read` (to resolve display names)
- `connections:write` (Socket Mode)

## Future Considerations
- Introduce a shared `RoomMessageEvent` abstraction across IRC/Discord/Slack.
- Optional allowlist for Slack channels at the room config level.
- Per‑workspace overrides (if multiple Slack workspaces are supported later).
