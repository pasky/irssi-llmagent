# Slack Setup Guide

This guide walks you through creating a Slack app, enabling Socket Mode, and configuring Muaddib's Slack frontend.

## Prerequisites
- Slack workspace admin permissions
- A Slack app that can be installed to your workspace

## 1) Create a Slack App
1. Go to https://api.slack.com/apps
2. Click **Create New App** → **From scratch**.
3. Pick a name (e.g., `muaddib`) and select your workspace.

## 2) Enable Socket Mode
1. In your app settings, open **Socket Mode**.
2. Toggle **Enable Socket Mode**.
3. Create an **App Token** with the `connections:write` scope.
4. Copy the `xapp-...` token for configuration.

## 3) Add OAuth Scopes
Go to **OAuth & Permissions** and add these bot scopes:
- `app_mentions:read`
- `chat:write`
- `channels:history`, `groups:history`, `im:history`, `mpim:history`
- `channels:read`, `groups:read`, `im:read`, `mpim:read`
- `files:read`
- `users:read`
- `assistant:write` *(optional, enables "typing..." indicator while bot is responding)*

## 4) Enable Event Subscriptions
Even in Socket Mode, you must enable events:
1. Go to **Event Subscriptions** and toggle **Enable Events**.
2. Under **Subscribe to bot events**, add:
   - `app_mention`
   - `message.im`
   - `message.channels`
   - (Optional) `message.groups`, `message.mpim`

## 5) Allow DMs to the App
1. Go to **App Home**.
2. Enable **Messages Tab** and **Allow users to send messages to this app**.

## 6) Install the App
1. In **OAuth & Permissions**, click **Install to Workspace**.
2. Copy the **Bot User OAuth Token** (`xoxb-...`).

## 7) Configure Muaddib
Edit `~/.muaddib/config.json` (or `$MUADDIB_HOME/config.json`) and add/enable the Slack block under `rooms`:

```json
"slack": {
  "enabled": true,
  "app_token": "xapp-...",
  "workspaces": {
    "T123": {
      "name": "AmazingB2BSaaS",
      "bot_token": "xoxb-..."
    }
  },
  "reply_start_thread": {
    "channel": true,
    "dm": false
  },
  "command": {
    "history_size": 20,
    "response_max_chars": 1600,
    "debounce": 3
  }
}
```

Notes:
- `T123` is the Slack **Team ID** for your workspace. If it's not shown in Workspace Settings, you can grab it from the URL when Slack is open in a browser (`https://app.slack.com/client/T123/...`).
- Slack uses **two tokens**: `xapp-` for Socket Mode connection and `xoxb-` for Web API calls.
- The Slack frontend reuses IRC command prompt/model configuration verbatim.

## 8) Run Muaddib
```bash
uv run muaddib
```

## 9) Test
Mention the bot in a channel:
```
@YourBotName hello
```
The bot should reply in a thread (by default) or in channel based on configuration.

## Troubleshooting
- **No response**: confirm Socket Mode is enabled and the app is running.
- **Permission errors**: ensure the scopes above are granted and the app is installed.
- **Token errors**: verify `xapp-` and `xoxb-` tokens and restart the app.
