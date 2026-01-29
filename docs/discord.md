# Discord Setup Guide

This guide walks you through creating a Discord bot, obtaining a token, and enabling the Muaddib Discord frontend (currently **beta**).

## Prerequisites
- You are a Discord server owner (or have admin permissions).
- You do not have a bot app yet.

## 1) Create a Discord Application
1. Go to the Discord Developer Portal: https://discord.com/developers/applications
2. Click **New Application**.
3. Enter a name (e.g., `muaddib`) and create it.

## 2) Create a Bot User
1. In your application, open **Bot** in the left sidebar.
2. Click **Add Bot** and confirm.
3. (Optional) Set a bot username and avatar.

## 3) Enable Required Privileged Intents
Muaddib needs message content to detect @mentions.
1. In **Bot** settings, locate **Privileged Gateway Intents**.
2. Enable **Message Content Intent**.

## 4) Copy the Bot Token
1. In **Bot** settings, click **Reset Token** (or **View Token** if already generated).
2. Copy the token and keep it secret.

## 5) Invite the Bot to Your Server
1. Go to **OAuth2** → **URL Generator**.
2. Under **Scopes**, check:
   - **bot**
3. Under **Bot Permissions**, check:
   - **View Channels**
   - **Send Messages**
   - **Send Messages in Threads**
   - **Attach Files**
   - **Read Message History**
   - (Optional) **Embed Links** if you want future rich replies
4. Copy the generated URL and open it in your browser.
5. Select your server and authorize the bot.

## 6) Configure Muaddib
Edit `~/.muaddib/config.json` (or `$MUADDIB_HOME/config.json`) and add/enable the Discord block under `rooms`:

```json
"discord": {
  "enabled": true,
  "token": "YOUR_DISCORD_BOT_TOKEN",
  "command": {
    "history_size": 40,
    "rate_limit": 30,
    "rate_period": 900
  }
}
```

Notes:
- The Discord frontend reuses the IRC serious-mode prompt/model configuration.
- All channels are enabled by default. The bot will only respond to @mentions or DMs.

## 7) Run Muaddib
```bash
uv run muaddib
```

## 8) Test
Mention the bot in a channel:
```
@YourBotName hello
```
The bot should reply to your message.

## Troubleshooting
- **No response**: confirm the bot is online and Message Content Intent is enabled.
- **Permission errors**: ensure the bot has the listed permissions in the channel.
- **Token errors**: verify you pasted the correct token and restarted the app.
