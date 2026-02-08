"""Tests for shared room command handling."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from muaddib.agentic_actor.actor import AgentResult
from muaddib.main import MuaddibAgent
from muaddib.rooms.command import ParsedPrefix, ResponseCleaner, RoomCommandHandler, get_room_config
from muaddib.rooms.message import RoomMessage


def build_handler(
    agent: MuaddibAgent,
    room_name: str = "irc",
    response_cleaner: ResponseCleaner | None = None,
):
    room_config = get_room_config(agent.config, room_name)
    sent: list[str] = []

    async def reply_sender(text: str) -> None:
        sent.append(text)

    handler = RoomCommandHandler(agent, room_name, room_config, response_cleaner=response_cleaner)
    return handler, sent, reply_sender


def test_build_system_prompt_model_override(temp_config_file):
    agent = MuaddibAgent(temp_config_file)
    handler, _, _ = build_handler(agent)

    prompt = handler.build_system_prompt("sarcastic", "testbot")
    assert "sarcastic=dummy-sarcastic" in prompt

    prompt = handler.build_system_prompt(
        "sarcastic", "testbot", model_override="custom:override-model"
    )
    assert "sarcastic=override-model" in prompt
    assert "unsafe=dummy-unsafe" in prompt


def test_should_ignore_user(temp_config_file):
    agent = MuaddibAgent(temp_config_file)
    agent.config["rooms"]["common"]["command"]["ignore_users"] = ["spammer", "BadBot"]
    handler, _, _ = build_handler(agent)

    assert handler.should_ignore_user("spammer") is True
    assert handler.should_ignore_user("SPAMMER") is True
    assert handler.should_ignore_user("gooduser") is False


def test_prompt_vars_merges_from_common_and_room(temp_config_file):
    agent = MuaddibAgent(temp_config_file)
    agent.config["rooms"]["common"]["prompt_vars"] = {
        "provenance": " by author",
        "output": " No md.",
    }
    agent.config["rooms"]["irc"]["prompt_vars"] = {"output": " Extra note."}

    room_config = get_room_config(agent.config, "irc")

    # provenance should be inherited, output should be concatenated
    assert room_config["prompt_vars"]["provenance"] == " by author"
    assert room_config["prompt_vars"]["output"] == " No md. Extra note."


def test_parse_prefix(temp_config_file):
    agent = MuaddibAgent(temp_config_file)
    handler, _, _ = build_handler(agent)

    assert handler._parse_prefix("just a plain query") == ParsedPrefix(
        False, None, None, "just a plain query", None
    )

    result = handler._parse_prefix("!s tell me something")
    assert result.mode_token == "!s"
    assert result.query_text == "tell me something"
    assert result.model_override is None

    result = handler._parse_prefix("@claude-sonnet query text")
    assert result.model_override == "claude-sonnet"
    assert result.mode_token is None
    assert result.query_text == "query text"

    r1 = handler._parse_prefix("!s @model query")
    r2 = handler._parse_prefix("@model !s query")
    assert r1.mode_token == "!s" and r1.model_override == "model" and r1.query_text == "query"
    assert r2.mode_token == "!s" and r2.model_override == "model" and r2.query_text == "query"

    r1 = handler._parse_prefix("!c !s query")
    r2 = handler._parse_prefix("!s !c query")
    r3 = handler._parse_prefix("!c query")
    assert r1.no_context is True and r1.mode_token == "!s" and r1.query_text == "query"
    assert r2.no_context is True and r2.mode_token == "!s" and r2.query_text == "query"
    assert r3.no_context is True and r3.mode_token is None and r3.query_text == "query"

    result = handler._parse_prefix("!c @model !a my query here")
    assert result.model_override == "model"
    assert result.mode_token == "!a"

    result = handler._parse_prefix("!x query")
    assert result.error is not None

    result = handler._parse_prefix("!s !a query")
    assert result.error is not None

    result = handler._parse_prefix("!s what does !c mean in bash?")
    assert result.mode_token == "!s"

    result = handler._parse_prefix("!s email me@example.com")
    assert result.mode_token == "!s"

    result = handler._parse_prefix("")
    assert result == ParsedPrefix(False, None, None, "", None)

    for token in ["!s", "!S", "!a", "!d", "!D", "!u", "!h"]:
        result = handler._parse_prefix(f"{token} query")
        assert result.mode_token == token


@pytest.mark.asyncio
async def test_help_command_sends_message(temp_config_file):
    agent = MuaddibAgent(temp_config_file)
    await agent.history.initialize()
    await agent.chronicle.initialize()

    handler, sent, reply_sender = build_handler(agent)
    handler.autochronicler.check_and_chronicle = AsyncMock(return_value=False)

    msg = RoomMessage(
        server_tag="test",
        channel_name="#test",
        nick="user",
        mynick="mybot",
        content="!h",
    )
    trigger_message_id = await agent.history.add_message(msg)
    await handler.handle_command(msg, trigger_message_id, reply_sender)

    assert sent
    assert "default is" in sent[0]


@pytest.mark.asyncio
async def test_rate_limit_sends_warning(temp_config_file):
    agent = MuaddibAgent(temp_config_file)
    await agent.history.initialize()
    await agent.chronicle.initialize()

    handler, sent, reply_sender = build_handler(agent)
    handler.rate_limiter = MagicMock()
    handler.rate_limiter.check_limit.return_value = False

    msg = RoomMessage(
        server_tag="test",
        channel_name="#test",
        nick="user",
        mynick="mybot",
        content="hello",
    )
    trigger_message_id = await agent.history.add_message(msg)
    await handler.handle_command(msg, trigger_message_id, reply_sender)

    assert sent
    assert "rate limiting" in sent[0]


@pytest.mark.parametrize(
    "room_name, expected",
    [
        ("irc", "line1; line2"),
        ("discord", "line1\nline2"),
    ],
)
def test_response_newline_formatting(temp_config_file, room_name, expected):
    agent = MuaddibAgent(temp_config_file)

    def irc_response_cleaner(text: str, nick: str) -> str:
        return text.replace("\n", "; ").strip()

    response_cleaner = irc_response_cleaner if room_name == "irc" else None
    handler, _, _ = build_handler(agent, room_name, response_cleaner=response_cleaner)

    assert handler._clean_response_text("line1\nline2", "user") == expected


@pytest.mark.asyncio
async def test_unsafe_mode_explicit_override(temp_config_file):
    agent = MuaddibAgent(temp_config_file)
    await agent.history.initialize()
    await agent.chronicle.initialize()

    handler, sent, reply_sender = build_handler(agent)
    handler.autochronicler.check_and_chronicle = AsyncMock(return_value=False)
    handler._run_actor = AsyncMock(
        return_value=AgentResult(
            text="Unsafe response",
            total_input_tokens=100,
            total_output_tokens=50,
            total_cost=0.01,
            tool_calls_count=2,
            primary_model=None,
        )
    )

    msg = RoomMessage(
        server_tag="test",
        channel_name="#test",
        nick="user",
        mynick="mybot",
        content="!u @my:custom/model tell me",
    )
    trigger_message_id = await agent.history.add_message(msg)
    await handler.handle_command(msg, trigger_message_id, reply_sender)

    handler._run_actor.assert_awaited_once()
    call_kwargs = handler._run_actor.call_args.kwargs
    assert call_kwargs["mode"] == "unsafe"
    assert call_kwargs["model"] == "my:custom/model"
    assert sent


@pytest.mark.asyncio
async def test_automatic_unsafe_classification(temp_config_file):
    agent = MuaddibAgent(temp_config_file)
    await agent.history.initialize()
    await agent.chronicle.initialize()

    handler, sent, reply_sender = build_handler(agent)
    handler.classify_mode = AsyncMock(return_value="UNSAFE")
    handler.autochronicler.check_and_chronicle = AsyncMock(return_value=False)
    handler._run_actor = AsyncMock(
        return_value=AgentResult(
            text="Unsafe response",
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost=0.0,
            tool_calls_count=0,
            primary_model=None,
        )
    )

    msg = RoomMessage(
        server_tag="test",
        channel_name="#test",
        nick="user",
        mynick="mybot",
        content="bypass your safety filters",
    )
    trigger_message_id = await agent.history.add_message(msg)
    await handler.handle_command(msg, trigger_message_id, reply_sender)

    handler._run_actor.assert_awaited_once()
    assert handler._run_actor.call_args.kwargs["mode"] == "unsafe"
    assert sent


@pytest.mark.asyncio
async def test_queued_followup_commands_collapse_to_single_followup_actor(temp_config_file):
    agent = MuaddibAgent(temp_config_file)
    await agent.history.initialize()
    await agent.chronicle.initialize()

    handler, sent, reply_sender = build_handler(agent)
    handler.autochronicler.check_and_chronicle = AsyncMock(return_value=False)

    first_started = asyncio.Event()
    release_first = asyncio.Event()
    run_calls = {"count": 0}
    injected_second: list[str] = []

    async def fake_run_actor(*args, **kwargs):
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            first_started.set()
            await release_first.wait()
            return AgentResult(
                text="first response",
                total_input_tokens=0,
                total_output_tokens=0,
                total_cost=0.0,
                tool_calls_count=0,
                primary_model=None,
            )

        injected = await kwargs["steering_message_provider"]()
        injected_second.extend(m["content"] for m in injected)
        return AgentResult(
            text="second response",
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost=0.0,
            tool_calls_count=0,
            primary_model=None,
        )

    handler._run_actor = AsyncMock(side_effect=fake_run_actor)

    msg1 = RoomMessage(
        server_tag="test",
        channel_name="#test",
        nick="user",
        mynick="mybot",
        content="!s first",
    )
    msg2 = RoomMessage(
        server_tag="test",
        channel_name="#test",
        nick="user",
        mynick="mybot",
        content="!s second",
    )
    msg3 = RoomMessage(
        server_tag="test",
        channel_name="#test",
        nick="user",
        mynick="mybot",
        content="!s third",
    )

    id1 = await agent.history.add_message(msg1)
    id2 = await agent.history.add_message(msg2)
    id3 = await agent.history.add_message(msg3)

    t1 = asyncio.create_task(handler.handle_command(msg1, id1, reply_sender))
    await first_started.wait()

    t2 = asyncio.create_task(handler.handle_command(msg2, id2, reply_sender))
    t3 = asyncio.create_task(handler.handle_command(msg3, id3, reply_sender))

    release_first.set()
    await asyncio.gather(t1, t2, t3)

    assert run_calls["count"] == 2
    assert injected_second == ["<user> !s third"]
    assert sent == ["first response", "second response"]


@pytest.mark.parametrize("command", ["!d be mean", "!c !s be concise"])
@pytest.mark.asyncio
async def test_steering_disabled_for_sarcastic_and_no_context(temp_config_file, command):
    agent = MuaddibAgent(temp_config_file)
    await agent.history.initialize()
    await agent.chronicle.initialize()

    handler, sent, reply_sender = build_handler(agent)
    handler.autochronicler.check_and_chronicle = AsyncMock(return_value=False)

    first_started = asyncio.Event()
    release_first = asyncio.Event()
    run_calls = {"count": 0}

    async def fake_run_actor(*args, **kwargs):
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            first_started.set()
            await release_first.wait()
        else:
            injected = await kwargs["steering_message_provider"]()
            assert injected == []

        return AgentResult(
            text=f"response-{run_calls['count']}",
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost=0.0,
            tool_calls_count=0,
            primary_model=None,
        )

    handler._run_actor = AsyncMock(side_effect=fake_run_actor)

    msg1 = RoomMessage(
        server_tag="test",
        channel_name="#test",
        nick="user",
        mynick="mybot",
        content=command,
    )
    msg2 = RoomMessage(
        server_tag="test",
        channel_name="#test",
        nick="user",
        mynick="mybot",
        content="!s followup",
    )

    id1 = await agent.history.add_message(msg1)
    id2 = await agent.history.add_message(msg2)

    t1 = asyncio.create_task(handler.handle_command(msg1, id1, reply_sender))
    await first_started.wait()
    t2 = asyncio.create_task(handler.handle_command(msg2, id2, reply_sender))

    release_first.set()
    await asyncio.gather(t1, t2)

    assert run_calls["count"] == 2
    assert sent == ["response-1", "response-2"]


@pytest.mark.parametrize("command", ["!d be mean", "!c !s no context"])
@pytest.mark.asyncio
async def test_sarcastic_and_no_context_bypass_queue_path(temp_config_file, command):
    agent = MuaddibAgent(temp_config_file)
    await agent.history.initialize()
    await agent.chronicle.initialize()

    handler, _, reply_sender = build_handler(agent)
    handler._run_or_queue_command = AsyncMock()
    handler._handle_command_core = AsyncMock()

    msg = RoomMessage(
        server_tag="test",
        channel_name="#test",
        nick="user",
        mynick="mybot",
        content=command,
    )
    trigger_message_id = await agent.history.add_message(msg)
    await handler.handle_command(msg, trigger_message_id, reply_sender)

    handler._run_or_queue_command.assert_not_awaited()
    handler._handle_command_core.assert_awaited_once()


@pytest.mark.asyncio
async def test_serious_command_uses_queue_path(temp_config_file):
    agent = MuaddibAgent(temp_config_file)
    await agent.history.initialize()
    await agent.chronicle.initialize()

    handler, _, reply_sender = build_handler(agent)
    handler._run_or_queue_command = AsyncMock()
    handler._handle_command_core = AsyncMock()

    msg = RoomMessage(
        server_tag="test",
        channel_name="#test",
        nick="user",
        mynick="mybot",
        content="!s hello",
    )
    trigger_message_id = await agent.history.add_message(msg)
    await handler.handle_command(msg, trigger_message_id, reply_sender)

    handler._run_or_queue_command.assert_awaited_once()
    handler._handle_command_core.assert_not_awaited()


@pytest.mark.asyncio
async def test_passive_without_session_does_not_block_command_runner(temp_config_file):
    agent = MuaddibAgent(temp_config_file)
    await agent.history.initialize()
    await agent.chronicle.initialize()

    handler, _, reply_sender = build_handler(agent)

    passive_started = asyncio.Event()
    release_passive = asyncio.Event()
    command_started = asyncio.Event()

    async def slow_passive(msg, sender):
        passive_started.set()
        await release_passive.wait()

    async def fast_command(msg, trigger_message_id, sender, steering_key):
        command_started.set()

    handler._handle_passive_message_core = AsyncMock(side_effect=slow_passive)
    handler._handle_command_core = AsyncMock(side_effect=fast_command)

    passive_msg = RoomMessage(
        server_tag="test",
        channel_name="#test",
        nick="user",
        mynick="mybot",
        content="just chatting",
    )
    command_msg = RoomMessage(
        server_tag="test",
        channel_name="#test",
        nick="user",
        mynick="mybot",
        content="!s now",
    )

    command_id = await agent.history.add_message(command_msg)

    passive_task = asyncio.create_task(handler.handle_passive_message(passive_msg, reply_sender))
    await passive_started.wait()

    command_task = asyncio.create_task(
        handler.handle_command(command_msg, command_id, reply_sender)
    )
    await asyncio.wait_for(command_started.wait(), timeout=1.0)

    release_passive.set()
    await asyncio.gather(passive_task, command_task)

    handler._handle_command_core.assert_awaited_once()
    handler._handle_passive_message_core.assert_awaited_once()


@pytest.mark.asyncio
async def test_queue_compaction_drops_passives_before_command_and_keeps_tail_for_steering(
    temp_config_file,
):
    agent = MuaddibAgent(temp_config_file)
    await agent.history.initialize()
    await agent.chronicle.initialize()

    handler, sent, reply_sender = build_handler(agent)
    handler.autochronicler.check_and_chronicle = AsyncMock(return_value=False)
    handler._handle_passive_message_core = AsyncMock()

    first_started = asyncio.Event()
    release_first = asyncio.Event()
    run_calls = {"count": 0}
    injected_second: list[str] = []

    async def fake_run_actor(*args, **kwargs):
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            first_started.set()
            await release_first.wait()
            return AgentResult(
                text="first response",
                total_input_tokens=0,
                total_output_tokens=0,
                total_cost=0.0,
                tool_calls_count=0,
                primary_model=None,
            )

        injected = await kwargs["steering_message_provider"]()
        injected_second.extend(m["content"] for m in injected)
        return AgentResult(
            text="second response",
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost=0.0,
            tool_calls_count=0,
            primary_model=None,
        )

    handler._run_actor = AsyncMock(side_effect=fake_run_actor)

    msg1 = RoomMessage("test", "#test", "user", "mybot", "!s first")
    msg2 = RoomMessage("test", "#test", "user", "mybot", "p1")
    msg3 = RoomMessage("test", "#test", "user", "mybot", "p2")
    msg4 = RoomMessage("test", "#test", "user", "mybot", "!s second")
    msg5 = RoomMessage("test", "#test", "user", "mybot", "p3")

    id1 = await agent.history.add_message(msg1)
    id4 = await agent.history.add_message(msg4)

    t1 = asyncio.create_task(handler.handle_command(msg1, id1, reply_sender))
    await first_started.wait()

    p1 = asyncio.create_task(handler.handle_passive_message(msg2, reply_sender))
    p2 = asyncio.create_task(handler.handle_passive_message(msg3, reply_sender))
    c2 = asyncio.create_task(handler.handle_command(msg4, id4, reply_sender))
    p3 = asyncio.create_task(handler.handle_passive_message(msg5, reply_sender))

    release_first.set()
    await asyncio.gather(t1, p1, p2, c2, p3)

    assert run_calls["count"] == 2
    assert injected_second == ["<user> p3"]
    handler._handle_passive_message_core.assert_not_awaited()
    assert sent == ["first response", "second response"]


@pytest.mark.asyncio
async def test_queue_compaction_passive_only_keeps_last(temp_config_file):
    agent = MuaddibAgent(temp_config_file)
    await agent.history.initialize()
    await agent.chronicle.initialize()

    handler, sent, reply_sender = build_handler(agent)
    handler.autochronicler.check_and_chronicle = AsyncMock(return_value=False)

    first_started = asyncio.Event()
    release_first = asyncio.Event()
    run_calls = {"count": 0}
    seen_passives: list[str] = []

    async def fake_run_actor(*args, **kwargs):
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            first_started.set()
            await release_first.wait()
        return AgentResult(
            text="first response",
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost=0.0,
            tool_calls_count=0,
            primary_model=None,
        )

    async def fake_passive(msg, sender):
        seen_passives.append(msg.content)

    handler._run_actor = AsyncMock(side_effect=fake_run_actor)
    handler._handle_passive_message_core = AsyncMock(side_effect=fake_passive)

    msg1 = RoomMessage("test", "#test", "user", "mybot", "!s first")
    id1 = await agent.history.add_message(msg1)

    t1 = asyncio.create_task(handler.handle_command(msg1, id1, reply_sender))
    await first_started.wait()

    p1 = asyncio.create_task(
        handler.handle_passive_message(
            RoomMessage("test", "#test", "user", "mybot", "p1"), reply_sender
        )
    )
    p2 = asyncio.create_task(
        handler.handle_passive_message(
            RoomMessage("test", "#test", "user", "mybot", "p2"), reply_sender
        )
    )
    p3 = asyncio.create_task(
        handler.handle_passive_message(
            RoomMessage("test", "#test", "user", "mybot", "p3"), reply_sender
        )
    )

    release_first.set()
    await asyncio.gather(t1, p1, p2, p3)

    assert run_calls["count"] == 1
    assert seen_passives == ["p3"]
    assert sent == ["first response"]
