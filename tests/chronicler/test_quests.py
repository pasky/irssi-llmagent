import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from muaddib.agentic_actor.actor import AgentResult
from muaddib.chronicler.chapters import chapter_append_paragraph
from muaddib.chronicler.tools import (
    QuestSnoozeExecutor,
    QuestStartExecutor,
    SubquestStartExecutor,
    _validate_quest_id,
)
from muaddib.providers import ModelSpec, UsageInfo


@pytest.mark.asyncio
async def test_quest_operator_triggers_and_announces(shared_agent):
    """Test that quests are triggered via heartbeat and run to completion."""
    agent = shared_agent
    # Configure quests operator
    arc = "testserver#testchan"
    agent.config.setdefault("chronicler", {}).setdefault("quests", {})["arcs"] = [arc]
    agent.config["chronicler"]["quests"]["prompt_reminder"] = "Quest instructions here"
    agent.config.setdefault("chronicler", {}).setdefault("quests", {})["cooldown"] = 0.01
    agent.config.setdefault("actor", {}).setdefault("quests", {})["cooldown"] = 0.01

    # Mock AgenticLLMActor to return an intermediate quest step, then finished
    # Include extra text around quest tags to verify extraction works
    intermediate_para = (
        'Here is my response: <quest id="abc">Intermediate step</quest> Hope that helps!'
    )
    finished_para = "Let me wrap up. <quest>All done. CONFIRMED ACHIEVED</quest> That concludes it."

    call_counter = {"count": 0}
    finished_event = asyncio.Event()
    third_call_event = asyncio.Event()

    class DummyActor:
        def __init__(self, *args, **kwargs):
            pass

        async def run_agent(
            self,
            context,
            *,
            progress_callback=None,
            persistence_callback=None,
            arc: str,
            current_quest_id: str | None = None,
        ):
            call_counter["count"] += 1
            if call_counter["count"] == 2:
                finished_event.set()
            if call_counter["count"] >= 3:
                third_call_event.set()
            text = intermediate_para if call_counter["count"] == 1 else finished_para
            return AgentResult(
                text=text,
                total_input_tokens=100,
                total_output_tokens=50,
                total_cost=0.01,
                tool_calls_count=2,
            )

    with patch("muaddib.main.AgenticLLMActor", new=DummyActor):
        # Ensure varlink sender mock
        agent.irc_monitor.varlink_sender = AsyncMock()
        agent.irc_monitor.get_mynick = AsyncMock(return_value="botnick")

        # Append initial quest paragraph (creates quest in DB, heartbeat will trigger)
        initial_para = '<quest id="abc">Start the mission</quest>'
        await chapter_append_paragraph(arc, initial_para, agent)

        # Keep triggering heartbeat until quest finishes (or timeout)
        async def heartbeat_until_done():
            while not finished_event.is_set():
                await agent.quests._heartbeat_tick()
                await asyncio.sleep(0.02)

        await asyncio.wait_for(heartbeat_until_done(), timeout=2.0)
        for _ in range(100):
            if agent.irc_monitor.varlink_sender.send_message.await_count == 2:
                break
            await asyncio.sleep(0.01)
        assert call_counter["count"] == 2
        assert not third_call_event.is_set()
        assert agent.irc_monitor.varlink_sender.send_message.await_count == 2

        # Verify both intermediate and finished paragraphs were appended and announced
        content = ""
        for _ in range(50):
            content = await agent.chronicle.render_chapter(arc)
            if "CONFIRMED ACHIEVED" in content:
                break
            await asyncio.sleep(0.01)
        assert "Start the mission" in content
        assert "Intermediate step" in content
        assert "CONFIRMED ACHIEVED" in content

        calls = agent.irc_monitor.varlink_sender.send_message.await_args_list
        assert any("Intermediate step" in c[0][1] for c in calls)
        assert any("CONFIRMED ACHIEVED" in c[0][1] for c in calls)

        # Verify full response (including extra text) was sent to IRC
        assert any("Here is my response" in c[0][1] for c in calls)
        assert any("That concludes it" in c[0][1] for c in calls)

        # But chronicle should only have quest XML, not the extra text
        assert "Here is my response" not in content
        assert "Hope that helps" not in content
        assert "Let me wrap up" not in content
        assert "That concludes it" not in content

        # Ensure no further quest steps are scheduled after finished
        await asyncio.sleep(0.05)
        assert call_counter["count"] == 2


@pytest.mark.asyncio
async def test_heartbeat_triggers_open_quests(shared_agent):
    """Test that heartbeat tick finds and prods ongoing quests."""
    from muaddib.chronicler import QuestStatus

    agent = shared_agent
    arc = "srv#chan"
    agent.config.setdefault("chronicler", {}).setdefault("quests", {})["arcs"] = [arc]
    agent.config["chronicler"]["quests"]["prompt_reminder"] = "Quest instructions here"
    agent.config.setdefault("chronicler", {}).setdefault("quests", {})["cooldown"] = 0.01
    agent.config.setdefault("actor", {}).setdefault("quests", {})["cooldown"] = 0.01

    # Mock AgenticLLMActor and mynick before seeding to prevent uncontrolled runs
    next_para = '<quest_finished id="q1">Done X. CONFIRMED ACHIEVED</quest_finished>'

    class DummyActor2:
        def __init__(self, *args, **kwargs):
            pass

        async def run_agent(
            self,
            context,
            *,
            progress_callback=None,
            persistence_callback=None,
            arc: str,
            current_quest_id: str | None = None,
        ):
            return AgentResult(
                text=next_para,
                total_input_tokens=100,
                total_output_tokens=50,
                total_cost=0.01,
                tool_calls_count=2,
            )

    with patch("muaddib.main.AgenticLLMActor", new=DummyActor2):
        agent.irc_monitor.varlink_sender = AsyncMock()
        agent.irc_monitor.get_mynick = AsyncMock(return_value="botnick")

        # Seed a quest paragraph (this creates the quest in DB via on_chronicle_append)
        await chapter_append_paragraph(arc, '<quest id="q1">Do X</quest>', agent)

        # Wait for initial trigger to complete
        await asyncio.sleep(0.05)

        # Reset to ONGOING so heartbeat can find it (initial run may have finished it)
        quest = await agent.chronicle.quest_get("q1")
        if quest and quest["status"] != QuestStatus.FINISHED.value:
            await agent.chronicle.quest_set_status("q1", QuestStatus.ONGOING)

            # Heartbeat tick should find the ongoing quest and prod it
            await agent.quests._heartbeat_tick()
            await asyncio.sleep(0.05)

        content = await agent.chronicle.render_chapter(arc)
        for _ in range(20):
            if "Done X" in content:
                break
            await asyncio.sleep(0.05)
            content = await agent.chronicle.render_chapter(arc)

        assert "Do X" in content
        assert "Done X" in content


@pytest.mark.asyncio
async def test_chapter_rollover_copies_unresolved_quests(shared_agent):
    agent = shared_agent
    arc = "serv#room"
    # Configure low paragraphs_per_chapter to trigger rollover deterministically
    agent.config.setdefault("chronicler", {})["paragraphs_per_chapter"] = 3
    agent.config.setdefault("chronicler", {}).setdefault("quests", {})["arcs"] = [arc]
    agent.config["chronicler"]["quests"]["prompt_reminder"] = "Quest instructions here"
    agent.config.setdefault("chronicler", {}).setdefault("quests", {})["cooldown"] = 0.01

    # Prevent operator sending and running during test; observe calls
    agent.irc_monitor.varlink_sender = AsyncMock()
    agent.irc_monitor.get_mynick = AsyncMock(return_value="botnick")
    actor_call_count = {"n": 0}

    class DummyActor3:
        def __init__(self, *args, **kwargs):
            pass

        async def run_agent(
            self,
            context,
            *,
            progress_callback=None,
            persistence_callback=None,
            arc: str,
            current_quest_id: str | None = None,
        ):
            actor_call_count["n"] += 1
            return AgentResult(
                text="",
                total_input_tokens=100,
                total_output_tokens=50,
                total_cost=0.01,
                tool_calls_count=2,
            )

    with (
        patch("muaddib.main.AgenticLLMActor", new=DummyActor3),
        patch("muaddib.providers.ModelRouter.call_raw_with_model") as mock_router,
    ):
        # Mock the model router to avoid network calls during chronicle summarization
        mock_client = MagicMock()
        mock_client.extract_text_from_response.return_value = (
            "Error: API error: Mock connection refused"
        )
        mock_router.return_value = (
            {"error": "Mock connection refused"},
            mock_client,
            ModelSpec("test", "model"),
            UsageInfo(None, None, None),
        )
        # Fill chapter to exactly the limit with a quest and normal paragraphs
        await chapter_append_paragraph(arc, '<quest id="carry">Carry over me</quest>', agent)
        await chapter_append_paragraph(arc, "Some other text", agent)
        await chapter_append_paragraph(arc, "Another paragraph", agent)

        # At this point we have 3 paragraphs (at the limit). Now check chapter state before rollover
        current_chapter_before = await agent.chronicle.get_or_open_current_chapter(arc)
        chapter_id_before = current_chapter_before["id"]

        # This append should trigger rollover (4th paragraph exceeds limit of 3)
        await chapter_append_paragraph(arc, "Trigger rollover now", agent)

        # Verify rollover happened by checking that current chapter changed
        current_chapter_after = await agent.chronicle.get_or_open_current_chapter(arc)
        chapter_id_after = current_chapter_after["id"]
        assert chapter_id_after != chapter_id_before, "Rollover should have created a new chapter"

        # Read the new chapter that was created during rollover (chapter_id_after)
        content = await agent.chronicle.render_chapter(arc, chapter_id=chapter_id_after)
        assert "Previous chapter recap:" in content
        assert "Carry over me" in content

        # Trigger heartbeat to run the quest
        await agent.quests._heartbeat_tick()
        await asyncio.sleep(0.05)

        # Quest should have been triggered via heartbeat
        assert actor_call_count["n"] >= 1  # at least initial quest triggered operator
        assert (
            agent.irc_monitor.varlink_sender.send_message.await_count >= 1
        )  # at least one announcement


def test_validate_quest_id():
    """Quest ID validation rejects invalid formats."""
    assert _validate_quest_id("valid-id") is None
    assert _validate_quest_id("valid_id_123") is None

    err = _validate_quest_id("")
    assert err is not None and "empty" in err

    err = _validate_quest_id("a" * 100)
    assert err is not None and "too long" in err

    err = _validate_quest_id("bad.id")
    assert err is not None and "dots" in err

    assert _validate_quest_id("bad id") is not None  # spaces
    assert _validate_quest_id("bad@id") is not None  # special chars


@pytest.mark.asyncio
async def test_quest_start_executor(shared_agent):
    """QuestStartExecutor creates quest and appends paragraph."""
    agent = shared_agent
    arc = "srv#chan"
    agent.config.setdefault("chronicler", {}).setdefault("quests", {})["arcs"] = [arc]

    executor = QuestStartExecutor(agent=agent, arc=arc)
    result = await executor.execute(
        id="test-quest", goal="Do something", success_criteria="Something done"
    )

    assert result == "Quest started: test-quest"
    quest = await agent.chronicle.quest_get("test-quest")
    assert quest is not None
    assert quest["id"] == "test-quest"


@pytest.mark.asyncio
async def test_subquest_start_executor(shared_agent):
    """SubquestStartExecutor creates subquest with prefixed ID."""
    agent = shared_agent
    arc = "srv#chan"
    agent.config.setdefault("chronicler", {}).setdefault("quests", {})["arcs"] = [arc]

    # First create parent quest
    await chapter_append_paragraph(arc, '<quest id="parent">Main goal</quest>', agent)

    executor = SubquestStartExecutor(agent=agent, arc=arc, parent_quest_id="parent")
    result = await executor.execute(id="child", goal="Sub task", success_criteria="Sub task done")

    assert result == "Subquest started: parent.child"
    quest = await agent.chronicle.quest_get("parent.child")
    assert quest is not None
    assert quest["parent_id"] == "parent"


@pytest.mark.asyncio
async def test_subquest_finish_resumes_parent(shared_agent):
    """When a sub-quest finishes, parent resumes on next heartbeat."""
    agent = shared_agent
    arc = "srv#chan"
    agent.config.setdefault("chronicler", {}).setdefault("quests", {})["arcs"] = [arc]
    agent.config["chronicler"]["quests"]["prompt_reminder"] = "Quest instructions"
    agent.config["chronicler"]["quests"]["cooldown"] = 0.01

    # Track which quest IDs trigger the actor
    triggered_quest_ids = []

    class TrackingActor:
        def __init__(self, *args, **kwargs):
            pass

        async def run_agent(
            self,
            context,
            *,
            progress_callback=None,
            persistence_callback=None,
            arc: str,
            current_quest_id: str | None = None,
        ):
            triggered_quest_ids.append(current_quest_id)
            # Return finished for sub-quest, continuation for parent
            if current_quest_id == "parent.child":
                text = '<quest_finished id="parent.child">Sub-task done. CONFIRMED ACHIEVED</quest_finished>'
            else:
                text = '<quest id="parent">Continuing parent</quest>'
            return AgentResult(
                text=text,
                total_input_tokens=100,
                total_output_tokens=50,
                total_cost=0.01,
                tool_calls_count=2,
            )

    with patch("muaddib.main.AgenticLLMActor", new=TrackingActor):
        agent.irc_monitor.varlink_sender = AsyncMock()
        agent.irc_monitor.get_mynick = AsyncMock(return_value="botnick")

        # Start with parent quest
        await chapter_append_paragraph(arc, '<quest id="parent">Main goal</quest>', agent)

        # Trigger parent via heartbeat
        await agent.quests._heartbeat_tick()
        await asyncio.sleep(0.05)
        assert "parent" in triggered_quest_ids

        # Create sub-quest
        await chapter_append_paragraph(arc, '<quest id="parent.child">Sub-task</quest>', agent)

        # Trigger child via heartbeat - child will finish and update DB
        await agent.quests._heartbeat_tick()
        await asyncio.sleep(0.05)
        assert "parent.child" in triggered_quest_ids

        # Parent should now be ONGOING again (child finished), heartbeat resumes it
        await agent.quests._heartbeat_tick()

        # Wait briefly for async heartbeat task to complete
        for _ in range(50):
            parent_triggers = [q for q in triggered_quest_ids if q == "parent"]
            if len(parent_triggers) >= 2:
                break
            await asyncio.sleep(0.02)

        # Verify parent was triggered again after child finished
        parent_triggers = [q for q in triggered_quest_ids if q == "parent"]
        assert len(parent_triggers) >= 2, (
            f"Parent should be triggered at least twice (initial + resume): {triggered_quest_ids}"
        )


@pytest.mark.asyncio
async def test_quest_start_with_make_plan_and_final_answer(shared_agent):
    """make_plan + quest_start + final_answer in same turn all execute."""
    from muaddib.agentic_actor import AgenticLLMActor

    agent = shared_agent
    arc = "srv#chan"
    agent.config.setdefault("chronicler", {}).setdefault("quests", {})["arcs"] = [arc]

    executed_tools = []

    class MockClient:
        def cleanup_raw_text(self, text):
            return text

        def has_tool_calls(self, response):
            return "tool_calls" in response

        def extract_tool_calls(self, response):
            return response.get("tool_calls", [])

        def format_assistant_message(self, response):
            return {"role": "assistant", "content": ""}

        def format_tool_results(self, results):
            return [{"role": "user", "content": r["content"]} for r in results]

    async def mock_call_raw(model, messages, *args, **kwargs):
        return (
            {
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "make_plan",
                        "input": {"plan": "Step 1: Do thing\nStep 2: Do other thing"},
                    },
                    {
                        "id": "2",
                        "name": "quest_start",
                        "input": {
                            "id": "plan-test",
                            "goal": "Test goal",
                            "success_criteria": "Test criteria",
                        },
                    },
                    {
                        "id": "3",
                        "name": "final_answer",
                        "input": {"answer": "Quest started: plan-test"},
                    },
                ]
            },
            MockClient(),
            ModelSpec("test", "model"),
            UsageInfo(None, None, None),
        )

    actor = AgenticLLMActor(
        config=agent.config,
        model="anthropic:claude-3-5-sonnet",
        system_prompt_generator=lambda: "Test prompt",
        agent=agent,
    )

    # Wrap executors to track calls
    original_create = __import__(
        "muaddib.agentic_actor.tools", fromlist=["create_tool_executors"]
    ).create_tool_executors

    def tracking_create(*args, **kwargs):
        executors = original_create(*args, **kwargs)
        original_make_plan = executors["make_plan"].execute
        original_quest_start = executors["quest_start"].execute

        async def track_make_plan(**kw):
            executed_tools.append("make_plan")
            return await original_make_plan(**kw)

        async def track_quest_start(**kw):
            executed_tools.append("quest_start")
            return await original_quest_start(**kw)

        executors["make_plan"].execute = track_make_plan
        executors["quest_start"].execute = track_quest_start
        return executors

    with (
        patch(
            "muaddib.agentic_actor.actor.ModelRouter.call_raw_with_model",
            new=AsyncMock(side_effect=mock_call_raw),
        ),
        patch(
            "muaddib.agentic_actor.actor.create_tool_executors",
            side_effect=tracking_create,
        ),
    ):
        result = await actor.run_agent([{"role": "user", "content": "test"}], arc=arc)

    assert "make_plan" in executed_tools, "make_plan should execute"
    assert "quest_start" in executed_tools, "quest_start should execute"
    assert "Quest started: plan-test" in result.text


@pytest.mark.asyncio
async def test_quest_snooze_executor(shared_agent):
    """QuestSnoozeExecutor sets resume_at on the quest."""
    agent = shared_agent
    arc = "srv#chan"
    agent.config.setdefault("chronicler", {}).setdefault("quests", {})["arcs"] = [arc]

    await chapter_append_paragraph(arc, '<quest id="snooze-test">Goal</quest>', agent)

    executor = QuestSnoozeExecutor(agent=agent, quest_id="snooze-test")
    result = await executor.execute(until="14:30")

    assert "Quest snoozed until" in result

    quest = await agent.chronicle.quest_get("snooze-test")
    assert quest is not None
    assert quest["resume_at"] is not None
    resume_utc = datetime.strptime(quest["resume_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    resume_local = resume_utc.astimezone()
    assert resume_local.hour == 14
    assert resume_local.minute == 30


@pytest.mark.asyncio
async def test_quest_snooze_invalid_time(shared_agent):
    """QuestSnoozeExecutor rejects invalid time formats."""
    agent = shared_agent
    arc = "srv#chan"
    agent.config.setdefault("chronicler", {}).setdefault("quests", {})["arcs"] = [arc]

    await chapter_append_paragraph(arc, '<quest id="snooze-invalid">Goal</quest>', agent)

    executor = QuestSnoozeExecutor(agent=agent, quest_id="snooze-invalid")

    result = await executor.execute(until="invalid")
    assert "Error: Invalid time format" in result

    result = await executor.execute(until="25:00")
    assert "Error: Invalid time" in result

    result = await executor.execute(until="12:60")
    assert "Error: Invalid time" in result


@pytest.mark.asyncio
async def test_quest_snooze_nonexistent_quest(shared_agent):
    """QuestSnoozeExecutor returns error for nonexistent quest."""
    agent = shared_agent

    executor = QuestSnoozeExecutor(agent=agent, quest_id="nonexistent")
    result = await executor.execute(until="14:30")

    assert "Error: Quest 'nonexistent' not found" in result


@pytest.mark.asyncio
async def test_snoozed_quest_not_ready_for_heartbeat(shared_agent):
    """Snoozed quests are not returned by quests_ready_for_heartbeat."""
    from datetime import timedelta

    agent = shared_agent
    arc = "srv#chan"
    agent.config.setdefault("chronicler", {}).setdefault("quests", {})["arcs"] = [arc]

    await chapter_append_paragraph(arc, '<quest id="snoozed-heartbeat">Goal</quest>', agent)

    future_time = (datetime.now(UTC) + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    await agent.chronicle.quest_set_resume_at("snoozed-heartbeat", future_time)

    ready = await agent.chronicle.quests_ready_for_heartbeat(arc, 0)
    quest_ids = [q["id"] for q in ready]
    assert "snoozed-heartbeat" not in quest_ids
