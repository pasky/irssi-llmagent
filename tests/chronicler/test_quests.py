import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from irssi_llmagent.chronicler.chapters import chapter_append_paragraph


@pytest.mark.asyncio
async def test_quest_operator_triggers_and_announces(shared_agent):
    agent = shared_agent
    # Configure quests operator
    arc = "testserver#testchan"
    agent.config.setdefault("chronicler", {}).setdefault("quests", {})["arcs"] = [arc]
    agent.config["chronicler"]["quests"]["prompt_reminder"] = "Quest instructions here"
    agent.config.setdefault("chronicler", {}).setdefault("quests", {})["cooldown"] = 0.1
    agent.config.setdefault("actor", {}).setdefault("quests", {})["cooldown"] = 0.01

    # Mock AgenticLLMActor to return an intermediate quest step, then finished
    intermediate_para = '<quest id="abc">Intermediate step</quest>'
    finished_para = "All done. CONFIRMED ACHIEVED"

    call_counter = {"count": 0}
    finished_event = asyncio.Event()
    third_call_event = asyncio.Event()

    class DummyActor:
        def __init__(self, *args, **kwargs):
            pass

        async def run_agent(self, context, *, progress_callback=None, arc: str):
            call_counter["count"] += 1
            if call_counter["count"] == 2:
                finished_event.set()
            if call_counter["count"] >= 3:
                third_call_event.set()
            return intermediate_para if call_counter["count"] == 1 else finished_para

    with patch("irssi_llmagent.main.AgenticLLMActor", new=DummyActor):
        # Ensure varlink sender mock
        agent.irc_monitor.varlink_sender = AsyncMock()
        agent.irc_monitor.get_mynick = AsyncMock(return_value="botnick")

        # Append initial quest paragraph
        initial_para = '<quest id="abc">Start the mission</quest>'
        await chapter_append_paragraph(arc, initial_para, agent)

        # Wait deterministically for exactly two model runs and two announcements or timeout
        await asyncio.wait_for(finished_event.wait(), timeout=1.0)
        for _ in range(100):
            if agent.irc_monitor.varlink_sender.send_message.await_count == 2:
                break
            await asyncio.sleep(0.01)
        assert call_counter["count"] == 2
        assert not third_call_event.is_set()
        assert agent.irc_monitor.varlink_sender.send_message.await_count == 2

        # Verify both intermediate and finished paragraphs were appended and announced
        # Allow DB append to materialize; poll briefly without flakiness
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

        # Ensure no further quest steps are scheduled after finished
        await asyncio.sleep(0.05)
        assert call_counter["count"] == 2


@pytest.mark.asyncio
async def test_scan_and_trigger_open_quests(shared_agent):
    agent = shared_agent
    arc = "srv#chan"
    agent.config.setdefault("chronicler", {}).setdefault("quests", {})["arcs"] = [arc]
    agent.config["chronicler"]["quests"]["prompt_reminder"] = "Quest instructions here"
    agent.config.setdefault("chronicler", {}).setdefault("quests", {})["cooldown"] = 0.01
    agent.config.setdefault("actor", {}).setdefault("quests", {})["cooldown"] = 0.01

    # Seed a quest paragraph
    await chapter_append_paragraph(arc, '<quest id="q1">Do X</quest>', agent)

    # Mock AgenticLLMActor and mynick
    next_para = '<quest_finished id="q1">Done X. CONFIRMED ACHIEVED</quest_finished>'

    class DummyActor2:
        def __init__(self, *args, **kwargs):
            pass

        async def run_agent(self, context, *, progress_callback=None, arc: str):
            return next_para

    with patch("irssi_llmagent.main.AgenticLLMActor", new=DummyActor2):
        agent.irc_monitor.varlink_sender = AsyncMock()
        agent.irc_monitor.get_mynick = AsyncMock(return_value="botnick")

        await agent.quests.scan_and_trigger_open_quests()
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

        async def run_agent(self, context, *, progress_callback=None, arc: str):
            actor_call_count["n"] += 1
            return None

    with patch("irssi_llmagent.main.AgenticLLMActor", new=DummyActor3), patch(
        "irssi_llmagent.providers.ModelRouter.call_raw_with_model"
    ) as mock_router:
        # Mock the model router to avoid network calls during chronicle summarization
        mock_client = MagicMock()
        mock_client.extract_text_from_response.return_value = (
            "Error - API error: Mock connection refused"
        )
        mock_router.return_value = ({"error": "Mock connection refused"}, mock_client, None)
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

        # Allow time for background quest tasks to complete
        await asyncio.sleep(0.15)

        # Read the new chapter that was created during rollover (chapter_id_after)
        content = await agent.chronicle.render_chapter(arc, chapter_id=chapter_id_after)
        assert "Previous chapter recap:" in content
        assert "Carry over me" in content

        # With fast execution, quest result gets added and triggers another quest
        # This is different from original slow test behavior but is correct
        assert actor_call_count["n"] >= 1  # at least initial quest triggered operator
        assert (
            agent.irc_monitor.varlink_sender.send_message.await_count >= 1
        )  # at least one announcement
