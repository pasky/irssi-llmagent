import pytest

from irssi_llmagent.varlink import VarlinkSender


class DummySender(VarlinkSender):
    def __init__(self):
        super().__init__("/tmp/does-not-matter.sock")
        self.sent = []

    async def send_call(self, method: str, parameters: dict | None = None):
        # Capture messages instead of real varlink
        if method == "org.irssi.varlink.SendMessage":
            self.sent.append(parameters["message"])  # type: ignore[index]
        return {"parameters": {"success": True}}


@pytest.mark.asyncio
async def test_send_short_message_no_split():
    sender = DummySender()
    target = "#test"
    msg = "hello world"
    ok = await sender.send_message(target, msg, server="irc")
    assert ok
    assert sender.sent == [msg]


@pytest.mark.asyncio
async def test_split_into_two_parts_when_over_limit():
    sender = DummySender()
    target = "#t"
    # Compute payload limit as used in implementation
    max_payload = max(1, 512 - 12 - len(target.encode("utf-8")))
    msg = "a" * (max_payload + 10)
    ok = await sender.send_message(target, msg, server="irc")
    assert ok
    assert len(sender.sent) == 2
    assert len(sender.sent[0].encode("utf-8")) <= max_payload
    assert len(sender.sent[1].encode("utf-8")) <= max_payload
    # Combined content should start with original and be a prefix (second possibly truncated)
    combined = (sender.sent[0] + sender.sent[1]).replace(" ", "")
    assert msg.startswith(combined) or combined.startswith(msg)


@pytest.mark.asyncio
async def test_second_part_truncated_not_more_than_two_messages():
    sender = DummySender()
    target = "#longtarget"
    max_payload = max(1, 512 - 12 - len(target.encode("utf-8")))
    # Create a message that would require 3 parts if not truncated
    msg = "b" * (max_payload * 3)
    ok = await sender.send_message(target, msg, server="irc")
    assert ok
    assert len(sender.sent) == 2
    assert len(sender.sent[0].encode("utf-8")) <= max_payload
    assert len(sender.sent[1].encode("utf-8")) <= max_payload
