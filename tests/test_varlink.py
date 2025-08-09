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


# UTF-8 boundary tests merged from separate file per contributor preference
@pytest.mark.asyncio
async def test_split_preserves_utf8_characters_and_words():
    sender = DummySender()
    target = "#t"

    # Russian text with multibyte chars; ensure split falls near multibyte boundary
    text = (
        "Коротко: у LessWrong он фигурирует как ранний техно‑скептик, но его анализ считают "
        "упрощённым/ошибочным, а насилие — однозначно аморальным и контрпродуктивным; "
        "манифест обсуждают как исторический кейс, фокус смещают на управление рисками и безопасность технологий (в т.ч. AI), а не на неолуддизм."
    )

    # Force a split by appending filler to exceed limit
    filler = " X" * 200
    msg = text + filler

    ok = await sender.send_message(target, msg, server="irc")
    assert ok
    assert len(sender.sent) in (1, 2)

    combined = "".join(sender.sent)
    # If truncated to two messages, the combined should be a prefix of the original
    assert msg.startswith(combined)

    # Ensure no character loss around split
    assert "безопасность технологий" in combined or "безопасност" in combined


@pytest.mark.asyncio
async def test_no_character_loss_at_boundary():
    sender = DummySender()
    target = "#t"
    max_payload = max(1, 512 - 12 - len(target.encode("utf-8")))

    # Build a string whose byte length is just over the limit, with a multibyte char at the cut
    base = "a" * (max_payload - 1)
    # Cyrillic soft sign is two bytes in UTF-8; ensure cut would have fallen between bytes if naive
    s = base + "ь" + " продолжение"

    ok = await sender.send_message(target, s, server="irc")
    assert ok
    assert len(sender.sent) == 2
    combined = "".join(sender.sent)
    # Should exactly equal original or be a prefix if second part truncated (not here)
    assert combined.startswith(s)
    assert "ь п" in combined  # The soft sign followed by space should be preserved at split
