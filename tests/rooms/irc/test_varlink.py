import pytest

from muaddib.rooms.irc import VarlinkSender


class DummySender(VarlinkSender):
    def __init__(self):
        super().__init__("/tmp/does-not-matter.sock")
        self.sent: list[str] = []

    async def send_call(self, method: str, parameters: dict | None = None):
        # Capture messages instead of real varlink
        if method == "org.irssi.varlink.SendMessage":
            self.sent.append(parameters["message"])  # type: ignore[index]
        return {"parameters": {"success": True}}


class DummySenderTruncating(VarlinkSender):
    """Simulates an IRC server enforcing the 512-byte limit including prefix.

    It truncates the message payload if the constructed line would exceed 512 bytes.
    Records whether truncation occurred.
    """

    def __init__(self, prefix_len: int = 60):
        super().__init__("/tmp/does-not-matter.sock")
        self.sent: list[str] = []
        self.truncated: list[bool] = []
        self.prefix_len = prefix_len  # e.g., ":nick!user@host " length + tags if any

    async def send_call(self, method: str, parameters: dict | None = None):
        if method == "org.irssi.varlink.SendMessage":
            target = parameters["target"]  # type: ignore[index]
            msg = parameters["message"]  # type: ignore[index]
            # Build an approximate IRC line: ":prefix PRIVMSG <target> :<msg>\r\n"
            # Compute bytes and truncate from the end of msg to fit 512 bytes total
            fixed = f"PRIVMSG {target} :".encode()
            msg_b = msg.encode("utf-8")
            total = self.prefix_len + len(fixed) + len(msg_b) + 2  # +2 for CRLF
            did_truncate = False
            if total > 512:
                did_truncate = True
                # How many bytes must be removed from msg
                over = total - 512
                keep = max(0, len(msg_b) - over)
                # Cut at UTF-8 boundary
                cut = keep
                while cut > 0 and (msg_b[cut - 1] & 0xC0) == 0x80:
                    cut -= 1
                msg_b = msg_b[:cut]
                msg = msg_b.decode("utf-8", errors="ignore")
            self.sent.append(msg)
            self.truncated.append(did_truncate)
        return {"parameters": {"success": True}}


@pytest.mark.asyncio
async def test_split_never_truncates_when_fitting_in_two():
    """Test that messages fitting in 2 * max_payload are never truncated.

    Regression test for issue where a sentence boundary early in the message
    would be preferred even if it left too much content for the second message.
    """
    sender = DummySender()
    target = "#test"
    max_payload = max(1, 512 - 12 - len(target.encode("utf-8")) - 60)

    # Create a message just under 2 * max_payload with sentence boundary early
    # This should NOT truncate - it should find a valid split point
    # Even if that means splitting at a less-nice boundary
    early_sentence = 50  # Sentence very early

    # "A"*50 + ". " + "B"*Xs where total > max_payload but <= 2*max_payload
    msg = (
        "A" * early_sentence
        + ". "
        + "B" * (max_payload - 20)  # This part alone exceeds when combined with As
    )
    total_len = len(msg)
    assert total_len > max_payload  # Needs split
    assert total_len <= 2 * max_payload  # Should fit in 2

    sender.sent.clear()
    await sender.send_message(target, msg, server="irc")

    # Must have 2 messages
    assert len(sender.sent) == 2

    # Combined must equal original (no truncation!)
    combined = sender.sent[0] + sender.sent[1]
    assert combined == msg, f"Truncation detected! Lost: {msg[len(combined) :]!r}"


@pytest.mark.asyncio
async def test_split_adds_ellipsis_when_exceeding_two_messages():
    """Test that messages exceeding 2 * max_payload get truncated with ellipsis."""
    sender = DummySender()
    target = "#test"
    max_payload = max(1, 512 - 12 - len(target.encode("utf-8")) - 60)

    # Create a message that's way too long
    msg = "A" * (3 * max_payload)  # 3x max_payload, won't fit in 2

    sender.sent.clear()
    await sender.send_message(target, msg, server="irc")

    # Must have 2 messages (we cap at 2)
    assert len(sender.sent) == 2

    # Second message should end with ellipsis
    assert sender.sent[1].endswith("...")

    # Combined should be shorter than original
    combined = sender.sent[0] + sender.sent[1]
    assert len(combined) < len(msg)


@pytest.mark.asyncio
async def test_split_integrity_with_multibyte_and_pipeline():
    """Single high-signal test that catches both classes of regressions.

    - Contains multibyte Cyrillic text to detect mid-codepoint splits or drops
    - Contains a long real-world shell pipeline to detect server-side truncation
      and boundary token loss (e.g., 'tee')
    - Asserts exact-prefix property across concatenated parts and that the
      simulated server never truncated (thanks to our safety margin)
    - Also implicitly asserts we cap at max two messages
    """
    sender = DummySenderTruncating()
    target = "#t"

    # Compute the sender-side payload so we can position the seam precisely
    max_payload = max(1, 512 - 12 - len(target.encode("utf-8")) - 100)

    # Build a head that includes the pipeline and lands just before the seam
    # We take a prefix of the pipeline to fit under the payload, keeping a 'tee' segment
    # Choose a small pipeline segment that still includes the tee token
    tee_seg = "| sudo tee /etc/apt/keyrings/spotify.gpg"
    # Build head: tee_seg + ASCII filler to exactly max_payload-1 bytes
    head_bytes_target = max_payload - 1
    ph_bytes = len(tee_seg.encode("utf-8"))
    filler_count = max(0, head_bytes_target - ph_bytes)

    # Build by bytes so the seam lands exactly between bytes of a multibyte code point.
    tee_b = tee_seg.encode("utf-8")
    filler_b = b"A" * filler_count
    # Append first byte of 'Ж' (0xD0) to end the head at byte index max_payload-1
    first_byte = b"\xd0"
    # Tail: continuation byte, 'X' sentinel, and Cyrillic chars that fit in 2 messages
    # Use fewer Cyrillic chars to fit in 2 * max_payload (each Ж is 2 bytes)
    cyrillic_count = (max_payload - 10) // 2  # Fit in second message with margin
    tail_b = b"\x96" + b"X" + ("Ж" * cyrillic_count).encode("utf-8")
    msg_b = tee_b + filler_b + first_byte + tail_b
    # Decode to str for sending
    msg = msg_b.decode("utf-8")

    ok = await sender.send_message(target, msg, server="irc")
    assert ok

    # At most two messages
    assert len(sender.sent) <= 2

    combined = "".join(sender.sent)
    # Strong invariants:
    # 1) No internal loss at character level
    assert combined == msg[: len(combined)]
    # 2) No internal loss at byte level (detects silent drops from naive decode errors="ignore")
    assert combined.encode("utf-8") == msg.encode("utf-8")[: len(combined.encode("utf-8"))]
    # 3) Key substrings from pipeline intact within combined
    assert "| sudo tee /etc/apt/keyrings/spotify.gpg" in combined
    # 4) Simulated server did not need to truncate
    assert not any(sender.truncated)


@pytest.mark.asyncio
async def test_split_prefers_middle_and_delimiter_priority():
    """Test that splitting prefers middle positions and respects delimiter priority.

    The algorithm should:
    1. Split as close to the middle as possible (not at the end)
    2. Prefer higher-priority delimiters: sentence > semicolon > comma > hyphen > space
    """
    sender = DummySender()
    target = "#test"

    # Compute max_payload for this target
    max_payload = max(1, 512 - 12 - len(target.encode("utf-8")) - 60)

    # Test 1: Sentence delimiter (". ") should be preferred when it's a valid split point
    # Create a message with a sentence boundary that leaves room for second part
    # Message = "A"s + ". " + "B"s, where second part fits in max_payload
    # Use (max_payload - 50) As + ". " + (max_payload - 50) Bs = ~2*max_payload - 98
    msg1 = "A" * (max_payload - 50) + ". " + "B" * (max_payload - 50)
    sender.sent.clear()
    await sender.send_message(target, msg1, server="irc")
    assert len(sender.sent) == 2
    # First part should end with ". " (the sentence boundary)
    assert sender.sent[0].endswith(". ")
    # Check it split at the sentence, not at the end
    first_len = len(sender.sent[0].encode("utf-8"))
    assert first_len < max_payload * 0.95  # Should be at sentence, not forced to end

    # Test 2: With semicolon and spaces, semicolon should be preferred when valid
    # Ensure semicolon is placed where splitting there leaves second part fitting
    msg2 = "X" * (max_payload - 50) + "; " + "Y " * 25 + "Z" * (max_payload - 100)
    sender.sent.clear()
    await sender.send_message(target, msg2, server="irc")
    assert len(sender.sent) == 2
    # Should split at semicolon, not at a later space
    assert sender.sent[0].endswith("; ")

    # Test 3: Comma vs space - comma should be preferred when it's a valid split
    msg3 = "W" * (max_payload - 50) + ", " + "V " * 25 + "U" * (max_payload - 100)
    sender.sent.clear()
    await sender.send_message(target, msg3, server="irc")
    assert len(sender.sent) == 2
    assert sender.sent[0].endswith(", ")

    # Test 4: Only spaces available - should split at space nearest to target
    # Use fewer words so message fits in 2 * max_payload (allows middle split)
    msg4 = " ".join(["word"] * 100)  # Fits in ~2x max_payload
    sender.sent.clear()
    await sender.send_message(target, msg4, server="irc")
    assert len(sender.sent) == 2
    first_bytes = len(sender.sent[0].encode("utf-8"))
    # Should be reasonably close to middle (within 70% of max) when total fits in 2 messages
    assert max_payload * 0.3 < first_bytes < max_payload * 0.8

    # Test 5: Verify sentence beats space when both are valid split points
    # Put sentence boundary at ~60% (valid for both messages to fit) with spaces after
    # Total ~1.5 * max_payload so sentence at 60% leaves ~40% = valid
    pos_60 = int(max_payload * 0.6)
    remaining = int(max_payload * 0.4) - 10  # Leave room in second part
    msg5 = "A" * (pos_60 - 2) + ". " + "B " * 20 + "C" * remaining
    sender.sent.clear()
    await sender.send_message(target, msg5, server="irc")
    assert len(sender.sent) == 2
    # Should prefer sentence boundary over later spaces
    assert sender.sent[0].endswith(". ")
