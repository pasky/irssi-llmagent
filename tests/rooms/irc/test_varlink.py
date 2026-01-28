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
    # Tail begins with the continuation byte (0x96), which a naive errors="ignore" decoder will drop,
    # then 'X' sentinel, then a long run of 'Ж'
    tail_b = b"\x96" + b"X" + ("Ж" * 5000).encode("utf-8")
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
    # 3) Seam must be followed by multibyte immediately
    if len(sender.sent) == 2:
        assert ord(sender.sent[1][0]) > 127
    # 4) Key substrings from pipeline intact within combined
    assert "| sudo tee /etc/apt/keyrings/spotify.gpg" in combined
    # 5) Simulated server did not need to truncate
    assert not any(sender.truncated)
