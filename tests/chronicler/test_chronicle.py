import pytest

from muaddib.chronicler.chronicle import Chronicle


@pytest.mark.asyncio
async def test_chronicle_append_and_render(temp_db_path):
    chron = Chronicle(temp_db_path)
    await chron.initialize()

    # Initially, render should say no chapters
    empty_render = await chron.render_chapter("work-arc")
    assert "No chapters" in empty_render

    # Append first paragraph (opens chapter implicitly)
    para = await chron.append_paragraph("work-arc", "Wrapped up CI flake; pin runner image next.")
    assert para["id"] > 0

    # Append another
    await chron.append_paragraph("work-arc", "Drafted chronicler MVP tests.")

    # Render full chapter
    out = await chron.render_chapter("work-arc")
    assert "# Arc: work-arc — Chapter" in out
    assert "Paragraphs:" in out
    assert "Wrapped up CI flake" in out
    assert "Drafted chronicler MVP tests." in out

    # Render last 1
    out_last = await chron.render_chapter("work-arc", last_n=1)
    assert "Drafted chronicler MVP tests." in out_last
    assert "Wrapped up CI flake" not in out_last
