import pytest

from downsmash.stream_parser import StreamParser

class TestStream:
    def test_open_fails(self):
        with pytest.raises(RuntimeError):
            StreamParser("asdmakdmlsdmfa")

    def test_color_frames(self):
        stream = StreamParser("data/ice1.mp4")
        stream.get_frame(color=True)
