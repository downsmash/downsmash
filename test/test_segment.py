import pytest

from downsmash.segmenter import Segmenter

class TestSegment:
    def test_ice1(self):
        segmenter = Segmenter("data/ice1.mp4")
        with pytest.raises(RuntimeError, match=r".*gapless.*"):
            segmenter.parse()
