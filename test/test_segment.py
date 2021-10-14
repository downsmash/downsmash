import pytest

from downsmash.segmenter import Segmenter
from downsmash.watcher import watch

class TestSegment:
    def test_ice1(self):
        with pytest.raises(RuntimeError, match=r".*gapless.*"):
            watch("data/ice1.mp4")
