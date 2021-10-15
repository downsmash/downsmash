import pytest

from downsmash.watcher import watch


class TestSegment:
    def test_ice1(self):
        with pytest.raises(RuntimeError, match=r".*gapless.*"):
            watch("data/ice1.mp4")

    def test_ice2(self):
        with pytest.raises(RuntimeError, match=r".*gapless.*"):
            watch("data/ice2.mp4")

    def test_smuckers(self):
        with pytest.raises(RuntimeError, match=r".*gapless.*"):
            watch("data/smuckers.mp4")
