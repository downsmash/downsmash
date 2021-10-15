import numpy as np

from downsmash.util import scale_to_interval

class TestUtil:
    def test_scale_to_interval_with_equal_params(self):
        array = np.array([3, 3, 3])
        assert (scale_to_interval(array, 5, 5) == np.array([5, 5, 5])).all()
