#!/usr/bin/python
import numpy as np

class ROI:

    def __init__(self, top, left, height, width):
        self.top, self.left, self.height, self.width = [int(x) for x in (top, left, height, width)]

    def __str__(self):
        return '{height}x{width}+{top}+{left}'.format(**self.__dict__)

    def __repr__(self):
        return '{height}x{width}+{top}+{left}'.format(**self.__dict__)

    def __and__(self, other):
        overlap_left, overlap_top = max(self.left, other.left), max(self.top, other.top)
        overlap_right = min(self.left + self.width, other.left + other.width)
        overlap_bottom = min(self.top + self.height, other.top + other.height)
        if overlap_left > overlap_right or overlap_top > overlap_bottom:
            return None
        else:
            return ROI(overlap_top, overlap_left, overlap_bottom - overlap_top, overlap_right - overlap_left)

    def __getitem__(self, given):
        if isinstance(given, slice):
            if given.step:
                raise ValueError('ROI getitem steps are not supported')
            top, left = given.start
            bottom, right = given.end
            return ROI(top * self.height, left * self.width, (bottom - top) * self.height, (right - left) * self.width)
        else:
            y, x = given
            return np.array((int(self.top + y * self.height),
                             int(self.left + x * self.width)))

