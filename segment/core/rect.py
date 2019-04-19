#!/usr/bin/python
import numpy as np


class Rect:

    def __init__(self, top, left, height, width):
        self.top = int(top)
        self.left = int(left)
        self.height = int(height)
        self.width = int(width)
        self.children = []

    def __repr__(self):
        this_repr = ['{height}x{width}+{top}+{left}'.format(**self.__dict__)]
        for child in self.children:
            for line in repr(child).split("\n"):
                this_repr.append("-> " + line)

        return "\n".join(this_repr)

    def __and__(self, other):
        """Return the intersection of ''self'' and ''other''.
        """
        overlap_left = max(self.left, other.left)
        overlap_top = max(self.top, other.top)
        overlap_right = min(self.left + self.width, other.left + other.width)
        overlap_bottom = min(self.top + self.height, other.top + other.height)

        if overlap_left > overlap_right or overlap_top > overlap_bottom:
            return None
        return Rect(overlap_top,
                    overlap_left,
                    overlap_bottom - overlap_top,
                    overlap_right - overlap_left)

    def to_mask(self, height, width):
        """Generate a zero-one mask from this rectangle.
        """
        mask = np.zeros((height, width, 3))

        mask[self.top:(self.top + self.height + 1),
             self.left:(self.left + self.width + 1)] = (1, 1, 1)

        return mask.astype(np.uint8)

    def subregion(self, pct_top, pct_left, pct_height, pct_width,  #pylint:disable=too-many-arguments
                  padding=0):
        """Return the subregion from (pct_top)*100% to (pct_top + pct_height)*100%,
        (pct_left)*100% to (pct_left + pct_width)*100%, plus a bevel of (padding)*100%,
        intersected with the screen.
        """
        top = self.top + (pct_top - padding) * self.height
        left = self.left + (pct_left - padding) * self.width

        height = (padding + pct_height + padding) * self.height
        width = (padding + pct_width + padding) * self.width

        return Rect(top, left, height, width) & self
