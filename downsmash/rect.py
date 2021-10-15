#!/usr/bin/python
"""Module for Rect class.
"""
import numpy as np


class Rect:
    """This class represents a rectangle.

    Used for representing the game screen and regions of interest (ROIs.)
    """

    def __init__(self, top, left, height, width):
        self.top = int(top)
        self.left = int(left)
        self.height = int(height)
        self.width = int(width)

    def __repr__(self):
        return "{height}x{width}+{top}+{left}".format(**self.__dict__)

    def __and__(self, other):
        """Return the intersection of `self` and `other`."""
        overlap_left = max(self.left, other.left)
        overlap_top = max(self.top, other.top)
        overlap_right = min(self.left + self.width, other.left + other.width)
        overlap_bottom = min(self.top + self.height, other.top + other.height)

        if overlap_left > overlap_right or overlap_top > overlap_bottom:
            return None
        return Rect(
            overlap_top,
            overlap_left,
            overlap_bottom - overlap_top,
            overlap_right - overlap_left,
        )

    def subregion(self, pct_top, pct_left, pct_height, pct_width, padding=0):
        """Return the subregion from
        (pct_top)*100% to (pct_top + pct_height)*100%,
        (pct_left)*100% to (pct_left + pct_width)*100%,
        plus a bevel of (padding)*100%,
        intersected with the screen.
        """
        top = self.top + (pct_top - padding) * self.height
        left = self.left + (pct_left - padding) * self.width

        height = (padding + pct_height + padding) * self.height
        width = (padding + pct_width + padding) * self.width

        return Rect(top, left, height, width) & self
