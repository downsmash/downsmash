"""Module for StreamParser class.
"""

from random import randint
import logging

import cv2
import numpy as np

from .cluster import get_clusters
from .rect import Rect
from .template_matcher import TemplateMatcher

LOGGER = logging.getLogger(__name__)


class StreamParser:
    """This class is effectively a wrapper around OpenCV's VideoCapture class
    that provides utility functions for parsing video streams.
    """

    def __init__(self, filename, debug=False):
        self.filename = filename
        cap = cv2.VideoCapture(filename)

        if not cap.isOpened():
            raise RuntimeError('File "{0}" could not be read from.'
                               .format(filename))

        self.cap = cap
        try:
            self.length = self.cap.get(7) / self.cap.get(5)
        except ZeroDivisionError:
            raise RuntimeError('Video "{0}" has no frames.'.format(filename))

        self.shape = Rect(0, 0, self.cap.get(4), self.cap.get(3))
        self.debug = debug

    def locate(self, feature, roi=None, matcher=TemplateMatcher(),
               num_samples=10):
        """asdfkjasdlfjas
        """
        peaks = []
        best_scale_log = []

        for (_, scene) in self.sample_frames(num_samples=num_samples):
            cv2.imwrite("scene.png", scene)
            scene = cv2.imread("scene.png")

            if roi:
                max_clusters = 1
                mask = roi.to_mask(self.shape.height, self.shape.width)
            else:
                max_clusters = None
                mask = None

            scale, these_peaks = matcher.match(feature, scene, mask=mask)

            if scale:
                best_scale_log += [scale]

                these_peaks = sorted(these_peaks, key=lambda pt: pt[1])
                these_peaks = [loc for loc, corr in these_peaks]
                if self.debug:
                    LOGGER.warning("%s",
                                   "\t".join(str(k) for k in these_peaks))

                peaks.extend(these_peaks)

        clusters = get_clusters(peaks, max_clusters=max_clusters)

        feature_locations = [np.array(max(set(cluster), key=cluster.count))
                             for cluster in clusters]
        feature_locations = sorted(feature_locations, key=lambda pt: pt[1])

        if best_scale_log:
            median_best_scale = np.mean(best_scale_log)
        else:
            median_best_scale = None

        return (median_best_scale, feature_locations)

    def get_frame(self, time=None, color=False):
        """Attempt to retrieve a frame from `self.stream`.

        Parameters:
            - `time`: If set, get the frame nearest to this timestamp.
               If unset, get the next frame in sequence.
            - `color`: If false, convert the frame from color to grayscale.
        """
        if time is not None:
            self.cap.set(0, int(time * 1000))

        success, frame = self.cap.read()

        if success:
            cv2.imwrite('scene.png', frame)
            frame = cv2.imread('scene.png', cv2.IMREAD_COLOR)

            LOGGER.info('%d\n', time)
            if color:
                return frame
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return None

    def sample_frame_timestamps(self, start, end, num_samples, fuzz):
        """
        """
        framerate = self.cap.get(5)
        for time in np.linspace(start, end, num=num_samples):
            time += randint(-1 * fuzz, fuzz) / framerate

            if time < start:
                time = start
            elif time > end:
                time = end

            yield time

    def sample_frames(self, start=None, end=None, interval=None,
                      num_samples=None, fuzz=0, color=False):
        """Generate frames based on the parameters given.
        """
        if (interval is None and num_samples is None) or \
                None not in (interval, num_samples):
            raise ValueError('exactly one of (interval, num_samples) '
                             'must be set')

        if start is None or start < 0:
            start = 0
        if end is None or end > self.length:
            end = self.length

        total_time = end - start

        if num_samples is None:
            num_samples = total_time // interval

        for time in self.sample_frame_timestamps(start, end, num_samples, fuzz):
            frame = self.get_frame(time, color=color)
            if frame is not None:
                yield (time, frame)
