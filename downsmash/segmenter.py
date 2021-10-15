#!/usr/bin/python

import itertools

import cv2
import pandas as pd

from . import PERCENT, LOGGER
from .stream_parser import StreamParser
from .util import timeify, compute_minimum_kernel_density, bisect


class Segmenter:
    def __init__(self, filename, view, config):
        self.filename = filename
        self.stream = StreamParser(filename)
        self.view = view

        self.interval = config.get("polling_interval", 5)

        self.frames = self.stream.sample_frames(interval=self.interval)
        self.confidence = [(time, self.calculate_frame_confidence(scene, PERCENT, view.ports))
                           for (time, scene) in self.frames]

        self.confidence = pd.DataFrame(self.confidence, columns=['time', 'conf'])

    def calculate_frame_confidence(self, scene, feature, rois):
        """Estimate the maximum correlation of any ROI in _scene_
        to the unscaled _feature_.
        """

        scaled_feature = cv2.resize(feature, (0, 0), fx=self.view.scale, fy=self.view.scale)
        scaled_feature = cv2.Laplacian(scaled_feature, cv2.CV_8U)

        percent_corrs = []
        for roi in rois:
            if roi is not None:
                scene_roi = scene[roi.top:(roi.top + roi.height), roi.left:(roi.left + roi.width)]
                scene_roi = cv2.Laplacian(scene_roi, cv2.CV_8U)

                corr_map = cv2.matchTemplate(scene_roi, scaled_feature, cv2.TM_CCOEFF_NORMED)
                _, max_corr, _, _ = cv2.minMaxLoc(corr_map)
                percent_corrs.append(max_corr)
        return max(percent_corrs)

    def get_threshold(self):
        """Return an approximate threshold value to decide whether a frame
        contains Melee.
        """
        confs = self.confidence['conf']

        return compute_minimum_kernel_density(confs)

    def get_segments(self, threshold):
        """Return the approximate match start and end times for
        the given video.

        """
        # Perform median smoothing.
        self.confidence['median'] = self.confidence['conf'].rolling(5).median()
        self.confidence['median'] = self.confidence['median'].fillna(method='bfill')
        self.confidence['median'] = self.confidence['median'].fillna(method='ffill')

        # Now classify as Melee/no Melee based on whether we are greater/less
        # than the threshold.
        groups = itertools.groupby(self.confidence.iterrows(),
                                   lambda row: row[1]['median'] > threshold)
        groups = [(k, list(g)) for k, g in groups]
        segments = [(self.interval * g[0][0],
                     self.interval * g[-1][0]) for k, g in groups if k]

        for idx, segment in enumerate(segments):
            start, end = segment
            LOGGER.warning("Estimated game %d is %s-%s", idx + 1, timeify(start), timeify(end))

        return segments

    def refine_segments(self, segments):
        for idx, segment in enumerate(segments):
            start, end = segment
            start = self.find_segment_boundary(start, 0.5)
            end = self.find_segment_boundary(end, 0.5)
            segments[idx] = (start, end)
            LOGGER.warning("Estimated game %d is %s-%s", idx + 1, timeify(start), timeify(end))

        return segments

    def find_segment_boundary(self, time, tolerance):
        """Find the time index of a match segment boundary (start or end)
        near _time_, accurate to within _tolerance_ seconds.
        Uses the bisection method to find an approximate solution for
        f(t) = conf_at(t) - self.threshold = 0.
        """

        threshold = self.get_threshold()
        def conf_at(time):
            scene = self.stream.get_frame(time)
            if scene is not None:
                conf = self.calculate_frame_confidence(scene, PERCENT, self.view.ports)
                return conf - threshold
            return 0 - threshold

        window = self.interval
        for _ in range(20):
            start = max(0, time - window)
            # Have to read from strictly before the end of the video.
            end = min(self.stream.length - tolerance, time + window)
            try:
                return bisect(conf_at, start, end, tolerance)
            except ValueError:  # bad interval --- no sign change
                window += tolerance

                # Make sure we didn't hit the boundaries of the video.
                if start == 0:
                    return start
                if end == self.stream.length - tolerance:
                    return end

        raise ValueError("Could not find a match boundary.")
