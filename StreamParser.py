#!/usr/bin/python

import cv2
import numpy as np
from random import randint
import logging
import scipy.stats

from Rect import Rect
from TemplateMatcher import TemplateMatcher

logging.basicConfig(format="%(message)s")


class StreamParser:

    def __init__(self, filename):
        self.filename = filename
        self.vc = cv2.VideoCapture(filename)
        self.shape = Rect(0, 0, self.vc.get(4), self.vc.get(3))

    def parse(self):
        raise NotImplementedError

    def locate(self, feature, roi=None, tm=TemplateMatcher(), N=10,
               debug=False):
        peaks = []
        best_scale_log = []

        for (n, scene) in self.sample_frames(num_samples=N):
            cv2.imwrite("scene.png", scene)
            scene = cv2.imread("scene.png")

            if roi:
                mask = roi.to_mask(self.shape.height, self.shape.width)
            else:
                mask = None

            scale, these_peaks = tm.match(feature, scene,
                                          mask=mask,
                                          debug=debug)

            if scale:
                best_scale_log += [scale]

                these_peaks = sorted(these_peaks, key=lambda pt: pt[1])
                these_peaks = [loc for loc, corr in these_peaks]

                peaks.extend(these_peaks[:tm.max_clusters])

        feature_locations = [np.array(max(set(cluster), key=cluster.count))
                             for cluster in tm.get_clusters(peaks)]
        feature_locations = sorted(feature_locations, key=lambda pt: pt[1])

        if best_scale_log:
            mean_best_scale = sum(best_scale_log) / len(best_scale_log)
        else:
            mean_best_scale = None

        return (mean_best_scale, feature_locations)

    def sample_frames(self, start=None, end=None, interval=None,
                      num_samples=None, fuzz=0):
        if (interval is None and num_samples is None) or \
                None not in (interval, num_samples):
            raise ValueError('exactly one of (interval, num_samples) '
                             'must be set')

        video_length = self.vc.get(7) / self.vc.get(5)
        if not start or start < 0:
            start = 0
        if not end or end > video_length:
            end = video_length

        total_time = end - start

        if not num_samples:
            num_samples = total_time // interval

        for time in np.linspace(start, end, num=num_samples):
            time += randint(-1 * fuzz, fuzz) / self.vc.get(5)

            if time < start:
                time = start
            elif time > end:
                time = end

            self.vc.set(0, int(time * 1000))
            success, frame = self.vc.read()

            if success:
                yield (time, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        return

    def overlay_map(self, num_samples=50, begin=None, end=None):
        """Run a skewness-kurtosis filter on a sample of frames and
        edge-detect.

        The areas of the video containing game feed should come back black.
        Areas containing overlay or letterboxes will be visibly white.
        """
        data = None
        for time, frame in self.sample_frames(num_samples=num_samples,
                                              start=begin, end=end):
            if not data:
                data = [frame]
            else:
                data += [frame]

        sd_map = np.sqrt(np.var(data, axis=0))
        skew_map = scipy.stats.skew(data, axis=0)
        kurt_map = scipy.stats.kurtosis(data, axis=0)
        min_map = np.minimum(skew_map, kurt_map)

        map_min = min(min_map.flatten())
        map_max = max(min_map.flatten())

        # Clip to [0, 255], with 0=min and 255=max
        clipped = ((min_map - map_min)/(map_max - map_min) * 255)
        clipped = clipped.astype(np.uint8)

        # Blur and edge detect.
        blurred = cv2.blur(clipped, (5, 5))
        edges = cv2.Canny(blurred, 50, 150)

        # Areas that are constant throughout the video (letterboxes) will
        # have 0 skew, 0 kurt, and 0 variance, so the skew-kurt filter
        # will miss them
        edges[np.where(sd_map < 0.01)] = 255

        return edges
