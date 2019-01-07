#!/usr/bin/python

import cv2
import numpy as np
from random import randint
import logging
import scipy.stats
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelmax

from .rect import Rect
from .templateMatcher import TemplateMatcher

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

        for (_, scene) in self.sample_frames(num_samples=N):
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
                if debug:
                    logging.warn("%s", "\t".join(str(k) for k in these_peaks))

                peaks.extend(these_peaks)

        feature_locations = [np.array(max(set(cluster), key=cluster.count))
                             for cluster in tm.get_clusters_DBSCAN(peaks)]
        feature_locations = sorted(feature_locations, key=lambda pt: pt[1])

        if best_scale_log:
            median_best_scale = np.mean(best_scale_log)
        else:
            median_best_scale = None

        return (median_best_scale, feature_locations)

    def sample_frames(self, start=None, end=None, interval=None,
                      num_samples=None, fuzz=0, color=False):
        if (interval is None and num_samples is None) or \
                None not in (interval, num_samples):
            raise ValueError('exactly one of (interval, num_samples) '
                             'must be set')

        # TODO Make sure the VC object actually works so we don't divide by
        # zero here
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
                cv2.imwrite('scene.png', frame)
                frame = cv2.imread('scene.png', cv2.IMREAD_COLOR)

                logging.info('%d\n', time)
                if color:
                    yield (time, frame)
                else:
                    yield (time, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        return
