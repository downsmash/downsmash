#!/usr/bin/python

import logging
import itertools

import numpy as np
import cv2
import pandas as pd
from sklearn.neighbors.kde import KernelDensity
from scipy.signal import argrelmin

from pkg_resources import resource_string

from .rect import Rect
from .streamParser import StreamParser
from .templateMatcher import TemplateMatcher
from .viewfinder import Viewfinder

# Read in percent sign
nparr = np.fromstring(resource_string("core.resources", "pct.png"), np.uint8)
PERCENT = cv2.imdecode(nparr, 1)

logger = logging.getLogger(__name__)


class Segmenter(StreamParser):

    def __init__(self, filename, polling_interval=2, min_gap=10):
        StreamParser.__init__(self, filename)
        self.polling_interval = polling_interval
        self.min_gap = min_gap
        self.data = dict()

    def parse(self):
        vf = Viewfinder(self.filename, polling_interval=self.polling_interval)

        logging.warn("Video OK, looking for screen...")
        self.data["screen"] = vf.detect_screen()
        logger.warn("Estimated screen is {screen}".format(**self.data))

        self.data["scale"] = vf.scale
        logger.warn("Estimated scale is {scale}".format(**self.data))

        logging.warn("Checking port locations...")
        self.data["ports"], _, _ = vf.detect_ports()
        if not self.data["ports"]:
            raise RuntimeError("No ports found!")
        logger.warn("Ports are at {0} {1} {2} {3}".format(*self.data["ports"]))

        logging.warn("Beginning segmentation...")
        self.data["matches"] = self.detect_match_chunks()

        for n, match in enumerate(self.data["matches"]):
            start, end = match
            logger.warn("Estimated game {0} is {1}-{2} s".format(n + 1, start, end))

        logging.warn("Refining match boundaries...")
        for n, match in enumerate(self.data["matches"]):
            start, end = match
            start = self.find_match_boundary(start)
            end = self.find_match_boundary(end)
            self.data["matches"][n] = (start, end)
            logger.warn("Estimated game {0} is {1}-{2} s".format(n + 1, start, end))

    def calculate_frame_confidence(self, scene):
        cv2.imwrite("scene.png", scene)
        scene = cv2.imread("scene.png")

        scaled_percent = cv2.resize(
            PERCENT, (0, 0), fx=self.data["scale"], fy=self.data["scale"])
        scaled_percent = cv2.Laplacian(scaled_percent, cv2.CV_8U)

        percent_corrs = []
        for port_number, roi in enumerate(self.data["ports"]):
            if roi is not None:
                scene_roi = scene[roi.top:(roi.top + roi.height),
                                  roi.left:(roi.left + roi.width)]
                scene_roi = cv2.Laplacian(scene_roi, cv2.CV_8U)

                corr_map = cv2.matchTemplate(scene_roi, scaled_percent,
                                             cv2.TM_CCOEFF_NORMED)
                _, max_corr, _, max_loc = cv2.minMaxLoc(corr_map)
                percent_corrs.append(max_corr)
        return max(percent_corrs)

    def detect_match_chunks(self, max_error=.06):
        conf_series = []

        for (t, scene) in self.sample_frames(interval=self.polling_interval):
            conf = self.calculate_frame_confidence(scene)
            point = [t, conf]

            logger.info("{0}\t{1}".format(*point))
            conf_series.append(point)

        conf_series = pd.DataFrame(conf_series, columns=['time', 'conf'])
        confs = conf_series['conf']

        # Trim outliers for robustness.
        p05, p95 = confs.quantile((0.05, 0.95))
        samples = np.linspace(p05, p95, num=100)

        # Find the minimum kernel density.
        kde = KernelDensity(kernel='gaussian', bandwidth=.005)
        kde = kde.fit(np.array(confs).reshape(-1, 1))
        e = kde.score_samples(samples.reshape(-1, 1))

        rel_mins = argrelmin(e)[0]
        deepest_min = max(rel_mins, key=lambda idx: min(e[idx - 1] - e[idx],
                                                        e[idx + 1] - e[idx]))
        self.threshold = samples[deepest_min]

        # How separated are the two groups?
        mean_positive = np.mean(confs[confs >= self.threshold])
        mean_negative = np.mean(confs[confs < self.threshold])
        logger.warn("Group means are (+){0} (-){1}".format(
                    mean_positive, mean_negative))

        if mean_positive - mean_negative < 0.1 or mean_negative > 0.5:
            logger.warn("This looks like an edited/gapless set"
                        "(mean_pos - mean_neg = {:03f})".format(
                        mean_positive - mean_negative))
            raise RuntimeError()

        # Perform median smoothing.
        conf_series['median'] = conf_series['conf'].rolling(5).median()
        conf_series['median'] = conf_series['median'].fillna(method='bfill')
        conf_series['median'] = conf_series['median'].fillna(method='ffill')

        # Now classify as Melee/no Melee based on whether we are greater/less
        # than the threshold.
        groups = itertools.groupby(conf_series.iterrows(),
                                   lambda row: row[1]['median'] > self.threshold)
        groups = [(k, list(g)) for k, g in groups]
        matches = [(self.polling_interval * g[0][0],
                    self.polling_interval * g[-1][0]) for k, g in groups if k]

        return matches

    def find_match_boundary(self, t):
        # Bisection method.
        # f(t) = conf_at(t) - self.threshold
        def conf_at(t):
            scene = self.get_frame(t)
            return self.calculate_frame_confidence(scene)

        window = self.min_gap * 1.5

        start = max(0, t - window / 2)
        end = min(self.length, t + window / 2)

        # First make sure we have an interval to which bisection is applicable.
        # Also compute start and end confs.
        plus_to_minus = None
        minus_to_plus = None
        while not (plus_to_minus or minus_to_plus):
            if plus_to_minus is not None:
                start += 1
            if minus_to_plus is not None:
                end -= 1

            if start >= end:
                return None

            start_conf = conf_at(start)
            end_conf = conf_at(end)

            plus_to_minus = (start_conf > self.threshold > end_conf)
            minus_to_plus = (start_conf < self.threshold < end_conf)

        while end - start > 0.1:
            middle = (start + end) / 2
            middle_conf = conf_at(middle)
            if np.sign(middle_conf - self.threshold) == np.sign(start_conf - self.threshold):
                start = middle
            else:
                end = middle

        return (start + end) / 2
