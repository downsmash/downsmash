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

        self.data["screen"] = vf.detect_screen()
        self.data["scale"] = vf.scale
        logger.warn("Screen is at {0}".format(self.data["screen"]))

        logger.warn("Estimated scale is {scale}".format(**self.data))

        self.data["ports"], _, _ = vf.detect_ports()
        if not self.data["ports"]:
            raise RuntimeError("No ports found!")

        logger.warn("Ports are at {0} {1} {2} {3}".format(*self.data["ports"]))

        self.data["chunks"] = self.detect_match_chunks()

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

            logger.warn("{0}\t{1}".format(*point))
            conf_series.append(point)

        # Perform median smoothing.
        conf_series = pd.DataFrame(conf_series, columns=['time', 'conf'])
        conf_series['median'] = conf_series['conf'].rolling(5).median()
        conf_series['median'] = conf_series['median'].fillna(method='bfill')
        conf_series['median'] = conf_series['median'].fillna(method='ffill')

        # Find the minimum kernel density.
        kde = KernelDensity(kernel='gaussian', bandwidth=.005)
        kde = kde.fit(np.array(conf_series['median']).reshape(-1, 1))
        e = kde.score_samples(np.linspace(0, 1, num=100).reshape(-1, 1))

        rel_mins = argrelmin(e)[0]
        deepest_min = max(rel_mins, key=lambda idx: min(e[idx - 1] - e[idx],
                                                        e[idx + 1] - e[idx]))

        # Now classify as Melee/no Melee based on whether we are greater/less
        # than the argrelmin.
        split_point = deepest_min / 100
        groups = itertools.groupby(conf_series.iterrows(),
                                   lambda row: row[1]['median'] > split_point)
        groups = [(k, list(g)) for k, g in groups]
        matches = [(self.polling_interval * g[0][0],
                    self.polling_interval * g[-1][0]) for k, g in groups if k]

        return matches
