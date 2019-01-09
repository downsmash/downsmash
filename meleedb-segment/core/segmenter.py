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


class Segmenter(StreamParser):

    def __init__(self, filename, polling_interval=2, min_gap=10):
        StreamParser.__init__(self, filename)
        self.polling_interval = polling_interval
        self.min_gap = min_gap
        self.data = dict()

    def parse(self):
        vf = Viewfinder(self.filename, polling_interval=self.polling_interval)

        self.data["screen"] = vf.detect_screen()
        self.data["scale"] = (self.data["screen"].width / 548 + self.data["screen"].height / 411) / 2
        logging.warn("Screen is at {0}".format(self.data["screen"]))

        logging.warn("Estimated scale is {scale}".format(**self.data))

        self.data["ports"] = vf.detect_ports()
        if not self.data["ports"]:
            raise RuntimeError("No ports found!")

        logging.warn("Ports are at {0} {1} {2} {3}".format(*self.data["ports"]))

        self.data["chunks"] = self.detect_match_chunks()

<<<<<<< HEAD
=======
    def detect_screen(self):
        """Attempt to detect the screen.
        """

        # Detect the percent signs in a random sample of frames.
        tm = TemplateMatcher(max_clusters=2, scales=np.arange(0.6, 1.1, 0.015))
        scale, pct_locations = self.locate(PERCENT, tm=tm, N=30)

        # Group the returned locations to within 5 px tolerance on y-axis.
        pct_locations = sorted(pct_locations, key=lambda l: l[0] // 5)
        if not pct_locations:
            raise RuntimeError("No percent signs found!")

        location_groups = itertools.groupby(pct_locations, lambda l: l[0] // 5)
        location_groups = [(k, list(g)) for k, g in location_groups]

        # Choose the biggest group.
        # TODO: Try to get locate() to use KDE.
        _, pct_locations = max(location_groups, key=lambda g: len(g[1]))
        pct_locations = list(pct_locations)

        # Approximate screen Y-pos from percents.
        height, width = [x * scale / 0.05835 for x in PERCENT.shape[:2]]
        top = np.mean(pct_locations, axis=0)[0] - .871 * height

        # Determine the X-pos by the skewness-kurtosis method.
        logging.info("Generating skew-kurtosis map...")
        overlay = self.overlay_map()
        leftmost_pct = min(pct_locations, key=lambda pos: pos[1])[1]

        leftmost_port = None
        best_goods = 0

        # The leftmost percent sign can be one of four ports.
        for port_no in range(4):
            left = leftmost_pct - (.2 + .2381 * port_no) * width
            screen = Rect(top, left, height, width) & self.shape
            pixels = overlay[screen.top:(screen.top + screen.height),
                             screen.left:(screen.left + screen.width)]
            goods = np.count_nonzero(pixels == 0)
            if goods > best_goods:
                best_goods = goods
                leftmost_port = port_no

        left = leftmost_pct - (.2 + .2381 * leftmost_port) * width
        # self.data["scale"] = (height / 411 + width / 548) / 2
        self.data["scale"] = scale

        return Rect(top, left, height, width) & self.shape

    def detect_ports(self, max_error=0.06):
        ports = []
        # TODO DRY this out
        for port_number in range(4):
            (pct_top, pct_left) = self.data["screen"][.87, .2 + .2381 * port_number]
            pct_roi_top = pct_top - max_error * self.data["screen"].height
            pct_roi_left = pct_left - max_error * self.data["screen"].width
            pct_roi_height = (.06 + 2 * max_error) * self.data["screen"].height
            pct_roi_width = (.06 + 2 * max_error) * self.data["screen"].width
            pct_roi = Rect(pct_roi_top, pct_roi_left,
                           pct_roi_height, pct_roi_width)

            pct_roi &= self.data["screen"]

            tm = TemplateMatcher(max_clusters=1, scales=[self.data["scale"]],
                                 worst_match=0.6)
            scale, location = self.locate(PERCENT, N=10, tm=tm, roi=pct_roi)
            if scale is None:
                ports.append(None)
                continue

            error = location[0] - (pct_top, pct_left)
            logging.warn("Detected port {0} at {1} "
                         "(error {2[0]}px, {2[1]}px)"
                         .format(port_number + 1, location[0], error))

            (port_top, port_left) = self.data["screen"][.75, .0363 + .24 * port_number]
            port_roi_top = port_top - max_error * self.data["screen"].height
            port_roi_left = port_left - max_error * self.data["screen"].width
            port_roi_height = (.18 + 2 * max_error) * self.data["screen"].height
            port_roi_width = (.1833 + 2 * max_error) * self.data["screen"].width
            port_roi = Rect(port_roi_top, port_roi_left,
                            port_roi_height, port_roi_width)

            port_roi &= self.data["screen"]
            ports.append(port_roi)
        else:
            ports.append(None)

        return ports

>>>>>>> master
    def calculate_frame_confidence(self, scene):
        cv2.imwrite("scene.png", scene)
        scene = cv2.imread("scene.png")

        scaled_percent = cv2.resize(
            PERCENT, (0, 0), fx=self.data["scale"], fy=self.data["scale"])
        scaled_percent = cv2.Canny(scaled_percent, 50, 200)

        percent_corrs = []
        for port_number, roi in enumerate(self.data["ports"]):
            if roi is not None:
                scene_roi = scene[roi.top:(roi.top + roi.height),
                                  roi.left:(roi.left + roi.width)]
                scene_roi = cv2.Canny(scene_roi, 50, 200)

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

            logging.warn("{0}\t{1}".format(*point))
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
