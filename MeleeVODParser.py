#!/usr/bin/python

import numpy as np
import cv2
import logging
import pandas as pd
from sklearn.cluster import DBSCAN
from itertools import groupby

from ROI import ROI
from StreamParser import StreamParser
from TemplateMatcher import TemplateMatcher


class MeleeVODParser(StreamParser):

    def __init__(self, filename, polling_interval=2, min_gap=10):
        StreamParser.__init__(self, filename)
        self.polling_interval = polling_interval
        self.min_gap = min_gap

    def parse(self):
        self.screen = self.detect_screen()
        logging.warn("Screen is at {0}".format(self.screen))
        logging.warn("Estimated scale is {scale}".format(**self.__dict__))
        self.ports = self.detect_ports()
        logging.warn("Ports are at {0} {1} {2} {3}".format(*self.ports))
        self.chunks = self.detect_match_chunks()
        for chunk in self.chunks:
            logging.warn(chunk)

    def detect_screen(self):
        percent = cv2.imread("assets/pct.png")
        tm = TemplateMatcher(max_clusters=2)
        scale, percent_locations = self.locate(percent, tm=tm, N=30)
        print(percent_locations)

        clock_digit_locations = []
        for digit in [2, 3, 4, 5, 6, 8]:
            feature = cv2.imread("assets/{0}_time.png".format(digit))
            tm = TemplateMatcher(scales=np.arange(0.8, 1.0, 0.03))
            _, digit_locations = self.locate(feature, tm=tm, N=20)

            clock_digit_locations.extend(digit_locations)

        print(clock_digit_locations)

        # Estimate bounding box from percent and clock locations
        height, width = [x * scale / 0.05835 for x in percent.shape[:2]]
        top = np.mean(percent_locations, axis=0)[0] - .871 * height

        clock_center = np.mean(clock_digit_locations, axis=0)[1]
        left = clock_center - .475 * width

        self.scale = (height / 411 * width / 548)**0.5

        return ROI(top, left, height, width) & self.roi

    def detect_ports(self, max_error=0.06):
        ports = []
        percent = cv2.imread("assets/pct.png")
        for port_number in range(4):
            (pct_top, pct_left) = self.screen[.87, .2 + .2381 * port_number]
            pct_roi_top = pct_top - max_error * self.screen.height
            pct_roi_left = pct_left - max_error * self.screen.width
            pct_roi_height = (.06 + 2 * max_error) * self.screen.height
            pct_roi_width = (.06 + 2 * max_error) * self.screen.width
            pct_roi = ROI(pct_roi_top, pct_roi_left,
                          pct_roi_height, pct_roi_width)

            pct_roi &= self.screen

            tm = TemplateMatcher(max_clusters=1)
            scale, location = self.locate(percent, N=10, tm=tm,
                                          roi=pct_roi, debug=True)
            if scale is None:
                ports.append(None)
                continue

            error = location[0] - (pct_top, pct_left)
            logging.warn("Detected port {0} at {1} "
                         "(error {2[0]}px, {2[1]}px)"
                         .format(port_number + 1, location[0], error))

            (port_top, port_left) = self.screen[.75, .0363 + .24 * port_number]
            port_roi_top = port_top - max_error * self.screen.height
            port_roi_left = port_left - max_error * self.screen.width
            port_roi_height = (.18 + 2 * max_error) * self.screen.height
            port_roi_width = (.1833 + 2 * max_error) * self.screen.width
            port_roi = ROI(port_roi_top, port_roi_left,
                           port_roi_height, port_roi_width)

            port_roi &= self.screen
            ports.append(port_roi)
        else:
            ports.append(None)

        return ports

    def detect_match_chunks(self, max_error=.06):
        percent = cv2.imread("assets/pct.png")
        corr_series = []

        for (t, scene) in self.sample_frames(interval=self.polling_interval):
            cv2.imwrite("scene.png", scene)
            scene = cv2.imread("scene.png")

            scaled_percent = cv2.resize(
                percent, (0, 0), fx=self.scale, fy=self.scale)
            scaled_percent = cv2.Canny(scaled_percent, 50, 200)

            percent_corrs = []
            for port_number, roi in enumerate(self.ports):
                if roi is not None:
                    scene_roi = scene[roi.top:(roi.top + roi.height),
                                      roi.left:(roi.left + roi.width)]
                    scene_roi = cv2.Canny(scene_roi, 50, 200)

                    corr_map = cv2.matchTemplate(scene_roi, scaled_percent,
                                                 cv2.TM_CCOEFF_NORMED)
                    _, max_corr, _, max_loc = cv2.minMaxLoc(corr_map)
                    percent_corrs.append(max_corr)

            point = [t, max(percent_corrs)]
            corr_series.append(point)

        corr_series = np.array(corr_series)

        medians = pd.rolling_median(corr_series[:, 1], self.min_gap //
                                    self.polling_interval, center=True)[2:-2]

        clusters = DBSCAN(eps=0.03, min_samples=10).fit(medians.reshape(-1, 1))

        dataframe = zip(corr_series[:, 0][2:-2], medians, clusters.labels_)
        dataframe = list(dataframe)

        labels = list(set(x[2] for x in dataframe))
        cluster_means = [sum(cluster) / len(cluster) for cluster
                         in [[x[1] for x in dataframe if x[2] == label]
                         for label in labels]]
        cluster_means = list(zip(labels, cluster_means))

        game_label = max(cluster_means, key=lambda x: x[1])[0]
        game_lists = [(k, list(v)) for k, v
                      in groupby(dataframe, lambda pt: pt[2])]
        games = [[v[0][0], v[-1][0]] for k, v in game_lists if k == game_label]

        return games

    def get_percents(self, chunk, interval=0.2):
        start, end = chunk
        for time, frame in self.sample_frames(start=start, end=end,
                                              interval=interval):
            percents = [time]
            for idx, port in enumerate(self.ports):
                if port is None:
                    continue

                digit_locations = []
                for digit in range(0, 10):
                    if digit == 1:
                        feature = cv2.imread("assets/{0}_pct_cropped.png"
                                             .format(digit), 0)
                    else:
                        feature = cv2.imread("assets/{0}_pct.png"
                                             .format(digit), 0)

                    pct_scales = np.arange(self.scale - 0.01,
                                           self.scale + 0.05,
                                           0.03)

                    tm = TemplateMatcher(scales=pct_scales,
                                         thresh_min=175,
                                         thresh_max=350)
                    scale, peaks = tm.match(feature, frame, roi=port)

                    if scale:
                        for loc, corr in peaks:
                            if corr > 0.3:
                                digit_locations.append([digit, loc, corr])

                digit_locations = tm.get_clusters(digit_locations,
                                                  key=lambda x: x[1])
                digits = [max(cluster, key=lambda x: x[2])
                          for cluster in digit_locations]
                digits = [str(digit) for digit, location, corr
                          in sorted(digits, key=lambda x: x[1][1])]
                percents.append(''.join(digits))
        print("\t".join(str(x) for x in percents), flush=True)
