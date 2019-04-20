#!/usr/bin/python

import logging
import itertools
from dataclasses import dataclass

import numpy as np
import cv2
import pandas as pd
from sklearn.neighbors.kde import KernelDensity
from scipy.signal import argrelmin

from pkg_resources import resource_string

from .rect import Rect
from .stream_parser import StreamParser
from .viewfinder import Viewfinder

# Read in percent sign
NPARR = np.fromstring(resource_string("core.resources", "pct.png"), np.uint8)
PERCENT = cv2.imdecode(NPARR, 1)

LOGGER = logging.getLogger(__name__)


@dataclass
class MatchData:
    """Wrapper class for parsed match data.
    """
    screen: Rect
    scale: float
    ports: list
    matches: list
    threshold: float

class MatchDataBuilder(StreamParser):

    def __init__(self, filename, polling_interval=2, min_gap=10):
        StreamParser.__init__(self, filename)

        # These are segmentation strategy parameters
        self.polling_interval = polling_interval
        self.min_gap = min_gap

        # Computed during segmentation
        self.threshold = None

    def parse(self):
        """
        """
        match = MatchData()

        view = Viewfinder(self.filename, polling_interval=self.polling_interval)

        LOGGER.warning("Video OK, looking for screen...")

        scale, screen = view.detect_screen()
        LOGGER.warning("Estimated screen is %s", screen)
        LOGGER.warning("Estimated scale is %.03f", scale)

        LOGGER.warning("Correcting screen...")
        scale, screen = view.correct_screen(scale, screen)
        LOGGER.warning("Estimated screen is %s", screen)
        LOGGER.warning("Estimated scale is %.03f", scale)

        # TODO There needs to be some kind of robustness check on these to
        # make sure no ill-conditioning weirdness happens with OLS.
        match.scale = scale
        match.screen = screen

        LOGGER.warning("Checking port locations...")
        ports, _, _ = view.detect_ports(scale, screen)
        if not ports:
            raise RuntimeError("No ports found!")
        LOGGER.warning("Ports are at %s", " ".join(str(s) for s in ports))

        match.ports = ports

        LOGGER.warning("Beginning segmentation...")
        segments = self.segment(match)

        def timeify(time):
            time = float(time)
            mins, secs = time // 60, time % 60
            return "{:.0f}:{:05.2f}".format(mins, secs)

        for idx, segment in enumerate(segments):
            start, end = segment
            LOGGER.warning("Estimated game %d is %s-%s", idx + 1, timeify(start), timeify(end))

        LOGGER.warning("Refining segment boundaries...")
        for idx, segment in enumerate(segments):
            start, end = match
            start = self.find_segment_boundary(segment, start)
            end = self.find_segment_boundary(segment, end)
            segments[idx] = (start, end)
            LOGGER.warning("Estimated game %d is %s-%s", idx + 1, timeify(start), timeify(end))

        match.segments = matches

        return match

    @staticmethod
    def calculate_frame_confidence(match, scene):
        """Estimate the maximum correlation of any percent ROI in __scene__
        to the PERCENT image.
        """
        cv2.imwrite("scene.png", scene)
        scene = cv2.imread("scene.png")

        scaled_percent = cv2.resize(PERCENT, (0, 0), fx=match.scale, fy=match.scale)
        scaled_percent = cv2.Laplacian(scaled_percent, cv2.CV_8U)

        percent_corrs = []
        for roi in match.ports:
            if roi is not None:
                scene_roi = scene[roi.top:(roi.top + roi.height), roi.left:(roi.left + roi.width)]
                scene_roi = cv2.Laplacian(scene_roi, cv2.CV_8U)

                corr_map = cv2.matchTemplate(scene_roi, scaled_percent, cv2.TM_CCOEFF_NORMED)
                _, max_corr, _, _ = cv2.minMaxLoc(corr_map)
                percent_corrs.append(max_corr)
        return max(percent_corrs)

    @staticmethod
    def compute_minimum_kernel_density(series):
        """Estimate the value within the range of _series_ that is the furthest
        away from most observations.
        """
        # Trim outliers for robustness.
        p05, p95 = series.quantile((0.05, 0.95))
        samples = np.linspace(p05, p95, num=100)

        # Find the minimum kernel density.
        kde = KernelDensity(kernel='gaussian', bandwidth=.005)
        kde = kde.fit(np.array(series).reshape(-1, 1))
        estimates = kde.score_samples(samples.reshape(-1, 1))

        rel_mins = argrelmin(estimates)[0]
        def depth(idx):
            return min(estimates[idx - 1] - estimates[idx],
                       estimates[idx + 1] - estimates[idx])
        deepest_min = max(rel_mins, key=depth)
        return samples[deepest_min]

    def segment(self, match):
        """Return the approximate match start and end times for
        the given video.
        """
        conf_series = []

        for (time, scene) in self.sample_frames(interval=self.polling_interval):
            conf = self.calculate_frame_confidence(match, scene)

            LOGGER.info("%d\t%f", time, conf)
            conf_series.append([time, conf])

        conf_series = pd.DataFrame(conf_series, columns=['time', 'conf'])
        confs = conf_series['conf']

        match.threshold = self.compute_minimum_kernel_density(confs)

        # How separated are the two groups?
        mean_positive = np.mean(confs[confs >= match.threshold])
        mean_negative = np.mean(confs[confs < match.threshold])
        LOGGER.warning("Group means are (+)%.03f (-)%.03f", mean_positive, mean_negative)

        # TODO Replace magic numbers
        # TODO This error message needs to be more descriptive - something about
        # false negatives
        if mean_positive - mean_negative < 0.1 or mean_negative > 0.5:
            LOGGER.warning("This looks like an edited/gapless set"
                           "(mean_pos - mean_neg = %.03f)", mean_positive - mean_negative)
            raise RuntimeError()

        # Perform median smoothing.
        conf_series['median'] = conf_series['conf'].rolling(5).median()
        conf_series['median'] = conf_series['median'].fillna(method='bfill')
        conf_series['median'] = conf_series['median'].fillna(method='ffill')

        # Now classify as Melee/no Melee based on whether we are greater/less
        # than the threshold.
        groups = itertools.groupby(conf_series.iterrows(),
                                   lambda row: row[1]['median'] > match.threshold)
        groups = [(k, list(g)) for k, g in groups]
        segments = [(self.polling_interval * g[0][0],
                     self.polling_interval * g[-1][0]) for k, g in groups if k]

        return segments

    def find_segment_boundary(self, match, time, tolerance=0.1):
        """Find the time index of a match segment boundary (start or end)
        near _time_, accurate to within _tolerance_ seconds.
        Uses the bisection method to find an approximate solution for
        f(t) = conf_at(t) - self.threshold = 0.
        """
        def conf_at(time):
            scene = self.get_frame(time)
            if scene is not None:
                return self.calculate_frame_confidence(match, scene)
            return None

        # TODO Magic number
        window = self.min_gap * 1.5

        start = max(0, time - window / 2)
        # Have to read from strictly before the end of the video.
        # length() - 1, etc.
        end = min(self.length - tolerance, time + window / 2)

        # First make sure we have an interval to which bisection is applicable
        # (that is, one on which f(t) changes sign.)
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

            plus_to_minus = (start_conf > match.threshold > end_conf)
            minus_to_plus = (start_conf < match.threshold < end_conf)

        while end - start > tolerance:
            middle = (start + end) / 2
            middle_conf = conf_at(middle)
            if (match.threshold > middle_conf and match.threshold > start_conf
                    or match.threshold < middle_conf and match.threshold < start_conf):
                start = middle
            else:
                end = middle

        return (start + end) / 2
