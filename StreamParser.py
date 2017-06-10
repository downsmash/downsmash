#!/usr/bin/python

import cv2
import numpy as np
from itertools import groupby
from sklearn.cluster import DBSCAN
from random import randint
import pandas as pd
import logging

logging.basicConfig(format="%(message)s")


class ROI:

    def __init__(self, top, left, height, width):
        self.top, self.left, self.height, self.width = [int(x) for x in (top, left, height, width)]

    def __str__(self):
        return '{height}x{width}+{top}+{left}'.format(**self.__dict__)

    def __repr__(self):
        return '{height}x{width}+{top}+{left}'.format(**self.__dict__)

    def __and__(self, other):
        overlap_left, overlap_top = max(self.left, other.left), max(self.top, other.top)
        overlap_right = min(self.left + self.width, other.left + other.width)
        overlap_bottom = min(self.top + self.height, other.top + other.height)
        if overlap_left > overlap_right or overlap_top > overlap_bottom:
            return None
        else:
            return ROI(overlap_top, overlap_left, overlap_bottom - overlap_top, overlap_right - overlap_left)

    def __getitem__(self, given):
        if isinstance(given, slice):
            if given.step:
                raise ValueError('ROI getitem steps are not supported')
            top, left = given.start
            bottom, right = given.end
            return ROI(top * self.height, left * self.width, (bottom - top) * self.height, (right - left) * self.width)
        else:
            y, x = given
            return np.array((int(self.top + y * self.height),
                             int(self.left + x * self.width)))


class StreamParser:

    def __init__(self, filename):
        self.filename = filename
        self.vc = cv2.VideoCapture(filename)
        self.roi = ROI(0, 0, self.vc.get(4), self.vc.get(3))

    def parse(self):
        raise NotImplementedError

    def sample_frames(self, start=None, end=None, interval=None, num_samples=None, fuzz=0):
        if (interval is None and num_samples is None) or None not in (interval, num_samples):
            raise ValueError('exactly one of (interval, num_samples) must be set')

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

    def get_clusters(self, pts, max_distance=14, key=lambda x: x):
        clusters = []
        for pt in pts:
            for idx, cluster in enumerate(clusters):
                if min(np.linalg.norm(np.subtract(key(pt), key(x))) for x in cluster) < max_distance:
                    clusters[idx] += [pt]
                    break
            else:
                clusters.append([pt])

        return clusters

    def find_best_scale(self, feature, scene, min_scale=0.5, max_scale=1.0, scale_delta=0.03, min_corr=0.8):
        best_corr = 0
        best_scale = 0

        for scale in np.arange(min_scale, max_scale + scale_delta, scale_delta):
            scaled_feature = cv2.resize(feature, (0, 0), fx=scale, fy=scale)

            result = cv2.matchTemplate(scene, scaled_feature, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_corr:
                best_corr = max_val
                best_scale = scale

        if best_corr > min_corr:
            return best_scale
        else:
            return None

    def locate(self, feature, roi=None, max_clusters=None, N=10, min_scale=0.5, max_scale=1.0, debug=False):
        peaks = []
        best_scale_log = []

        for (n, scene) in self.sample_frames(num_samples=N):
            cv2.imwrite("scene.png", scene)
            scene = cv2.imread("scene.png")

            scale, these_peaks = self.multiple_template_match(feature, scene, roi=roi, min_scale=min_scale, max_scale=max_scale, debug=debug)
            if debug:
                logging.warn("{0} {1}".format(scale, these_peaks))

            if scale:
                best_scale_log += [scale]

                these_peaks = [loc for loc, corr in sorted(these_peaks, key=lambda pt: pt[1])]
                if max_clusters:
                    peaks.extend(these_peaks[:max_clusters])
                else:
                    peaks.extend(these_peaks)

        feature_locations = [np.array(max(set(cluster), key=cluster.count))for cluster in self.get_clusters(peaks)]
        feature_locations = sorted(feature_locations, key=lambda pt: pt[1])

        if roi is not None:
            feature_locations = [np.array((roi.top, roi.left)) + loc for loc in feature_locations]

        mean_best_scale = sum(best_scale_log) / len(best_scale_log) if best_scale_log else None

        return (mean_best_scale, feature_locations)

    def multiple_template_match(self, feature, scene, roi=None, scale=None, min_scale=0.5, max_scale=1.0, max_distance=14, min_corr=0.8, debug=False, threshold_min=50, threshold_max=200):
        if roi is not None:
            scene = scene[roi.top:(roi.top + roi.height), roi.left:(roi.left + roi.width)]

        if not scale:
            scale = self.find_best_scale(feature, scene, min_scale=min_scale, max_scale=max_scale, min_corr=min_corr)
        peaks = []

        if scale:
            scaled_feature = cv2.resize(feature, (0, 0), fx=scale, fy=scale)

            canny_scene = cv2.Canny(scene, threshold_min, threshold_max)
            canny_feature = cv2.Canny(scaled_feature, threshold_min, threshold_max)

            # Threshold for peaks.
            corr_map = cv2.matchTemplate(canny_scene, canny_feature, cv2.TM_CCOEFF_NORMED)
            _, max_corr, _, max_loc = cv2.minMaxLoc(corr_map)

            good_points = list(zip(*np.where(corr_map >= max_corr - self.tolerance)))
            if debug:
                print(max_corr, good_points)
            clusters = self.get_clusters(good_points, max_distance=max_distance)
            peaks = [max([(pt, corr_map[pt]) for pt in cluster], key=lambda pt: pt[1]) for cluster in clusters]

        return (scale, peaks)


class MeleeVODParser(StreamParser):

    def __init__(self, filename, polling_interval=2, tolerance=0.05, min_gap=10):
        StreamParser.__init__(self, filename)
        self.polling_interval = polling_interval
        self.tolerance = tolerance
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
        scale, percent_locations = self.locate(percent, N=30, max_clusters=2)
        print(percent_locations)

        clock_digit_locations = []
        for digit in [2, 3, 4, 5, 6, 8]:
            feature = cv2.imread("assets/{0}_time.png".format(digit))
            _, digit_locations = self.locate(feature, N=20, min_scale=0.8)

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
            (percent_top, percent_left) = self.screen[.87, .2 + .2381 * port_number]
            percent_roi_top = percent_top - max_error * self.screen.height
            percent_roi_left = percent_left - max_error * self.screen.width
            percent_roi_height = (.06 + 2 * max_error) * self.screen.height
            percent_roi_width = (.06 + 2 * max_error) * self.screen.width
            percent_roi = ROI(percent_roi_top, percent_roi_left, percent_roi_height, percent_roi_width) & self.screen

            scale, location = self.locate(percent, max_clusters=1, N=10, roi=percent_roi, debug=True)
            if scale:
                error = location[0] - (percent_top, percent_left)
                logging.warn("Detected port {0} at {1} (error {2[0]}px, {2[1]}px)".format(
                    port_number + 1, location[0], error))

                (port_top, port_left) = self.screen[.748, .0363 + .24 * port_number]
                port_roi_top = port_top - max_error * self.screen.height
                port_roi_left = port_left - max_error * self.screen.width
                port_roi_height = (.18 + 2 * max_error) * self.screen.height
                port_roi_width = (.1833 + 2 * max_error) * self.screen.width
                port_roi = ROI(port_roi_top, port_roi_left, port_roi_height, port_roi_width) & self.screen
                ports.append(port_roi)
            else:
                ports.append(None)

        return ports

    def detect_match_chunks(self, max_error=.06):
        percent = cv2.imread("assets/pct.png")
        corr_series = []

        for (time, scene) in self.sample_frames(interval=self.polling_interval):
            cv2.imwrite("scene.png", scene)
            scene = cv2.imread("scene.png")

            scaled_percent = cv2.resize(
                percent, (0, 0), fx=self.scale, fy=self.scale)
            scaled_percent = cv2.Canny(scaled_percent, 50, 200)

            percent_corrs = []
            for port_number, roi in enumerate(self.ports):
                if roi is not None:
                    scene_roi = scene[roi.top:(roi.top + roi.height), roi.left:(roi.left + roi.width)]
                    scene_roi = cv2.Canny(scene_roi, 50, 200)

                    corr_map = cv2.matchTemplate(scene_roi, scaled_percent, cv2.TM_CCOEFF_NORMED)
                    _, max_corr, _, max_loc = cv2.minMaxLoc(corr_map)
                    percent_corrs.append(max_corr)

            point = [time, max(percent_corrs)]
            corr_series.append(point)

        corr_series = np.array(corr_series)

        medians = pd.rolling_median(corr_series[:, 1], self.min_gap //
                                    self.polling_interval, center=True)[2:-2]

        clusters = DBSCAN(eps=0.03, min_samples=10).fit(medians.reshape(-1, 1))

        dataframe = list(zip(corr_series[:, 0][2:-2], medians, clusters.labels_))

        labels = list(set(x[2] for x in dataframe))
        cluster_means = [sum(cluster) / len(cluster) for cluster in [[x[1] for x in dataframe if x[2] == label] for label in labels]]
        cluster_means = list(zip(labels, cluster_means))

        game_label = max(cluster_means, key=lambda x: x[1])[0]
        game_groups = [(k, list(v)) for k, v in groupby(dataframe, lambda pt: pt[2])]
        games = [[v[0][0], v[-1][0]] for k, v in game_groups if k == game_label]

        return games

    def get_percents(self, chunk, interval=0.2):
        start, end = chunk
        for time, frame in self.sample_frames(start=start, end=end, interval=interval):
            percents = [time]
            for idx, port in enumerate(self.ports):
                if port is not None:
                    # port_roi = port[.5:1]
                    digit_locations = []
                    for digit in range(0, 10):
                        if digit == 1:
                            feature = cv2.imread("assets/{0}_pct_cropped.png".format(digit), 0)
                        else:
                            feature = cv2.imread("assets/{0}_pct.png".format(digit), 0)

                        scale, peaks = self.multiple_template_match(feature, frame, roi=port, min_scale=self.scale - 0.1, max_scale=self.scale + 0.05, threshold_min=175, threshold_max=350)
                        if scale:
                            for loc, corr in peaks:
                                if corr > 0.3:
                                    digit_locations.append([digit, loc, corr])

                    digit_locations = self.get_clusters(digit_locations, key=lambda x: x[1])
                    digits = [max(cluster, key=lambda x: x[2]) for cluster in digit_locations]
                    digits = list(str(digit) for digit, location, corr in sorted(digits, key=lambda x: x[1][1]))
                    percents.append(''.join(digits))
            print("\t".join(str(x) for x in percents), flush=True)
