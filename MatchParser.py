#!/usr/bin/python

import cv2
import numpy as np
from itertools import groupby
from sklearn.cluster import KMeans
from random import randint
import logging

logging.basicConfig(format="%(message)s")


class ROI:

    def __init__(self, top_left, bottom_right):
        self.top, self.left = top_left
        self.bottom, self.right = bottom_right

    def __str__(self):
        return "{0}x{1}+{2}+{3}".format(self.bottom - self.top, self.right - self.left, self.top, self.left)

    def __repr__(self):
        return "{0}x{1}+{2}+{3}".format(self.bottom - self.top, self.right - self.left, self.top, self.left)

    def affine_location(self, y, x):
        return np.array((int((1 - y) * self.top + y * self.bottom),
                         int((1 - x) * self.left + x * self.right)))


class MatchParser:

    def __init__(self, stream, polling_interval=2, tolerance=0.05, min_gap=10):
        self.stream = stream
        self.polling_interval = polling_interval
        self.tolerance = tolerance
        self.min_gap = min_gap

    def parse(self):
        self.geometry = self.__detect_geometry()
        self.ports = self.__detect_ports()
        self.chunks = self.__detect_match_chunks()

    def __detect_geometry(self):
        percent = cv2.imread("assets/pct.png")
        scale, percent_locations = multiple_template_match(
            self, percent, max_clusters=2)

        # Estimate bounding box from percent and clock locations
        height, width = [x * scale / 0.05835 for x in percent.shape[:2]]
        top = np.mean(percent_locations, axis=0)[0] - .871 * height

        clock_digit_locations = []
        for digit in [2, 3, 4, 5, 6, 8]:
            feature = cv2.imread("assets/{0}_time.png".format(digit))
            scale, feature_locations = multiple_template_match(
                self, feature, N=10, min_scale=0.8)

            clock_digit_locations.extend(feature_locations)

        clock_center = np.mean(clock_digit_locations, axis=0)[1]
        left = clock_center - .475 * width

        self.scale = np.mean([height / 411, width / 548])

        return ROI((top, left), (top + height, left + width))

    def __detect_ports(self, max_error=0.04):
        height, width = self.geometry.bottom - \
            self.geometry.top, self.geometry.right - self.geometry.left

        percent_locations = [
            self.geometry.affine_location(.87, .2 + .2381 * n) for n in range(4)]

        percent_rois = [ROI((int(loc[0] - max_error * height),          int(loc[1] - max_error * width)),
                            (int(loc[0] + (.06 + max_error) * height),  int(loc[1] + (.06 + max_error) * width))) for loc in percent_locations]

        ports = []

        percent = cv2.imread("assets/pct.png")
        for port_number, percent_roi in enumerate(percent_rois):
            scale, location = multiple_template_match(
                self, percent, max_clusters=1, roi=percent_roi)
            if scale:
                location = location[0]
                error = percent_locations[port_number] - location
                logging.warn("Detected port {0} at location {1} (absolute error {2[0]}px, {2[1]}px)".format(
                    port_number + 1, location, error))
                port_roi = ROI(self.geometry.affine_location(.748 - max_error, .0363 + .24 * port_number - max_error) - error,
                               self.geometry.affine_location(.928 + max_error, .2196 + .24 * port_number + max_error) - error)
                ports.append(port_roi)
            else:
                ports.append(None)

        return ports

    def __detect_match_chunks(self, max_error=.04):
        percent = cv2.imread("assets/pct.png")
        corr_series = []

        for (time, scene) in spaced_frames(self, interval=int(self.polling_interval * 30)):
            cv2.imwrite("scene.png", scene)
            scene = cv2.imread("scene.png")

            scaled_percent = cv2.resize(
                percent, (0, 0), fx=self.scale, fy=self.scale)
            scaled_percent = cv2.Canny(scaled_percent, 50, 200)

            percent_corrs = []
            for port_number, roi in enumerate(self.ports):
                if roi is not None:
                    scene_roi = scene[roi.top:roi.bottom, roi.left:roi.right]
                    scene_roi = cv2.Canny(scene_roi, 50, 200)

                    corr_map = cv2.matchTemplate(
                        scene_roi, scaled_percent, cv2.TM_CCOEFF_NORMED)
                    _, max_corr, _, max_loc = cv2.minMaxLoc(corr_map)
                    percent_corrs.append(max_corr)

            corr_series.append((round(time) / 1000, max(percent_corrs)))

        def moving_average(series, n=5):
            return np.convolve(series, np.ones((n,)) / n, mode='valid')

        averages = moving_average([x[1] for x in corr_series],
                                  n=int(self.min_gap // self.polling_interval))
        averages = zip(
            [x[0] + self.min_gap // 2 for x in corr_series], averages)

        kmeans = KMeans(n_clusters=4).fit(
            np.array([corr for time, corr in corr_series]).reshape(-1, 1))

        centers = kmeans.cluster_centers_.tolist()
        min_cluster_idx = centers.index(min(centers))
        points = zip([time for time, corr in corr_series], kmeans.labels_)

        # Throw out the lowest cluster
        groups = [(k, list(v)) for k, v in groupby(
            points, lambda pt: pt[1] != min_cluster_idx)]
        games = [[v[0][0], v[-1][0]] for k, v in groups if k]

        return games


def find_best_scale(feature, scene, min_scale=0.5, max_scale=1.0, scale_delta=0.02, min_corr=0.8):
    best_corr = 0
    best_scale = 0

    scale = min_scale
    for scale in np.arange(min_scale, max_scale + scale_delta, scale_delta):
        scaled_feature = cv2.resize(feature, (0, 0), fx=scale, fy=scale)

        result = cv2.matchTemplate(scene, scaled_feature, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_corr:
            best_corr = max_val
            best_scale = scale

    if best_corr > min_corr:
        return best_scale
    else:
        return None


def spaced_frames(parser, interval=None, num_frames=None, fuzz=4):
    if (interval is None and num_frames is None) or None not in (interval, num_frames):
        raise ValueError('exactly one of (interval, num_frames) must be set')

    vc = cv2.VideoCapture(parser.stream)
    total_frames = int(vc.get(7))

    if not interval:
        interval = total_frames // num_frames

    for frame_pos in range(0, total_frames, interval):
        frame_pos += randint(-1 * fuzz, fuzz)
        frame_pos = min([max([0, frame_pos]), total_frames])
        vc.set(1, frame_pos)
        success, frame = vc.read()

        if success:
            yield (vc.get(0), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return


def get_clusters(pts, max_distance=14):
    clusters = []
    for pt in pts:
        for idx, cluster in enumerate(clusters):
            if min(np.linalg.norm(np.subtract(pt, x)) for x in cluster) < max_distance:
                clusters[idx] += [pt]
                break
        else:
            clusters.append([pt])

    return clusters


def multiple_template_match(parser, feature, roi=None, max_clusters=None, N=50, min_scale=0.5, max_scale=1.0):
    peaks = []
    best_scale_log = []

    for (n, scene) in spaced_frames(parser, num_frames=N):
        cv2.imwrite("scene.png", scene)

        scene = cv2.imread("scene.png")
        if roi is not None:
            scene = scene[roi.top:roi.bottom, roi.left:roi.right]

        best_scale = find_best_scale(
            feature, scene, min_scale=min_scale, max_scale=max_scale)
        if best_scale:
            best_scale_log += [best_scale]
            scaled_feature = cv2.resize(
                feature, (0, 0), fx=best_scale, fy=best_scale)

            # Threshold for peaks.
            # TODO explain what is going on here.
            corr_map = cv2.matchTemplate(
                scene, scaled_feature, cv2.TM_CCOEFF_NORMED)

            _, max_corr, _, max_loc = cv2.minMaxLoc(corr_map)

            good_points = zip(
                *np.where(corr_map >= max_corr - parser.tolerance))

            clusters = get_clusters(good_points)

            these_peaks = [max(cluster, key=lambda pt: corr_map[pt])
                           for cluster in clusters]

            if max_clusters:
                peaks.extend(list(sorted(these_peaks, key=lambda pt: corr_map[
                             pt], reverse=True))[:max_clusters])
            else:
                peaks.extend(these_peaks)

    feature_locations = [np.array(max(set(cluster), key=cluster.count))
                         for cluster in get_clusters(peaks)]
    feature_locations = sorted(feature_locations, key=lambda pt: pt[1])

    if roi is not None:
        feature_locations = [
            np.array((roi.top, roi.left)) + loc for loc in feature_locations]

    mean_best_scale = sum(best_scale_log) / \
        len(best_scale_log) if best_scale_log else None

    return (mean_best_scale, feature_locations)
