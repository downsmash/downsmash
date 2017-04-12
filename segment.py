#!/usr/bin/python

import argparse
import cv2
import numpy as np
from itertools import groupby
from sklearn.cluster import KMeans
# import json
import logging
from random import random

logging.basicConfig(format="%(message)s")


def random_frames(filename, num_frames=50):
    vc = cv2.VideoCapture(filename)
    total_frames = vc.get(7)

    for _ in range(num_frames):
        frame_pos = int(random() * total_frames)
        vc.set(1, frame_pos)
        success, frame = vc.read()

        if success:
            yield (frame_pos, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return


def spaced_frames(filename, interval=None, num_frames=None):
    if (interval is None and num_frames is None) or None not in (interval, num_frames):
        raise ValueError('exactly one of (interval, num_frames) must be set')

    vc = cv2.VideoCapture(filename)
    total_frames = int(vc.get(7))

    if not interval:
        interval = total_frames // num_frames

    for frame_pos in range(0, total_frames, interval):
        vc.set(1, frame_pos)
        success, frame = vc.read()

        if success:
            yield (frame_pos, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return


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


def multiple_template_match(feature, source, roi=None, max_clusters=None, num_frames=50, min_scale=0.5, max_scale=1.0, tolerance=0.05):
    peaks = []
    best_scale_log = []

    for (n, scene) in spaced_frames(source, num_frames=num_frames):
        cv2.imwrite("scene.png", scene)

        scene = cv2.imread("scene.png")
        if roi is not None:
            scene = scene[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]

        best_scale = find_best_scale(feature, scene, min_scale=min_scale, max_scale=max_scale)
        if best_scale:
            best_scale_log += [best_scale]
            scaled_feature = cv2.resize(feature, (0, 0), fx=best_scale, fy=best_scale)

            # Threshold for peaks.
            # Explain what is going on here.
            corr_map = cv2.matchTemplate(scene, scaled_feature, cv2.TM_CCOEFF_NORMED)

            _, max_corr, _, max_loc = cv2.minMaxLoc(corr_map)

            good_points = zip(*np.where(corr_map >= max_corr - tolerance))
            clusters = get_clusters(good_points)

            these_peaks = [max(cluster, key=lambda pt: corr_map[pt]) for cluster in clusters]

            if max_clusters:
                peaks.extend(list(sorted(these_peaks, key=lambda pt: corr_map[
                             pt], reverse=True))[:max_clusters])
            else:
                peaks.extend(these_peaks)

    feature_locations = [max(set(cluster), key=cluster.count) for cluster in get_clusters(peaks)]
    feature_locations = sorted(feature_locations, key=lambda pt: pt[1])

    if roi is not None:
        feature_locations = [(roi[0][0] + loc[0], roi[1][0] + loc[1]) for loc in feature_locations]

    mean_best_scale = sum(best_scale_log) / len(best_scale_log) if best_scale_log else None

    return (mean_best_scale, feature_locations)


def get_stream_geometry(stream, polling_interval=2):
    percent = cv2.imread("assets/pct.png")
    mean_best_scale, percent_locations = multiple_template_match(percent, stream, max_clusters=2)

    # Estimate bounding box from percent and clock locations
    est_height, est_width = [x * mean_best_scale / 0.05835 for x in percent.shape[:2]]
    est_top = np.mean(percent_locations, axis=0)[0] - .871 * est_height

    clock_digit_locations = []
    for digit in [2, 3, 4, 5, 6, 8]:
        feature = cv2.imread("assets/{0}_time.png".format(digit))
        mean_best_scale, feature_locations = multiple_template_match(
            feature, stream, num_frames=10, min_scale=0.8)

        clock_digit_locations.extend(feature_locations)

    est_clock_center = np.mean(clock_digit_locations, axis=0)[1]
    est_left = est_clock_center - .475 * est_width

    return ((est_top, est_left), (est_top + est_height, est_left + est_width))


def get_ports(stream, polling_interval=2, max_error=.1):
    (est_top, est_left), (est_bottom, est_right) = get_stream_geometry(
        stream, polling_interval=polling_interval)
    est_height, est_width = est_bottom - est_top, est_right - est_left

    est_percent_locations = [np.array((est_top + .87 * est_height,
                                       est_left + (.2 + .2381 * n) * est_width)) for n in range(4)]
    percent_rois = [np.array(((int(loc[0] - max_error * est_height), int(loc[0] + (.06 + max_error) * est_height)),
                              (int(loc[1] - max_error * est_width),  int(loc[1] + (.06 + max_error) * est_width)))) for loc in est_percent_locations]

    percent_locations = []

    percent = cv2.imread("assets/pct.png")
    for port_number, roi in enumerate(percent_rois):
        scale, location = multiple_template_match(percent, stream, max_clusters=1, roi=roi)
        if scale:
            location = location[0]
            percent_locations.append((port_number, location))
            error = est_percent_locations[port_number] - location
            logging.warn("Detected port {0} at location {1} (absolute error {2[0]:.2f}px, {2[1]:.2f}px)".format(
                port_number + 1, location, error))


def get_match_timestamps(stream, polling_interval=2, min_gap=10, max_error=.1):
    (est_top, est_left), (est_bottom, est_right) = get_stream_geometry(
        stream, polling_interval=polling_interval)
    est_height, est_width = est_bottom - est_top, est_right - est_left
    est_scale = np.mean([est_height / 411, est_width / 548])

    # print(est_top, est_left, est_height, est_width)

    est_percent_locations = [np.array((est_top + .87 * est_height,
                                       est_left + (.2 + .2381 * n) * est_width)) for n in range(4)]
    percent_rois = [np.array(((int(loc[0] - max_error * est_height), int(loc[0] + (.06 + max_error) * est_height)),
                              (int(loc[1] - max_error * est_width),  int(loc[1] + (.06 + max_error) * est_width)))) for loc in est_percent_locations]

    percent_locations = []

    percent = cv2.imread("assets/pct.png")
    for port_number, roi in enumerate(percent_rois):
        scale, location = multiple_template_match(percent, stream, max_clusters=1, roi=roi)
        if scale:
            location = location[0]
            percent_locations.append((port_number, location))
            error = est_percent_locations[port_number] - location
            logging.warn("Detected port {0} at location {1} (absolute error {2[0]:.2f}px, {2[1]:.2f}px)".format(
                port_number + 1, location, error))

    # Do percent sign template matching and tabulate maximum correlation coefficients
    corr_series = []
    for (n, scene) in spaced_frames(stream, interval=int(polling_interval * 30)):
        cv2.imwrite("scene.png", scene)
        scene = cv2.imread("scene.png")

        scaled_feature = cv2.resize(percent, (0, 0), fx=est_scale, fy=est_scale)
        corr_map = cv2.matchTemplate(scene, scaled_feature, cv2.TM_CCOEFF_NORMED)
        corr_series += [[n / 30, max(corr_map[loc] for port, loc in percent_locations)]]

    # Cluster the correlation values into probable games and probable gaps using k-means with k=2
    kmeans = KMeans(n_clusters=2).fit(np.array([corr for time, corr in corr_series]).reshape(-1, 1))
    threshold = sum(kmeans.cluster_centers_)[0] / 2
    logging.warn("Threshold is {:.4f}".format(threshold))

    # with open('out.tsv', 'w+') as f:
    #     f.write("\n".join("\t".join(str(x) for x in corr) for corr in corr_series))

    def moving_average(series, n=5):
        return np.convolve(series, np.ones((n,)) / n, mode='valid')

    averages = zip([x[0] + min_gap // 2 for x in corr_series], moving_average([x[1]
                                                                               for x in corr_series], n=int(min_gap // polling_interval)))
    groups = [(k, list(v)) for k, v in groupby(averages, lambda pt: pt[1] >= threshold * 1)]
    games = [[v[0], v[-1]] for k, v in groups if k]

    return games


def __main__():
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument("input", help="the JSON file (filename/players)", type=str)
    parser.add_argument("file", help="stream", type=str)

    args = parser.parse_args()

    # with open(args.input) as f:
    #    data = json.load(f)
    stream = args.file

    for stamp in get_match_timestamps(stream):
        print(stamp)


if __name__ == "__main__":
    __main__()
