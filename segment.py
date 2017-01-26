#!/usr/bin/python

import argparse
import cv2
import numpy as np
from sklearn.cluster import KMeans
import json
from random import random


def random_frames(filename, mean_frames=50):
    vc = cv2.VideoCapture(filename)

    frames = 0
    total_frames = vc.get(7)

    while True:
        success, frame = vc.read()

        if success:
            frames += 1
            if random() < mean_frames / total_frames:
                yield (frames, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            return


def spaced_frames(filename, interval=150):
    vc = cv2.VideoCapture(filename)
    frames = 0

    while True:
        success, frame = vc.read()

        if success:
            frames += 1
            if frames % interval == 0:
                yield (frames, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            return


def find_best_scale(feature, scene, min_scale=0.5, max_scale=1.0, scale_delta=0.02, min_corr=0.8):
    best_corr = 0
    best_scale = 0

    scale = min_scale
    while scale <= max_scale:
        scaled_feature = cv2.resize(feature, (0, 0), fx=scale, fy=scale)

        result = cv2.matchTemplate(scene, scaled_feature, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_corr:
            best_corr = max_val
            best_scale = scale

        scale += scale_delta

    if best_corr > min_corr:
        return best_scale
    else:
        return None


def get_clusters(pts, max_distance=10):
    clusters = []
    for pt in pts:
        for idx, cluster in enumerate(clusters):
            if min(np.linalg.norm(np.subtract(pt, x)) for x in cluster) < max_distance:
                clusters[idx] += [pt]
                break
        else:
            clusters.append([pt])

    return clusters


def feature_threshold_peaks(feature, scene, tolerance=0.05, max_clusters=None):
    corr_map = cv2.matchTemplate(scene, feature, cv2.TM_CCOEFF_NORMED)
    _, max_corr, _, max_loc = cv2.minMaxLoc(corr_map)

    good_points = zip(*np.where(corr_map >= max_corr - tolerance))
    clusters = get_clusters(good_points)

    peaks = [max(cluster, key=lambda pt: corr_map[pt]) for cluster in clusters]

    if max_clusters:
        return list(sorted(peaks, key=lambda pt: corr_map[pt], reverse=True))[:max_clusters]
    else:
        return peaks


def multiple_template_match(feature, source, max_clusters=None, mean_frames=50):
    peaks = []
    best_scale_log = []

    for (n, scene) in random_frames(source, mean_frames=mean_frames):
        cv2.imwrite("scene.png", scene)
        scene = cv2.imread("scene.png")

        best_scale = find_best_scale(feature, scene)
        if best_scale:
            best_scale_log += [best_scale]
            scaled_feature = cv2.resize(feature, (0, 0), fx=best_scale, fy=best_scale)
            peaks.extend(feature_threshold_peaks(scaled_feature, scene, max_clusters=max_clusters))

    feature_locations = [max(set(cluster), key=cluster.count) for cluster in get_clusters(peaks)]
    feature_locations = sorted(feature_locations, key=lambda pt: pt[1])

    mean_best_scale = sum(best_scale_log) / len(best_scale_log) if best_scale_log else None

    return (mean_best_scale, feature_locations)


def __main__():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("input", help="the JSON file (filename/players)", type=str)

    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    source = data['filename']

    slots = sorted([int(port) - 1 for port in data['players'].keys()])
    icons = [data['players'][str(slot + 1)]['icon'] for slot in slots]

    percent = cv2.imread("assets/pct.png")
    mean_best_scale, percent_locations = multiple_template_match(percent, source, max_clusters=2)

    for digit in [0, 2, 4, 6, 8]:
        feature = cv2.imread("assets/{0}_time.png".format(digit))
        mean_best_scale, feature_locations = multiple_template_match(feature, source, mean_frames=10)
        print(digit, feature_locations)

    # Estimate bounding box from percent locations
    for (slot, pos) in zip(slots, percent_locations):
        estimated_size = [x * mean_best_scale / 0.05835 for x in percent.shape[:2]]
        estimated_top = pos[0] - .871 * estimated_size[0]
        estimated_left = pos[1] - (.202 + .2381 * slot) * estimated_size[1]
        print(pos)
        print(estimated_size, estimated_top, estimated_left)

    # Do percent sign template matching and tabulate maximum correlation coefficients
    corr_series = []
    for (n, scene) in spaced_frames(source, interval=60):
        cv2.imwrite("scene.png", scene)
        scene = cv2.imread("scene.png")

        scaled_feature = cv2.resize(percent, (0, 0), fx=mean_best_scale, fy=mean_best_scale)
        corr_map = cv2.matchTemplate(scene, scaled_feature, cv2.TM_CCOEFF_NORMED)
        corr_series += [[n // 30, max(corr_map[loc] for loc in percent_locations)]]

    # Perform k-means clustering with k=2 to determine threshold value for games/gaps
    kmeans = KMeans(n_clusters=2).fit(np.array([corr for time, corr in corr_series]).reshape(-1, 1))
    print(kmeans.cluster_centers_)
    threshold = sum(kmeans.cluster_centers_)[0] / 2
    gamevalues = [time for time, corr in corr_series if corr >= threshold]

    gaps = [(this, next) for this, next in zip(gamevalues, gamevalues[1:]) if next > this + 10]
    print(gaps)


if __name__ == "__main__":
    __main__()
