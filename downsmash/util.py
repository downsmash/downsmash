"""Various, mostly statistical, utility functions.
"""

from itertools import groupby

import cv2
import numpy as np
import scipy.stats
from scipy.signal import argrelmin
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity


def timeify(time):
    """Format a time in seconds to a minutes/seconds timestamp."""
    time = float(time)
    mins, secs = time // 60, time % 60
    return f"{mins:.0f}:{secs:05.2f}"


def get_clusters(pts, key=lambda x: x, max_clusters=None, max_distance=14):
    """Run DBSCAN on the `pts`, applying `key` first if necessary,
    post-process the results into a list of lists, and return it,
    taking only the largest `max_clusters` clusters.
    """
    if pts:
        kpts = [key(pt) for pt in pts]

        clustering = DBSCAN(eps=max_distance, min_samples=1).fit(kpts)

        # Post-processing.
        labeled_pts = list(zip(kpts, clustering.labels_))
        labeled_pts = sorted(labeled_pts, key=lambda p: p[1])

        clusters = [
            list(g) for _, g in groupby(labeled_pts, key=lambda p: p[1])
        ]
        clusters = [[p[0] for p in clust] for clust in clusters]
        clusters = list(sorted(clusters, key=len, reverse=True))

        return clusters[:max_clusters]
    return []


def compute_minimum_kernel_density(series):
    """Estimate the value within the range of _series_ that is the furthest
    away from most observations.
    """
    # Trim outliers for robustness.
    p05, p95 = series.quantile((0.05, 0.95))
    samples = np.linspace(p05, p95, num=100)

    # Find the minimum kernel density.
    kde = KernelDensity(kernel="gaussian", bandwidth=0.005)
    kde = kde.fit(np.array(series).reshape(-1, 1))
    estimates = kde.score_samples(samples.reshape(-1, 1))

    rel_mins = argrelmin(estimates)[0]

    def depth(idx):
        return min(
            estimates[idx - 1] - estimates[idx],
            estimates[idx + 1] - estimates[idx],
        )

    deepest_min = max(rel_mins, key=depth)

    return samples[deepest_min]


def scale_to_interval(array, new_min, new_max):
    """Scale the elements of _array_ linearly to lie between
    _new_min_ and _new_max_.
    """
    array_min = min(array.flatten())
    array_max = max(array.flatten())

    # array_01 is scaled between 0 and 1.
    if array_min == array_max:
        array_01 = np.zeros(array.shape)
    else:
        array_01 = (array - array_min) / (array_max - array_min)

    return new_min + (new_max - new_min) * array_01


def overlay_map(frames):
    """Run a skewness-kurtosis filter on a sample of frames and
    edge-detect.

    The areas of the video containing game feed should come back black.
    Areas containing overlay or letterboxes will be visibly white.
    """

    skew_map = scipy.stats.skew(frames, axis=0)
    kurt_map = scipy.stats.kurtosis(frames, axis=0)
    min_map = np.minimum(
        skew_map, kurt_map
    )  # pylint:disable=assignment-from-no-return

    min_map = scale_to_interval(min_map, 0, 255).astype(np.uint8)

    # Blur and edge detect.
    min_map = cv2.blur(min_map, (5, 5))
    edges = cv2.Laplacian(min_map, cv2.CV_8U)

    # Areas that are constant throughout the video (letterboxes) will
    # have 0 skew, 0 kurt, and 0 variance, so the skew-kurt filter
    # will miss them
    sd_map = np.sqrt(np.var(frames, axis=0))
    edges[np.where(sd_map < 0.01)] = 255
    _, edges = cv2.threshold(edges, 7, 255, cv2.THRESH_BINARY)

    return edges


def find_dlt(predicted, locations):
    """Determine the direct linear transformation that moves the percent signs
    to where they should be using OLS (ordinary least squares.)

    Specifically, compute the OLS solution of the following system:
    port_0_x_predicted * scale + shift_x = port_0_x_actual
    port_0_y_predicted * scale + shift_y = port_0_y_actual
    ...
    port_4_x_predicted * scale + shift_x = port_4_x_actual
    port_4_y_predicted * scale + shift_y = port_4_y_actual

    In matrix form Ax = b :
    [ p0x_pred 1 0 ] [ scale   ] = [ p0x_actual ]
    [ p0y_pred 0 1 ] [ shift_x ]   [ p0y_actual ]
    [ ...          ] [ shift_y ]   [ ...]
    [ p4x_pred 1 0 ]               [ p4x_actual ]
    [ p4y_pred 0 1 ]               [ p4x_actual ]
    """

    predicted_mat = []
    for (predicted_y, predicted_x) in predicted:
        predicted_mat.append([predicted_y, 0, 1])
        predicted_mat.append([predicted_x, 1, 0])

    actual_vec = []
    for (actual_y, actual_x) in locations:
        actual_vec.append(actual_y)
        actual_vec.append(actual_x)
    actual_vec = np.array(actual_vec).transpose()

    # TODO Check this thing's robustness
    ols, resid, _, _ = np.linalg.lstsq(predicted_mat, actual_vec, rcond=None)

    scale_factor, shift_x, shift_y = ols
    return (scale_factor, shift_x, shift_y)


def bisect(f, start, end, tolerance):
    # First make sure we have an interval to which bisection is applicable
    # (that is, one on which f(t) changes sign.)
    # Also compute start and end confs.
    start_value = f(start)
    end_value = f(end)

    plus_to_minus = start_value > 0 > end_value
    minus_to_plus = start_value < 0 < end_value

    if not (minus_to_plus or plus_to_minus):
        raise ValueError(f"bisect() got a bad interval [{start}, {end}]")

    while end - start > tolerance:
        middle = (start + end) / 2
        middle_value = f(middle)
        if (0 > middle_value and 0 > start_value) or (
            0 < middle_value and 0 < start_value
        ):
            start = middle
        else:
            end = middle

    return (start + end) / 2
