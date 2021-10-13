"""Various, mostly statistical, utility functions.
"""

from itertools import groupby

import numpy as np
from scipy.signal import argrelmin
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity


def timeify(time):
    """Format a time in seconds to a minutes/seconds timestamp.
    """
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

        clusters = [list(g) for _, g in groupby(labeled_pts, key=lambda p: p[1])]
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
    kde = KernelDensity(kernel='gaussian', bandwidth=.005)
    kde = kde.fit(np.array(series).reshape(-1, 1))
    estimates = kde.score_samples(samples.reshape(-1, 1))

    rel_mins = argrelmin(estimates)[0]

    def depth(idx):
        return min(estimates[idx - 1] - estimates[idx],
                   estimates[idx + 1] - estimates[idx])
    deepest_min = max(rel_mins, key=depth)

    return samples[deepest_min]
