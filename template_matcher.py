"""
"""

from itertools import groupby
import logging

import numpy as np
import cv2
from sklearn.cluster import DBSCAN

LOGGER = logging.getLogger(__name__)


class TemplateMatcher:
    """This class performs template matching on a StreamParser.
    """

    def __init__(self, scales=np.arange(0.5, 1.0, 0.03),
                 max_distance=14,
                 criterion=cv2.TM_CCOEFF_NORMED,
                 worst_match=0.75,
                 debug=False):
        self.scales = scales
        self.max_distance = max_distance

        self.criterion = criterion
        self.worst_match = worst_match
        self.debug = debug

    def match(self, feature, scene, mask=None, scale=None):
        """Find the location of _feature_ in _scene_, if there is one.

        Return a tuple containing the match's scale 

        Parameters:
            `feature`: A (small) image to be matched in _scene_.
            `scene`: A (large) image, usually raw data.
            `mask`: A subregion to narrow the search to.
            `scale`: A scaling factor to use for `feature.`
                     If None, will search for the best scaling factor
                     (by `self.criterion`) from `self.scales`.
        """
        if mask is not None:
            scene *= mask

        if scale is None:
            scale = self.find_best_scale(feature, scene)

        match_candidates = []

        if scale:
            scaled_feature = cv2.resize(feature, (0, 0), fx=scale, fy=scale)

            # Peaks in matchTemplate are good candidates.
            peak_map = cv2.matchTemplate(scene, scaled_feature,
                                         self.criterion)

            if self.criterion in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
                best_val, _, best_loc, _ = cv2.minMaxLoc(peak_map)
                good_points = np.where(peak_map <= self.worst_match)
            else:
                _, best_val, _, best_loc = cv2.minMaxLoc(peak_map)
                good_points = np.where(peak_map >= self.worst_match)

            good_points = list(zip(*good_points))

            if self.debug:
                LOGGER.warning("%f %f %f %s %s", scale, self.worst_match,
                               best_val, best_loc, good_points)

                cv2.imshow('edges', scene)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            clusters = self._get_clusters(good_points)

            match_candidates = [max(clust, key=lambda pt: peak_map[pt])
                                for clust in clusters]
            match_candidates = [(peak, peak_map[peak])
                                for peak in match_candidates]

        return (scale, peaks)

    def _get_clusters(self, pts, key=lambda x: x, max_clusters=None):
        """Run DBSCAN on the `pts`, applying `key` first if necessary,
        post-process the results into a list of lists, and return it,
        taking only the largest `max_clusters`.
        """
        if pts:
            kpts = [key(pt) for pt in pts]

            clustering = DBSCAN(eps=self.max_distance, min_samples=1).fit(kpts)

            # Post-processing.
            labeled_pts = list(zip(kpts, clustering.labels_))
            labeled_pts = sorted(labeled_pts, key=lambda p: p[1])

            clusters = [list(g) for l, g in groupby(labeled_pts, key=lambda p: p[1])]
            clusters = [[p[0] for p in clust] for clust in clusters]
            clusters = list(sorted(clusters, key=len, reverse=True))

            return clusters[:max_clusters]
        return []

    def find_best_scale(self, feature, scene):
        """Find the scale with the best correlation.
        """
        best_corr = 0
        best_scale = 0

        for scale in self.scales:
            scaled_feature = cv2.resize(feature, (0, 0), fx=scale, fy=scale)

            result = cv2.matchTemplate(scene, scaled_feature, self.criterion)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_corr:
                best_corr = max_val
                best_scale = scale

        if self.debug:
            LOGGER.warning("%f %f", best_scale, best_corr)

        if best_corr > self.worst_match:
            return best_scale
        return None
