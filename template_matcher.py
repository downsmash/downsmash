"""
"""

from itertools import groupby
import logging

import numpy as np
import cv2

from .cluster import get_clusters

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

    def match(self, feature, scene, mask=None, scale=None, crop=True,
              cluster=True):
        """Find the location of _feature_ in _scene_, if there is one.

        Return a tuple containing the best match scale and the best match
        candidates.

        Parameters
        ----------
        feature : ndarray
            A (small) image to be matched in _scene_, as an OpenCV-compatible
            array.
        scene : ndarray
            A (large) image, usually raw data, as an OpenCV-compatible array.
        mask : Rect
            A subregion to narrow the search to, as an array of zeros and
            ones (respectively, pixels to mask out and pixels to leave in)
            of the same size as `scene`.
        scale : float
            A scaling factor to use for `feature`. If None, will use the best
            scale as returned by `self._find_best_scale`.
        crop : bool
            Whether to crop the search region to the mask, if there is one.
        cluster : bool
            Whether to run DBSCAN on the matches for stability.

        Returns
        -------
        scale : float
            The scaling factor used for `candidates`.
            If `scale` was passed as a keyword argument, the same value will
            be returned.
        candidates : list[tuple(tuple(int, int), int)]
            A list of positions and criterion scores. To be returned, the
            template match at a position must exceed `self.worst_match`.
        """

        scene_working = scene.copy()

        if (mask is not None) and not crop:
            scene_working *= mask.to_mask()

        scene_working = scene_working[mask.top:(mask.top + mask.height),
                                      mask.left:(mask.left + mask.width)]

        if scale is None:
            scale = self._find_best_scale(feature, scene_working)

        match_candidates = []

        if scale:
            scaled_feature = cv2.resize(feature, (0, 0), fx=scale, fy=scale)

            # Peaks in matchTemplate are good candidates.
            peak_map = cv2.matchTemplate(scene_working, scaled_feature,
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

                cv2.imshow('edges', scene_working)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            if cluster:
                clusters = get_clusters(good_points,
                                        max_distance=self.max_distance)
            else:
                clusters = [(pt,) for pt in good_points]

            # TODO Break these down into more comprehensible comprehensions.
            match_candidates = [max(clust, key=lambda pt: peak_map[pt])
                                for clust in clusters]
            match_candidates = [((peak[0] + mask.top, peak[1] + mask.left),
                                 peak_map[peak])
                                for peak in match_candidates]

        return (scale, match_candidates)

    def _find_best_scale(self, feature, scene):
        """Find the scale with the best score (by `self.criterion`) from
        `self.scales`.


        Parameters
        ----------
        feature : ndarray
            A (small) image to be matched in _scene_, as an OpenCV-compatible
            array.
        scene : ndarray
            A (large) image, usually raw data, as an OpenCV-compatible array.

        Returns
        -------
        best_scale
            The scaling factor that obtains the best score, or None if no
            score is better than `self.worst_match`.
        """
        # TODO Greyscale/color?

        best_corr = 0
        best_scale = None

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
