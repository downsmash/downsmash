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

    def match(self, feature, scene, mask=None, scale=None, crop=True):
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
        mask : ndarray
            A subregion to narrow the search to, as an array of zeros and
            ones (respectively, pixels to mask out and pixels to leave in)
            of the same size as `scene`.
        scale : float
            A scaling factor to use for `feature`. If None, will use the best
            scale as returned by `self._find_best_scale`.

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
        scene_height, scene_width = scene.shape
        crop_top = crop_left = 0
        crop_bottom = scene_height
        crop_right = scene_width

        if mask is not None:
            scene_working *= mask
            if crop:
                mask_y = [y for y in range(scene_height) if 1 in mask[y]]
                mask_x = [x for x in range(scene_width) if 1 in mask[:, x]]

                crop_top, crop_bottom = min(mask_y), max(mask_y)
                crop_left, crop_right = min(mask_x), max(mask_x)

        scene_working = scene_working[crop_top:(crop_bottom + 1),
                                      crop_left:(crop_right + 1)]

        if scale is None:
            scale = self._find_best_scale(feature, scene)

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

            clusters = get_clusters(good_points,
                                    max_distance=self.max_distance)

            match_candidates = [max(clust, key=lambda pt: peak_map[pt])
                                for clust in clusters]
            match_candidates = [((peak[0] + crop_top, peak[1] + crop_left),
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
