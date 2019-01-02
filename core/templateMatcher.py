#!/usr/bin/python

import numpy as np
import cv2
import logging


class TemplateMatcher:

    def __init__(self, scales=np.arange(0.5, 1.0, 0.03),
                 max_clusters=None, max_distance=14,
                 thresh_min=50, thresh_max=200,
                 criterion=cv2.TM_CCOEFF_NORMED,
                 worst_match=0.75):
        self.scales = scales
        self.max_clusters = max_clusters
        self.max_distance = max_distance
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

        self.criterion = criterion
        self.worst_match = worst_match

    def match(self, feature, scene, mask=None, scale=None, edge=False,
              debug=False):
        if mask is not None:
            scene *= mask

        if scale is None:
            scale = self.find_best_scale(feature, scene, debug=debug)
        peaks = []

        if scale:
            scaled_feature = cv2.resize(feature, (0, 0), fx=scale, fy=scale)

            if edge:
                scene = cv2.Canny(scene, self.thresh_min, self.thresh_max)
                scaled_feature = cv2.Canny(scaled_feature, self.thresh_min,
                                           self.thresh_max)

            # Threshold for peaks.
            peak_map = cv2.matchTemplate(scene, scaled_feature,
                                         self.criterion)

            if self.criterion in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
                best_val, _, best_loc, _ = cv2.minMaxLoc(peak_map)
                good_points = np.where(peak_map <= self.worst_match)
            else:
                _, best_val, _, best_loc = cv2.minMaxLoc(peak_map)
                good_points = np.where(peak_map >= self.worst_match)

            good_points = list(zip(*good_points))

            if debug:
                logging.warn("%f %f %f %s %s", scale, self.worst_match,
                             best_val, best_loc, good_points)

                cv2.imshow('edges', scene)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            clusters = self.get_clusters(good_points,
                                         max_distance=self.max_distance)

            peaks = [max(clust, key=lambda pt: peak_map[pt])
                     for clust in clusters]
            peaks = [(peak, peak_map[peak]) for peak in peaks]

        return (scale, peaks)

    # TODO This can be replaced by DBSCAN without a significant drop in
    # performance
    def get_clusters(self, pts, max_distance=14, key=lambda x: x):
        clusters = []
        for pt in pts:
            for idx, cluster in enumerate(clusters):
                if min(np.linalg.norm(np.subtract(key(pt), key(x)))
                       for x in cluster) < max_distance:
                    clusters[idx] += [pt]
                    break
            else:
                clusters.append([pt])

        return clusters

    def find_best_scale(self, feature, scene, debug=False):
        best_corr = 0
        best_scale = 0

        for scale in self.scales:
            scaled_feature = cv2.resize(feature, (0, 0), fx=scale, fy=scale)

            result = cv2.matchTemplate(scene, scaled_feature, self.criterion)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_corr:
                best_corr = max_val
                best_scale = scale

        if debug:
            logging.warn("%f %f", best_scale, best_corr)

        if best_corr > self.worst_match:
            return best_scale
        else:
            return None