#!/usr/bin/python

import numpy as np
import cv2

# from ROI import ROI


class TemplateMatcher:

    def __init__(self, scales=np.arange(0.5, 1.0, 0.03),
                 max_clusters=None, max_distance=14,
                 min_corr=0.8,
                 thresh_min=50, thresh_max=200):
        self.scales = scales
        self.max_clusters = max_clusters
        self.max_distance = max_distance
        self.min_corr = min_corr
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

    def match(self, feature, scene, roi=None, scale=None, debug=False):
        if roi is not None:
            scene = scene[roi.top:(roi.top + roi.height),
                          roi.left:(roi.left + roi.width)]

        if not scale:
            scale = self.find_best_scale(feature, scene)
        peaks = []

        if scale:
            scaled_feature = cv2.resize(feature, (0, 0), fx=scale, fy=scale)

            canny_scene = cv2.Canny(scene, self.thresh_min, self.thresh_max)
            canny_feature = cv2.Canny(scaled_feature, self.thresh_min,
                                      self.thresh_max)

            # Threshold for peaks.
            corr_map = cv2.matchTemplate(canny_scene, canny_feature,
                                         cv2.TM_CCOEFF_NORMED)
            _, max_corr, _, max_loc = cv2.minMaxLoc(corr_map)

            good_points = np.where(corr_map >= max_corr - self.tolerance)
            good_points = list(zip(*good_points))

            if debug:
                print(max_corr, good_points)

            clusters = self.get_clusters(good_points,
                                         max_distance=self.max_distance)
            peaks = [max([(pt, corr_map[pt]) for pt in cluster],
                         key=lambda pt: pt[1]) for cluster in clusters]

        return (scale, peaks)

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

    def find_best_scale(self, feature, scene):
        best_corr = 0
        best_scale = 0

        for scale in self.scales:
            scaled_feature = cv2.resize(feature, (0, 0), fx=scale, fy=scale)

            result = cv2.matchTemplate(scene, scaled_feature,
                                       cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_corr:
                best_corr = max_val
                best_scale = scale

        if best_corr > self.min_corr:
            return best_scale
        else:
            return None

    def locate(self, feature, roi=None, max_clusters=None, N=10, debug=False):
        peaks = []
        best_scale_log = []

        for (n, scene) in self.sample_frames(num_samples=N):
            cv2.imwrite("scene.png", scene)
            scene = cv2.imread("scene.png")

            scale, these_peaks = self.match(feature, scene,
                                            roi=roi, debug=debug)
            # if debug: logging.warn("{0} {1}".format(scale, these_peaks))

            if scale:
                best_scale_log += [scale]

                these_peaks = sorted(these_peaks, key=lambda pt: pt[1])
                these_peaks = [loc for loc, corr in these_peaks]

                peaks.extend(these_peaks[:max_clusters])

        feature_locations = [np.array(max(set(cluster), key=cluster.count))
                             for cluster in self.get_clusters(peaks)]
        feature_locations = sorted(feature_locations, key=lambda pt: pt[1])

        if roi is not None:
            feature_locations = [np.array((roi.top, roi.left)) + loc
                                 for loc in feature_locations]

        if best_scale_log:
            mean_best_scale = sum(best_scale_log) / len(best_scale_log)
        else:
            mean_best_scale = None

        return (mean_best_scale, feature_locations)
