#!/usr/bin/python

import logging
import itertools

import numpy as np
import cv2
import scipy.stats

from pkg_resources import resource_string

from .rect import Rect
from .stream_parser import StreamParser
from .template_matcher import TemplateMatcher


LOGGER = logging.getLogger(__name__)

PERCENT_Y_POS = 358
PERCENT_X_POS = 110
PERCENT_X_POS_STEP = 132
PERCENT_HEIGHT = 32
PERCENT_WIDTH = 32

PORT_Y_POS = 308
PORT_X_POS = 20
PORT_X_POS_STEP = 132
PORT_HEIGHT = 74
PORT_WIDTH = 100

SCREEN_WIDTH = 548
SCREEN_HEIGHT = 411

# Read in percent sign
NPARR = np.fromstring(resource_string("core.resources", "pct.png"), np.uint8)
PERCENT = cv2.imdecode(NPARR, 1)

class Viewfinder(StreamParser):

    def __init__(self, filename, polling_interval=2):
        StreamParser.__init__(self, filename)
        self.polling_interval = polling_interval

    def get_pct_locations(self, scales):
        """Detect the percent signs in a random sample of frames.
        """
        matcher = TemplateMatcher(scales=scales)
        scale, pct_locations = self.locate(PERCENT, matcher=matcher, num_samples=30)

        if not scale or not pct_locations:
            raise RuntimeError('This doesn\'t appear to be Melee (no percent signs found!)')

        # Group the returned locations to within 5 px tolerance on y-axis.
        pct_locations = sorted(pct_locations, key=lambda l: l[0] // 5)
        location_groups = itertools.groupby(pct_locations, lambda l: l[0] // 5)
        location_groups = [(k, list(g)) for k, g in location_groups]

        # Choose the biggest group.
        _, pct_locations = max(location_groups, key=lambda g: len(g[1]))
        pct_locations = list(pct_locations)

        return (scale, pct_locations)

    def correct_screen(self, scale, screen):
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
        # TODO Move this out.
        _, predicted, locations = self.detect_ports(scale, screen)

        if not predicted or not locations:
            raise RuntimeError('This doesn\'t appear to be Melee (no percent signs found!)')
        # End move out

        predicted_mat = []
        for (predicted_y, predicted_x) in predicted:
            predicted_mat.append([predicted_y, 0, 1])
            predicted_mat.append([predicted_x, 1, 0])

        actual_vec = []
        for (actual_y, actual_x) in locations:
            actual_vec.append(actual_y)
            actual_vec.append(actual_x)
        actual_vec = np.array(actual_vec).transpose()


        ols, resid, _, _ = np.linalg.lstsq(predicted_mat, actual_vec, rcond=None)

        print(predicted_mat, actual_vec, ols, actual_vec - np.dot(predicted_mat, ols))
        # Scaling factor, translation x, translation y
        scale_factor, shift_x, shift_y = ols
        scale *= scale_factor
        screen.height *= scale_factor
        screen.width *= scale_factor
        screen.left *= scale_factor
        screen.left += shift_x
        screen.top *= scale_factor
        screen.top += shift_y

        return (scale, screen)

    def detect_screen(self):
        """Attempt to detect the screen.
        """
        scale, pct_locations = self.get_pct_locations(scales=np.arange(0.72, 0.87, 0.03))

        # Approximate screen Y-pos from percents.
        height = int(round(SCREEN_HEIGHT * scale))
        width = int(round(SCREEN_WIDTH * scale))
        top = int(round(np.mean(pct_locations, axis=0)[0] - PERCENT_Y_POS * scale))

        # Determine the X-pos by the skewness-kurtosis method.
        LOGGER.info("Generating skewness-kurtosis map...")
        skew_kurt = self.overlay_map() // 255

        t = max(0, top)
        possible_lefts = range(skew_kurt.shape[1] - width)
        if not possible_lefts:
            LOGGER.warning("Estimated screen is bigger than the video, assuming left = 0...")
            left = 0
        else:
            # TODO clean this up
            goodnesses = [sum(sum(skew_kurt[t:t + height, left:left + width]))
                          for left in possible_lefts]
            goodnesses = np.array(goodnesses)

            left = np.argmin(goodnesses)

        return (scale, Rect(top, left, height, width))

    @staticmethod
    def _scale_to_interval(array, new_min, new_max):
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

    def overlay_map(self, num_samples=50, start=None, end=None):
        """Run a skewness-kurtosis filter on a sample of frames and
        edge-detect.

        The areas of the video containing game feed should come back black.
        Areas containing overlay or letterboxes will be visibly white.
        """
        data = None
        for _, frame in self.sample_frames(num_samples=num_samples,
                                           start=start, end=end):
            if not data:
                data = [frame]
            else:
                data += [frame]

        sd_map = np.sqrt(np.var(data, axis=0))
        skew_map = scipy.stats.skew(data, axis=0)
        kurt_map = scipy.stats.kurtosis(data, axis=0)
        min_map = np.minimum(skew_map, kurt_map)  # pylint:disable=assignment-from-no-return

        min_map = self._scale_to_interval(min_map, 0, 255).astype(np.uint8)

        # Blur and edge detect.
        min_map = cv2.blur(min_map, (5, 5))
        edges = cv2.Laplacian(min_map, cv2.CV_8U)

        # Areas that are constant throughout the video (letterboxes) will
        # have 0 skew, 0 kurt, and 0 variance, so the skew-kurt filter
        # will miss them
        edges[np.where(sd_map < 0.01)] = 255
        _, edges = cv2.threshold(edges, 7, 255, cv2.THRESH_BINARY)

        return edges

    def detect_ports(self, scale, screen, max_error=0.06):
        """Find the approximate port locations.
        """
        ports = []
        predicted = []
        locations = []
        # TODO DRY this out
        for port_number in range(4):
            pct_roi = screen.subregion(PERCENT_Y_POS / SCREEN_HEIGHT,
                                       (PERCENT_X_POS + port_number * PERCENT_X_POS_STEP) / SCREEN_WIDTH,
                                       PERCENT_HEIGHT / SCREEN_HEIGHT,
                                       PERCENT_WIDTH / SCREEN_WIDTH,
                                       padding=max_error)

            pct_left = screen.left + (PERCENT_X_POS + port_number * PERCENT_X_POS_STEP) * scale
            pct_top = screen.top + PERCENT_Y_POS * scale

            matcher = TemplateMatcher(scales=[scale], worst_match=0.6, debug=False)
            portscale, location = self.locate(PERCENT, num_samples=10, matcher=matcher, roi=pct_roi)

            if portscale is None:
                ports.append(None)
                continue

            LOGGER.warning("Detected port %d at (%d, %d) (error %.03f px, %.03f px)",
                           port_number + 1, location[0][0], location[0][1],
                           location[0][0] - pct_top, location[0][1] - pct_left)

            predicted.append([pct_top, pct_left])
            locations.append(location[0])

            port_roi = screen.subregion(PORT_Y_POS / SCREEN_HEIGHT,
                                        (PORT_X_POS + port_number * PORT_X_POS_STEP) / SCREEN_WIDTH,
                                        PORT_HEIGHT / SCREEN_HEIGHT,
                                        PORT_WIDTH / SCREEN_WIDTH,
                                        padding=max_error)

            ports.append(port_roi)

        return (ports, predicted, locations)
