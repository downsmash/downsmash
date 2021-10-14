#!/usr/bin/python

import itertools

import numpy as np
import cv2
import scipy.stats

from . import PERCENT, LOGGER
from . import constants as c
from .rect import Rect
from .stream_parser import StreamParser
from .template_matcher import TemplateMatcher
from .util import overlay_map, find_dlt


class Viewfinder(StreamParser):

    def __init__(self, filename, config):
        StreamParser.__init__(self, filename)
        
        # TODO document these.
        frames_to_sample = config.get("frames_to_sample", 50)
        self.frames = [frame for _, frame
                       in self.sample_frames(num_samples=frames_to_sample)]

        self.percent_y_tolerance = config.get("percent_y_tolerance", 5)
        min_scale = config.get("min_scale", 0.72)
        max_scale = config.get("max_scale", 0.87)
        scale_step = config.get("scale_step", 0.03)

        self.scales = np.arange(min_scale, max_scale, scale_step)

    def get_scale_and_screen(self):
        """Estimate the screen location and scale.
        """
        scale, screen = self.detect_screen()
        LOGGER.warning("Estimated screen is %s", screen)
        LOGGER.warning("Estimated scale is %.03f", scale)

        LOGGER.warning("Correcting screen...")
        scale, screen = self.correct_screen(scale, screen)
        LOGGER.warning("Estimated screen is %s", screen)
        LOGGER.warning("Estimated scale is %.03f", scale)

        return (scale, screen)

    def detect_screen(self):
        """Attempt to detect the screen.
        """
        scale, pct_locations = self.get_pct_locations(self.scales)

        # Approximate screen Y-pos from percents.
        height = int(round(c.SCREEN_HEIGHT * scale))
        width = int(round(c.SCREEN_WIDTH * scale))
        top = int(round(np.mean(pct_locations, axis=0)[0] - c.PERCENT_Y_POS * scale))

        # Determine the X-pos by the skewness-kurtosis method.
        LOGGER.info("Generating skewness-kurtosis map...")
        skew_kurt = overlay_map(self.frames) // 255

        actual_width = skew_kurt.shape[1]
        estimated_width = width

        if actual_width < estimated_width:
            LOGGER.warning("Estimated screen is bigger than the video, assuming left = 0...")
            left = 0
        else:
            possible_lefts = range(actual_width - estimated_width)

            def badness(left):
                return sum(sum(skew_kurt[max(0, top):max(0, top) + height, left:left+width]))
            left = min(possible_lefts, key=badness)

        return (scale, Rect(top, left, height, width))

    def get_pct_locations(self, scales):
        """Detect the percent signs in a random sample of frames.
        """
        matcher = TemplateMatcher(scales=scales)
        scale, pct_locations = self.locate(self.frames,
                                           PERCENT,
                                           matcher=matcher)

        if not scale or not pct_locations:
            raise RuntimeError('This doesn\'t appear to be Melee (no percent signs found!)')

        # Group the returned locations on the y-axis.
        def y_bin(location):
            y, x = location
            return y // self.percent_y_tolerance

        pct_locations = sorted(pct_locations, key=y_bin)
        location_groups = itertools.groupby(pct_locations, key=y_bin)
        location_groups = [list(g) for k, g in location_groups]

        # Choose the biggest group.
        pct_locations = max(location_groups, key=len)
        pct_locations = list(pct_locations)

        return (scale, pct_locations)

    def correct_screen(self, scale, screen):
        _, predicted, locations = self.detect_ports(scale, screen)

        if not predicted or not locations:
            raise RuntimeError('This doesn\'t appear to be Melee (no percent signs found!)')

        # Scaling factor, translation x, translation y
        scale_factor, shift_x, shift_y = find_dlt(predicted, locations)
        scale *= scale_factor
        screen.height *= scale_factor
        screen.width *= scale_factor
        screen.left *= scale_factor
        screen.left += shift_x
        screen.top *= scale_factor
        screen.top += shift_y

        return (scale, screen)

    def get_ports(self, scale, screen):
        ports, _, _ = self.detect_ports(scale, screen)
        if not ports:
            raise RuntimeError("No ports found!")
        LOGGER.warning("Ports are at %s", " ".join(str(s) for s in ports))

        return ports

    def detect_ports(self, scale, screen, max_error=0.06):
        """Find the approximate port locations.
        """
        ports = []
        predicted = []
        locations = []
        # TODO DRY this out
        for port_number in range(4):
            pct_roi = screen.subregion(c.PERCENT_Y_POS / c.SCREEN_HEIGHT,
                                       (c.PERCENT_X_POS + port_number * c.PERCENT_X_POS_STEP) / c.SCREEN_WIDTH,
                                       c.PERCENT_HEIGHT / c.SCREEN_HEIGHT,
                                       c.PERCENT_WIDTH / c.SCREEN_WIDTH,
                                       padding=max_error)

            pct_left = screen.left + (c.PERCENT_X_POS + port_number * c.PERCENT_X_POS_STEP) * scale
            pct_top = screen.top + c.PERCENT_Y_POS * scale

            matcher = TemplateMatcher(scales=[scale], worst_match=0.6, debug=False)
            portscale, location = self.locate(self.frames, PERCENT, matcher=matcher, roi=pct_roi)

            if portscale is None:
                ports.append(None)
            else:
                LOGGER.warning("Detected port %d at (%d, %d) (error %.03f px, %.03f px)",
                               port_number + 1, location[0][0], location[0][1],
                               location[0][0] - pct_top, location[0][1] - pct_left)

                predicted.append([pct_top, pct_left])
                locations.append(location[0])

                port_roi = screen.subregion(c.PORT_Y_POS / c.SCREEN_HEIGHT,
                                            (c.PORT_X_POS + port_number * c.PORT_X_POS_STEP) / c.SCREEN_WIDTH,
                                            c.PORT_HEIGHT / c.SCREEN_HEIGHT,
                                            c.PORT_WIDTH / c.SCREEN_WIDTH,
                                            padding=max_error)

                port_roi.left = max(port_roi.left, 0)
                port_roi.top = max(port_roi.top, 0)
                ports.append(port_roi)

        return (ports, predicted, locations)
