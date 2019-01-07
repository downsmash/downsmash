#!/usr/bin/python

from .rect import Rect
from .streamParser import StreamParser
from .templateMatcher import TemplateMatcher


PERCENT_Y_PCT = .871
PERCENT_X_PCT = .2
PERCENT_X_PCT_STEP = .2381
PERCENT_HEIGHT = .06
PERCENT_WIDTH = .06

PORT_Y_PCT = .75
PORT_X_PCT = .0363
PORT_X_PCT_STEP = .24
PORT_HEIGHT = .18
PORT_WIDTH = .1833

# Read in percent sign
nparr = np.fromstring(resource_string("core.resources", "pct.png"), np.uint8)
PERCENT = cv2.imdecode(nparr, 1)

class Viewfinder(StreamParser):

    def __init__(self, filename, polling_interval=2):
        super().__init__(self, filename)
        self.polling_interval = polling_interval

    def get_feature_locations(self, feature, scales):

        #
        tm = TemplateMatcher(max_clusters=2, scales=scales)
        scale, pct_locations = self.locate(feature, tm=tm, N=30)

    def detect_screen(self):
        """Attempt to detect the screen.
        """

        # Detect the percent signs in a random sample of frames.
        tm = TemplateMatcher(max_clusters=2, scales=np.arange(0.6, 1.1, 0.015))
        scale, pct_locations = self.locate(PERCENT, tm=tm, N=30)

        # Group the returned locations to within 5 px tolerance on y-axis.
        pct_locations = sorted(pct_locations, key=lambda l: l[0] // 5)
        location_groups = itertools.groupby(pct_locations, lambda l: l[0] // 5)
        location_groups = [(k, list(g)) for k, g in location_groups]

        # Choose the biggest group.
        _, pct_locations = max(location_groups, key=lambda g: len(g[1]))
        pct_locations = list(pct_locations)

        # Approximate screen Y-pos from percents.
        height, width = [x * scale / 0.05835 for x in PERCENT.shape[:2]]
        top = np.mean(pct_locations, axis=0)[0] - PERCENT_Y_PCT * height

        # Determine the X-pos by the skewness-kurtosis method.
        logging.info("Generating skew-kurtosis map...")
        overlay = self.overlay_map()
        leftmost_pct = min(pct_locations, key=lambda pos: pos[1])[1]

        leftmost_port = None
        best_goods = 0

        # The leftmost percent sign can be one of four ports.
        for port_no in range(4):
            left = leftmost_pct - (PERCENT_X_PCT + PERCENT_X_PCT_STEP * port_no) * width
            screen = Rect(top, left, height, width) & self.shape
            pixels = overlay[screen.top:(screen.top + screen.height),
                             screen.left:(screen.left + screen.width)]
            goods = np.count_nonzero(pixels == 0)
            if goods > best_goods:
                best_goods = goods
                leftmost_port = port_no

        left = leftmost_pct - (PERCENT_X_PCT + PERCENT_X_PCT_STEP * leftmost_port) * width
        # self.data["scale"] = (height / 411 + width / 548) / 2
        self.data["scale"] = scale

        return Rect(top, left, height, width) & self.shape

    def overlay_map(self, num_samples=50, start=None, end=None):
        """Run a skewness-kurtosis filter on a sample of frames and
        edge-detect.

        The areas of the video containing game feed should come back black.
        Areas containing overlay or letterboxes will be visibly white.
        """
        data = None
        for time, frame in self.sample_frames(num_samples=num_samples,
                                              start=start, end=end):
            if not data:
                data = [frame]
            else:
                data += [frame]

        sd_map = np.sqrt(np.var(data, axis=0))
        skew_map = scipy.stats.skew(data, axis=0)
        kurt_map = scipy.stats.kurtosis(data, axis=0)
        min_map = np.minimum(skew_map, kurt_map)

        map_min = min(min_map.flatten())
        map_max = max(min_map.flatten())

        # Clip to [0, 255], with 0=min and 255=max
        clipped = ((min_map - map_min) / (map_max - map_min) * 255)
        clipped = clipped.astype(np.uint8)

        # Blur and edge detect.
        blurred = cv2.blur(clipped, (5, 5))
        edges = cv2.Canny(blurred, 50, 150)

        # Areas that are constant throughout the video (letterboxes) will
        # have 0 skew, 0 kurt, and 0 variance, so the skew-kurt filter
        # will miss them
        edges[np.where(sd_map < 0.01)] = 255

        return edges

    def detect_ports(self, max_error=0.06):
        ports = []
        # TODO DRY this out
        for port_number in range(4):
            pct_xloc = PERCENT_X_PCT + PERCENT_X_PCT_STEP * port_number
            (pct_top, pct_left) = self.data["screen"][PERCENT_Y_PCT, pct_xloc]
            pct_roi_top = pct_top - max_error * self.data["screen"].height
            pct_roi_left = pct_left - max_error * self.data["screen"].width
            pct_roi_height = (PERCENT_HEIGHT + 2 * max_error) * self.data["screen"].height
            pct_roi_width = (PERCENT_WIDTH + 2 * max_error) * self.data["screen"].width
            pct_roi = Rect(pct_roi_top, pct_roi_left,
                           pct_roi_height, pct_roi_width)

            pct_roi &= self.data["screen"]

            tm = TemplateMatcher(max_clusters=1, scales=[self.data["scale"]],
                                 worst_match=0.6)
            scale, location = self.locate(PERCENT, N=10, tm=tm, roi=pct_roi)
            if scale is None:
                ports.append(None)
                continue

            error = location[0] - (pct_top, pct_left)
            logging.warn("Detected port {0} at {1} "
                         "(error {2[0]}px, {2[1]}px)"
                         .format(port_number + 1, location[0], error))

            port_xloc = PORT_X_PCT + PORT_X_PCT_STEP * port_number
            (port_top, port_left) = self.data["screen"][PORT_Y_PCT, port_xloc]
            port_roi_top = port_top - max_error * self.data["screen"].height
            port_roi_left = port_left - max_error * self.data["screen"].width
            port_roi_height = (PORT_HEIGHT + 2 * max_error) * self.data["screen"].height
            port_roi_width = (PORT_WIDTH + 2 * max_error) * self.data["screen"].width
            port_roi = Rect(port_roi_top, port_roi_left,
                            port_roi_height, port_roi_width)

            port_roi &= self.data["screen"]
            ports.append(port_roi)

        return ports
