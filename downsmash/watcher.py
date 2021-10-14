import dataclasses, json

import numpy as np

from . import LOGGER
from .rect import Rect
from .viewfinder import Viewfinder
from .segmenter import Segmenter

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

@dataclasses.dataclass
class View:
    screen: Rect = None
    scale: float = None
    ports: list[Rect] = None

@dataclasses.dataclass
class MatchData:
    """Wrapper class for parsed match data.
    """
    view: View = None
    segments: list = None
    threshold: float = None

def watch(filename, config = None):
    if not config:
        config = {}

    match_data = MatchData()
    view = View()

    viewfinder = Viewfinder(filename, config)

    (view.scale, view.screen) = viewfinder.get_scale_and_screen()
    view.ports = viewfinder.get_ports(view.scale, view.screen)

    match_data.view = view

    segmenter = Segmenter(filename, view, config)

    match_data.threshold = segmenter.get_threshold()

    # How separated are the two groups?
    confs = segmenter.confidence['conf']
    mean_positive = np.mean(confs[confs >= match_data.threshold])
    mean_negative = np.mean(confs[confs < match_data.threshold])
    LOGGER.warning("Group means are (+)%.03f (-)%.03f", mean_positive, mean_negative)

    # TODO Replace magic numbers
    # TODO This error message needs to be more descriptive - something about
    # false negatives
    if mean_positive - mean_negative < 0.1 or mean_negative > 0.5:
        raise RuntimeError("This looks like an edited/gapless set"
                           "(mean_pos - mean_neg = %.03f)" % (mean_positive - mean_negative))

    match_data.segments = segmenter.get_segments(match_data.threshold)
    match_data.segments = segmenter.refine_segments(match_data.segments)

    return json.dumps(match_data, cls=EnhancedJSONEncoder, default=lambda o: o.__dict__, sort_keys=True, indent=2)
