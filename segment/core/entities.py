#!/usr/bin/env python
"""The segmenter's data model.
"""

from dataclasses import dataclass
from typing import List

@dataclass
class PortState:
    """Represents a player's state as a subgroup of SetState.
    """
    port_id: int
    stocks: int
    pct: float

@dataclass
class SetState:
    """Represents a timestamp from a set.
    """
    timestamp: float
    ports: List[PortState]

@dataclass
class Set:
    """Represents a Melee set.
    """
    id: int
    states: List[SetState]

@dataclass
class Source:
    """Represents a video object from which a Melee set may be parsed.
    """
    id: int
    video_slug: str
