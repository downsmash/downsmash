import json

class MatchData:
    def __init__(self, obj):
        self.screen = obj["screen"]
        self.ports = obj["ports"]
        self.chunks = obj["chunks"]
