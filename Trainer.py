#!/usr/bin/python
from MeleeVODParser import MeleeVODParser


class Trainer(MeleeVODParser):
    def __init__(self, filename):
        MeleeVODParser.__init__(self, filename)
