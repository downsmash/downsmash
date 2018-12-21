#!/usr/bin/python
from Segmenter import Segmenter


class Trainer(Segmenter):
    def __init__(self, filename):
        Segmenter.__init__(self, filename)
        
