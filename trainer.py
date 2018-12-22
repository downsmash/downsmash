#!/usr/bin/python
from segmenter import Segmenter
from entities import *


class Trainer(Segmenter):
    def __init__(self, filename):
        Segmenter.__init__(self, filename)

    def train(self):
        self.parse()

        for start, end in self.chunks:
            for t, scene in self.sample_frames(start=start, end=end,
                                               interval=0.5, color=True):
                for n, port in enumerate(self.ports):
                    if port:
                        port_img = scene[port.top:(port.top + port.height),
                                         port.left:(port.left + port.width)]

                        portstate = self.get_portstate(port_img)

    def get_portstate(self, img):
        return
