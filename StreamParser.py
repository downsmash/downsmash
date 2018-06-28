#!/usr/bin/python

import cv2
import numpy as np
from random import randint
import logging

from ROI import ROI

logging.basicConfig(format="%(message)s")


class StreamParser:

    def __init__(self, filename):
        self.filename = filename
        self.vc = cv2.VideoCapture(filename)
        self.roi = ROI(0, 0, self.vc.get(4), self.vc.get(3))

    def parse(self):
        raise NotImplementedError

    def sample_frames(self, start=None, end=None, interval=None,
                      num_samples=None, fuzz=0):
        if (interval is None and num_samples is None) or \
                None not in (interval, num_samples):
            raise ValueError('exactly one of (interval, num_samples) '
                             'must be set')

        video_length = self.vc.get(7) / self.vc.get(5)
        if not start or start < 0:
            start = 0
        if not end or end > video_length:
            end = video_length

        total_time = end - start

        if not num_samples:
            num_samples = total_time // interval

        for time in np.linspace(start, end, num=num_samples):
            time += randint(-1 * fuzz, fuzz) / self.vc.get(5)

            if time < start:
                time = start
            elif time > end:
                time = end

            self.vc.set(0, int(time * 1000))
            success, frame = self.vc.read()

            if success:
                yield (time, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        return
