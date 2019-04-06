#!/usr/bin/python
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import cv2

import logging
from collections import namedtuple

from slippi import Game

import sys
sys.path.append('..')

from core.streamParser import StreamParser
from core.segmenter import Segmenter
# from core.entities import *

PortState = namedtuple("PortState", "char, stocks, pct")

charas = ["falcon", "dk", "fox", "gw", "kirby", "bowser",
          "link", "luigi", "mario", "marth", "mewtwo",
          "ness", "peach", "pikachu", "ics", "jigglypuff",
          "samus", "yoshi", "zelda", "sheik", "falco", "younglink",
          "drmario", "roy", "pichu", "ganondorf"]

class TrainerDialog(QMainWindow):
    def __init__(self, img, states, last_state=None):
        super().__init__()
        self.state = 0

        self.setWindowTitle("Classify frame...")

        widget = QWidget(self)
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)

        self.states_list = QComboBox()
        self.states_list.addItem("")
        for state in states:
            self.states_list.addItem("{0} stocks {1}%".format(*state))

        self.states_list.currentIndexChanged.connect(self.set_state)
        if last_state:
            self.states_list.setCurrentIndex(last_state)
        self.states_list.setFocus()
        layout.addWidget(self.states_list)

        h, w, d = img.shape
        qimg = QImage(img.copy(), w, h, d * w, QImage.Format_RGB888)
        qimg = qimg.rgbSwapped()  # BGR2RGB
        self.imlabel = QLabel()
        pixmap = QPixmap.fromImage(qimg)
        self.imlabel.setPixmap(QPixmap(pixmap))
        layout.addWidget(self.imlabel)

        self.classify = QPushButton("Classify")
        self.classify.clicked.connect(self.close)
        layout.addWidget(self.classify)

    def keyPressEvent(self, event):
        if isinstance(event, QKeyEvent):
            if event.key() == Qt.Key_Return:
                self.close()

    def set_state(self, i):
        self.state = i


class Trainer(StreamParser):
    def __init__(self, filename, slippi_filename):
        super().__init__(filename)
        self.filename = filename.split("/")[-1][:-4]
        self.slippi = Game(slippi_filename)

        self.chars = self.slippi.start.players
        def get_states(port_idx):
            all_frame_states = [frame.ports[port_idx].leader for frame in self.slippi.frames]
            states = [(state.post.stocks, int(state.post.damage)) for state in all_frame_states]
            states = list(set(states))
            states = list(sorted(states, key=lambda state: (-state[0], state[1])))

            return states

        self.states = [get_states(port_idx) if port else None for port_idx, port in enumerate(self.chars)]

    def train(self):
        last_states = [None for _ in self.chars]
        for t, scene in self.sample_frames(interval=0.5, color=True):
            frameno = int(t * 60)
            for port_idx, port in enumerate(self.slippi.frames[0].ports):
                if port:
                    top = 395
                    height = 95
                    left = 23 + 153 * port_idx
                    width = 140
                    port_img = scene[top:(top + height), left:(left + width)]

                    state = self.get_portstate(port_img, self.states[port_idx], last_state=last_states[port_idx])
                    last_states[port_idx] = state
                    if state > 0:
                        stocks, pct = self.states[port_idx][state - 1]
                        char_id = self.chars[port_idx].character
                        costume_id = self.chars[port_idx].costume
                        img_filename = "sources/{0}-{1}-{2}-{3}-{4}-{5}.png".format(char_id, costume_id, stocks, pct, self.filename, frameno)
                        cv2.imwrite(img_filename, port_img)
                        logging.warn("Wrote %s", img_filename)

    def get_portstate(self, img, states, last_state=None):
        app = QApplication([])
        trainWindow = TrainerDialog(img, states, last_state=last_state)
        trainWindow.show()
        app.exec_()
        return trainWindow.state
