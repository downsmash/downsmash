#!/usr/bin/python
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QComboBox, QPushButton, QWidget, QLabel
import cv2

import logging
from streamParser import StreamParser
from segmenter import Segmenter
from entities import Characters  # PortState
from collections import namedtuple

PortState = namedtuple("PortState", "char, stocks, pct")


class TrainerDialog(QMainWindow):
    def set_char(self, char):
        self.char = self.charas.currentText()

    def set_stocks(self, stocks):
        self.stocks = self.stock.currentText()

    def set_pctval(self, pctval):
        self.pctval = self.pct.currentText()

    def __init__(self, img, last_state=None):
        super().__init__()

        self.setWindowTitle("Classify frame...")

        widget = QWidget(self)
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)

        # TODO DRY this out somehow

        self.charas = QComboBox()
        self.charas.addItems(char.name for char in Characters)

        if last_state and last_state.char:
            self.char = last_state.char
        else:
            self.char = [char.name for char in Characters][0]
        self.charas.setCurrentIndex(self.charas.findText(self.char))

        self.charas.currentIndexChanged.connect(self.set_char)
        layout.addWidget(self.charas)

        self.stock = QComboBox()
        self.stock.addItems(str(a) for a in range(5))

        if last_state and last_state.stocks:
            self.stocks = last_state.stocks
        else:
            self.stocks = "0"
        self.stock.setCurrentIndex(self.stock.findText(self.stocks))

        self.stock.currentIndexChanged.connect(self.set_stocks)
        layout.addWidget(self.stock)

        self.pct = QComboBox()
        self.pct.addItem("none")
        self.pct.addItems(str(a) for a in range(1000))

        if last_state and last_state.pctval:
            self.pctval = last_state.pctval
        else:
            self.pctval = "none"
        self.pct.setCurrentIndex(self.pct.findText(self.pctval))

        self.pct.currentIndexChanged.connect(self.set_pctval)
        layout.addWidget(self.pct)

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


class Trainer(StreamParser):
    def __init__(self, filename):
        super.__init__(self, filename)
        self.filename = filename.split("/")[-1][:-4]

    def train(self):
        self.parse()

        last_state = None
        for start, end in self.chunks:
            for t, scene in self.sample_frames(start=start, end=end,
                                               interval=0.5, color=True):
                for n, port in enumerate(self.ports):
                    if port:
                        port_img = scene[port.top:(port.top + port.height),
                                         port.left:(port.left + port.width)]

                        char, stocks, pct = self.get_portstate(port_img, last_state=last_state)
                        last_state = PortState(char=char, stocks=stocks, pct=pct)
                        img_filename = "sources/{0}-{1}-{2}-{3}-{4}.png".format(char, stocks, pct, self.filename, t)
                        cv2.imwrite(img_filename, port_img)
                        logging.warn("Wrote %s", img_filename)

    def get_portstate(self, img):
        app = QApplication([])
        trainWindow = TrainerDialog(img)
        trainWindow.show()
        app.exec_()
        return (trainWindow.char, trainWindow.stocks, trainWindow.pctval)
