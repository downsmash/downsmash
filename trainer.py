#!/usr/bin/python
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QComboBox, QPushButton, QWidget, QLabel

from segmenter import Segmenter
from entities import Characters


class TrainerDialog(QMainWindow):
    def set_char(self, char):
        self.char = char

    def set_stock(self, stock):
        self.stock = stock

    def set_pctval(self, pctval):
        self.pctval = pctval

    def __init__(self, img, last_state=None):
        super().__init__()

        self.setWindowTitle("Classify frame...")

        widget = QWidget(self)
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)

        self.charas = QComboBox()
        self.charas.addItems(sorted(Characters))
        self.charas.currentIndexChanged.connect(self.set_char)
        layout.addWidget(self.charas)

        self.stock = QComboBox()
        self.stock.addItems(str(a) for a in range(5))
        self.stock.currentIndexChanged.connect(self.set_stock)
        layout.addWidget(self.stock)

        self.pct = QComboBox()
        self.pct.addItem(None)
        self.pct.addItems(str(a) for a in range(1000))
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

                        char, stock, pctval = self.get_portstate(port_img)
                        print(char, stock, pctval)

    def get_portstate(self, img):
        app = QApplication([])
        trainWindow = TrainerDialog(img)
        trainWindow.show()
        app.exec_()
        return (trainWindow.char, trainWindow.stock, trainWindow.pctval)
