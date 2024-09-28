##############################################################################################
#                                                                                            #
#       PyQt5 matplotlib figure widget                                                       #
#       Author: Alexander Liptak (2019)                                                      #
#                                                                                            #
##############################################################################################

from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class MatplotlibWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, *args, **kwargs):
        super(MatplotlibWidget, self).__init__(parent)
        self.figure = Figure(dpi=96, facecolor="#ECECEC", layout="constrained", *args, **kwargs)
        self.canvas = FigureCanvas(self.figure)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)