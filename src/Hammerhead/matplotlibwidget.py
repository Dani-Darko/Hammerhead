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
        self.figure = Figure(dpi=96, facecolor="#F0F0F0", tight_layout=True, *args, **kwargs)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)