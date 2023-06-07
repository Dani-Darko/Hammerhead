from PyQt5.QtWidgets import QMainWindow, QApplication
from layouts import layout_main
import sys

class HammerHeadMain(QMainWindow, layout_main.Ui_MainWindow):
    def __init__(self, parent=None):
        super(HammerHeadMain, self).__init__(parent)
        self.setupUi(self)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    hammer_head_main = HammerHeadMain()
    hammer_head_main.show()
    sys.exit(app.exec())