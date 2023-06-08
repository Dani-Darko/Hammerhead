from PyQt5.QtCore import QEvent, QObject, QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QPolygon
from PyQt5.QtWidgets import QApplication, QMainWindow
from layouts import layoutMain
import sys

class HammerHeadMain(QMainWindow, layoutMain.Ui_MainWindow):
    def __init__(self, parent=None):
        super(HammerHeadMain, self).__init__(parent)
        self.setupUi(self)

        self.labelTestButton.setPixmap(QPixmap("./assets/img/buttonTemplate.png"))
        self.labelTestButton2.setPixmap(QPixmap("./assets/img/buttonTemplate.png"))
        self.labelTestButtonBB = self.getHexBoundingBox(self.labelTestButton)
        self.labelTestButtonBB2 = self.getHexBoundingBox(self.labelTestButton2)
        
    @staticmethod   
    def getHexBoundingBox(obj):
        x, y, w, h = obj.pos().x(), obj.pos().y(), obj.width(), obj.height()
        a = QPoint(x + (w // 2), y)
        b = QPoint(x + w,        y + (h // 4))
        c = QPoint(x + w,        y + (3 * (h // 4)))
        d = QPoint(x + (w // 2), y + h)
        e = QPoint(x,            y + (3 * (h // 4)))
        f = QPoint(x,            y + (h // 4))
        return QPolygon([a, b, c, d, e, f])
    
    def mouseMoveEvent(self, event):
        if self.labelTestButtonBB.containsPoint(event.pos(), Qt.OddEvenFill):
            self.labelTestButton.setEnabled(False)
            self.labelTestButton2.setEnabled(True)
        elif self.labelTestButtonBB2.containsPoint(event.pos(), Qt.OddEvenFill):
            self.labelTestButton2.setEnabled(False)
            self.labelTestButton.setEnabled(True)
        else:
            self.labelTestButton.setEnabled(True)
            self.labelTestButton2.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    hammer_head_main = HammerHeadMain()
    hammer_head_main.show()
    sys.exit(app.exec())