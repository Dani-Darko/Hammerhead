from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QPixmap, QPolygon
from PyQt5.QtWidgets import QApplication, QMainWindow
from layouts import layoutMain
import sys


class HammerHeadMain(QMainWindow, layoutMain.Ui_MainWindow):

    def __init__(self, parent=None):
        super(HammerHeadMain, self).__init__(parent)
        self.setupUi(self)
        self.setFixedSize(self.size())
                        
        self.buttonProperties = {"buttonHFM": {"object" : self.labelButtonHFM,
                                               "pixmap" : {"default": QPixmap("./assets/img/buttonHFMDefault.png"),
                                                           "hovered": QPixmap("./assets/img/buttonHFMHovered.png"),
                                                           "pressed": QPixmap("./assets/img/buttonHFMPressed.png")}},
                                 "buttonSM" : {"object" : self.labelButtonSM,
                                               "pixmap" : {"default": QPixmap("./assets/img/buttonSMDefault.png"),
                                                           "hovered": QPixmap("./assets/img/buttonSMHovered.png"),
                                                           "pressed": QPixmap("./assets/img/buttonSMPressed.png")}}}
        
        for button in self.buttonProperties.values():
            button["boundingBox"] = self.getHexBoundingBox(button["object"])
            button["state"] = "default"
            button["object"].setPixmap(button["pixmap"][button["state"]])
        
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
        for button in self.buttonProperties.values():
            if button["boundingBox"].containsPoint(event.pos(), Qt.OddEvenFill):
                if button["state"] != "hovered":
                    button["state"] = "hovered"
                    button["object"].setPixmap(button["pixmap"][button["state"]])
            else:
                if button["state"] != "default":
                    button["state"] = "default"
                    button["object"].setPixmap(button["pixmap"][button["state"]])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    hammer_head_main = HammerHeadMain()
    hammer_head_main.show()
    sys.exit(app.exec())