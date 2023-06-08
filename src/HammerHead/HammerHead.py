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
                        
        self.buttonProperties = {"buttonHFM"  : {"object" : self.labelButtonHFM},
                                 "buttonSM"   : {"object" : self.labelButtonSM},
                                 "buttonPred" : {"object" : self.labelButtonPred},
                                 "buttonGraph": {"object" : self.labelButtonGraph},
                                 "buttonOptim": {"object" : self.labelButtonOptim},
                                 "buttonSetup": {"object" : self.labelButtonSetup},
                                 "buttonStart": {"object" : self.labelButtonStart}}
        
        for label, button in self.buttonProperties.items():
            button["boundingBox"] = self.getHexBoundingBox(button["object"])
            button["state"] = "default"
            button["pixmap"] = {}
            for state in ["default", "hovered", "pressed"]:
                button["pixmap"][state] = QPixmap(f"./assets/img/{label.title()}{state}.png")
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
        
    @staticmethod
    def updateButtonState(button, state):
        if button["state"] != state:
            button["state"] = state
            button["object"].setPixmap(button["pixmap"][button["state"]])
    
    def mouseMoveEvent(self, event):
        for button in self.buttonProperties.values():
            if button["state"] == "pressed":
                return
            if button["boundingBox"].containsPoint(event.pos(), Qt.OddEvenFill):
                self.updateButtonState(button, "hovered")
            else:
                self.updateButtonState(button, "default")
                    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            for button in self.buttonProperties.values():
                if button["state"] == "hovered":
                    self.updateButtonState(button, "pressed")
                    return

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            for button in self.buttonProperties.values():
                if button["state"] == "pressed":
                    if button["boundingBox"].containsPoint(event.pos(), Qt.OddEvenFill):
                        print("EVENT TRIGGERED")
                        self.updateButtonState(button, "hovered")
                    else:
                        self.updateButtonState(button, "default")
                    return


if __name__ == "__main__":
    app = QApplication(sys.argv)
    hammer_head_main = HammerHeadMain()
    hammer_head_main.show()
    sys.exit(app.exec())