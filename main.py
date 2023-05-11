from PyQt5.QtWidgets import QMainWindow, QApplication
import main_window
import sys

class MainWindow(QMainWindow, main_window.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.doSomething.clicked.connect(self.something)
        self.SlidyBoi.valueChanged.connect(self.update_spinny_boi)

    def something(self):
        self.lol.setText("SOMETHING IS HAPPENING. Oh. It didn't.")
    def update_spinny_boi(self, value):
        self.spinyBoi.setValue(value)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec())