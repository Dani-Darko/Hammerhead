from PyQt5.QtWidgets import QMainWindow, QApplication
import main_window
import sys

class MainWindow(QMainWindow, main_window.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec())