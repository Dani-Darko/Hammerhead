# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'layoutMain.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 800)
        MainWindow.setMouseTracking(True)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMouseTracking(True)
        self.centralwidget.setObjectName("centralwidget")
        self.labelButtonHFM = QtWidgets.QLabel(self.centralwidget)
        self.labelButtonHFM.setGeometry(QtCore.QRect(20, 460, 100, 115))
        self.labelButtonHFM.setMouseTracking(True)
        self.labelButtonHFM.setText("")
        self.labelButtonHFM.setObjectName("labelButtonHFM")
        self.groupboxDynamicRegion = QtWidgets.QGroupBox(self.centralwidget)
        self.groupboxDynamicRegion.setGeometry(QtCore.QRect(20, 20, 560, 420))
        self.groupboxDynamicRegion.setTitle("")
        self.groupboxDynamicRegion.setObjectName("groupboxDynamicRegion")
        self.labelButtonSM = QtWidgets.QLabel(self.centralwidget)
        self.labelButtonSM.setGeometry(QtCore.QRect(74, 550, 100, 115))
        self.labelButtonSM.setMouseTracking(True)
        self.labelButtonSM.setText("")
        self.labelButtonSM.setObjectName("labelButtonSM")
        self.labelButtonPred = QtWidgets.QLabel(self.centralwidget)
        self.labelButtonPred.setGeometry(QtCore.QRect(47, 675, 50, 58))
        self.labelButtonPred.setMouseTracking(True)
        self.labelButtonPred.setText("")
        self.labelButtonPred.setObjectName("labelButtonPred")
        self.labelButtonGraph = QtWidgets.QLabel(self.centralwidget)
        self.labelButtonGraph.setGeometry(QtCore.QRect(20, 722, 50, 58))
        self.labelButtonGraph.setMouseTracking(True)
        self.labelButtonGraph.setText("")
        self.labelButtonGraph.setObjectName("labelButtonGraph")
        self.labelButtonOptim = QtWidgets.QLabel(self.centralwidget)
        self.labelButtonOptim.setGeometry(QtCore.QRect(74, 722, 50, 58))
        self.labelButtonOptim.setMouseTracking(True)
        self.labelButtonOptim.setText("")
        self.labelButtonOptim.setObjectName("labelButtonOptim")
        self.labelButtonStart = QtWidgets.QLabel(self.centralwidget)
        self.labelButtonStart.setGeometry(QtCore.QRect(530, 722, 50, 58))
        self.labelButtonStart.setMouseTracking(True)
        self.labelButtonStart.setText("")
        self.labelButtonStart.setObjectName("labelButtonStart")
        self.labelButtonSetup = QtWidgets.QLabel(self.centralwidget)
        self.labelButtonSetup.setGeometry(QtCore.QRect(476, 722, 50, 58))
        self.labelButtonSetup.setMouseTracking(True)
        self.labelButtonSetup.setText("")
        self.labelButtonSetup.setObjectName("labelButtonSetup")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "HammerHead"))
