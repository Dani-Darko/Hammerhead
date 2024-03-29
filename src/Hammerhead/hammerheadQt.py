##############################################################################################
#                                                                                            #
#       HAMMERHEAD - Harmonic MOO Model Expedient for the Reduction                          #
#                    Hydraulic-loss/Heat-transfer Enhancement Asset Design                   #
#       Author: Daniela Minerva Segura Galeana                                               #
#       Supervised by: Antonio J. Gil                                                        #
#                                                                                            #
##############################################################################################
#                                                                                            #
#       Main Hammerhead GUI controller script. This is called directly from hammerhead.py    #
#           with all relevant arguments, and is used to construct and manage the main        #
#           Hammerhead window.                                                               #
#                                                                                            #
##############################################################################################


# IMPORTS: HAMMERHEAD files ###############################################################################################################

from layouts import layoutMain                                                  # Layouts -> Main GUI window

# IMPORTS: PyQt5 ##########################################################################################################################

from PyQt5.QtCore import QPoint, Qt                                             # PyQt5 -> Core Qt objects
from PyQt5.QtGui import QIcon, QMouseEvent, QPixmap, QPolygon                   # PyQt5 -> Qt GUI interaction and drawing objects
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow                   # PyQt5 -> Qt Widget objects

# IMPORTS: Others #########################################################################################################################

from typing import Any                                                          # Other -> Python type hinting
from argparse import Namespace                                                  # Other -> Object containing argparse parsed arguments

import sys                                                                      # Other -> Misc system tools

###########################################################################################################################################


class HammerHeadMain(QMainWindow, layoutMain.Ui_MainWindow):

    def __init__(self, args, parent=None):
        super(HammerHeadMain, self).__init__(parent)
        self.setupUi(self)
        self.setFixedSize(self.size())                                          # Disable resizing or maximizing window
                        
        # Dictionary of button properties that will contain "button" objects (labels), their bounding box, current state, and all available pixmaps (dynamically filled within this function)
        self.buttonProperties = {"buttonHFM"  : {"object" : self.labelButtonHFM},
                                 "buttonSM"   : {"object" : self.labelButtonSM},
                                 "buttonPred" : {"object" : self.labelButtonPred},
                                 "buttonGraph": {"object" : self.labelButtonGraph},
                                 "buttonOptim": {"object" : self.labelButtonOptim},
                                 "buttonSetup": {"object" : self.labelButtonSetup},
                                 "buttonStart": {"object" : self.labelButtonStart}}
        
        # Process all button objects, setting default states, applying pixmaps and computing bounding boxes
        for label, button in self.buttonProperties.items():
            button["boundingBox"] = self.getHexBoundingBox(button["object"])    # Get hexagonal bounding box represented by the hexagonal pixmap
            button["state"] = "default"                                         # Set button state to default
            button["pixmap"] = {}                                               # Create empty pixmap dictionary, this will be filled in with all available button state pixmaps
            for state in ["default", "hovered", "pressed"]:                     # Iterate over all available button states
                button["pixmap"][state] = QPixmap(f"./assets/images/{label}{state.title()}.png")  # Load pixmap from assets directory (example: ./assets/img/buttonGraphDefault.png)
            button["object"].setPixmap(button["pixmap"][button["state"]])       # Set pixmap representing the current (default) button state
        
    @staticmethod   
    def getHexBoundingBox(obj: QLabel) -> QPolygon:
        """
        Compute a hexagonal bounding box for a QLabel "button" object,
            representing the underlying hexagonal pixmap
        
        Parameters
        ----------
        obj : QLabel                    Qt label object representing a pseudo-button
    
        Returns
        -------
        boundingBox : QPolygon          hexagonal "clickable area" bounding box
        """
        x, y, w, h = obj.pos().x(), obj.pos().y(), obj.width(), obj.height()    # Define shorthands for QLabel x and y position within the main window as well as its width and height
        a = QPoint(x + (w // 2), y)                                             # Compute six hexagonal vertices from the top, clockwise (this is the top-most vertex)
        b = QPoint(x + w,        y + (h // 4))                                  # ... upper right vertex
        c = QPoint(x + w,        y + (3 * (h // 4)))                            # ... lower right vertex
        d = QPoint(x + (w // 2), y + h)                                         # ... bottom vertex
        e = QPoint(x,            y + (3 * (h // 4)))                            # ... lower left vertex
        f = QPoint(x,            y + (h // 4))                                  # ... upper left vertex
        return QPolygon([a, b, c, d, e, f])                                     # Construct and return a QPolygon from the six hexagonal vertices
        
    @staticmethod
    def updateButtonState(button: dict[str, Any], state: str) -> None:
        """
        Convenience function for updating button state (or ignoring duplicate
            button update requests)
         
        Parameters
        ----------
        button : dict                   dictionary of button properties
        state : str                     target state of the button
        
        Returns
        -------
        None
        """
        if button["state"] != state:                                            # If the new button state differs from the current state (valid change)
            button["state"] = state                                             # ... update the button state
            button["object"].setPixmap(button["pixmap"][button["state"]])       # ... update the active pixmap representing the new button state
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """
        Overrides default mouse move event, triggered on mouse movement
            anywhere within the main window and all tracked child widgets.
            If movement over a button bounding box occurs, updates the 
            relevant button state (unless the button is held down).
        
        Parameters
        ----------
        event : QMouseEvent             mouse event supplied by Qt
        
        Returns
        -------
        None
        """
        for button in self.buttonProperties.values():                           # Iterate over all buttons
            if button["state"] == "pressed":                                    # If the button is pressed, ignore it (here we deal with default or hover states only)
                return                                                          # No need to iterate over any other buttons, if one is pressed no other buttons can be interacted with
            if button["boundingBox"].containsPoint(event.pos(), Qt.OddEvenFill):  # If mouse is hovered over the current button's bounding region
                self.updateButtonState(button, "hovered")                       # ... update button state to hovered (this will be ignored by updateButtonState if button is already in this state)
            else:                                                               # Otherwise, if mouse is not over the current button's bounding box
                self.updateButtonState(button, "default")                       # ... set it's state to default (this will be ignored by updateButtonState if button is already in this state)
                    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """
        Overrides default mouse press event, triggered on click of any mouse
            button within the main window and all tracked child widgets.
            If a button is currently in hovered state, this will "lock" it
            into a pressed state.
        
        Parameters
        ----------
        event : QMouseEvent             mouse event supplied by Qt
        
        Returns
        -------
        None
        """
        if event.button() != Qt.LeftButton:                                     # Only allow left mouse clicks, any other mouse buttons will be ignored
            return                                                              # ... no need to process event further as this was not a left-click
        for button in self.buttonProperties.values():                           # Iterate over all buttons
            if button["state"] == "hovered":                                    # ... if a button is currently in hovered state, it is eligible to be set to pressed
                self.updateButtonState(button, "pressed")                       # ... update button state to pressed (this will "lock" the button in this state as mouse move events will now be ignored)
                return                                                          # ... no need to examine other buttons as only one button can be hovered-over at a time

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """
        Overrides default mouse release event, triggered on release of any
            mouse button within the main window and all tracked child widgets.
            If a button is currently in pressed state, this will release it
            from that state. If the mouse release occurred within the bounding
            box of the button, trigger the corresponding event.
        
        Parameters
        ----------
        event : QMouseEvent             mouse event supplied by Qt
        
        Returns
        -------
        None
        """
        if event.button() != Qt.LeftButton:                                     # Only allow left mouse clicks, any other mouse buttons will be ignored
            return                                                              # ... no need to process event further as this was not a left-click
        for button in self.buttonProperties.values():                           # Iterate over all buttons
            if button["state"] == "pressed":                                    # If a button is currently in pressed state, it is eligible to be "released" into hovered or default state
                if button["boundingBox"].containsPoint(event.pos(), Qt.OddEvenFill):  # ... if the release event happened within the bounding box of the pressed button
                    print("EVENT TRIGGERED")                                    # ... trigger the relevant event
                    self.updateButtonState(button, "hovered")                   # ... return the button to hovered state as mouse is still over the button
                else:                                                           # ... if the release event happened away from the button, treat this as a mis-click (no action is performed)
                    self.updateButtonState(button, "default")                   # ... return button default state as it is no longer being hovered-over
                return                                                          # ... no need to examine other buttons as only one button can be clicked at a time


def launch_gui(args: Namespace) -> None:
    """
    Helper function that creates Qt application, attaches the Hammerhead main
        UI window to it, and manages the execution of the GUI until it is
        manually closed

    Parameters
    ----------
    args : Namespace                    argparse object containing parsed arguments

    Returns
    -------
    None
    """
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('./assets/images/hammeheadMascot.png'))
    hammerHeadMain = HammerHeadMain(args)
    hammerHeadMain.show()
    sys.exit(app.exec())

###############################################################################
    
if __name__ == "__main__":
    raise RuntimeError("This script is not intended for execution. Instead, execute hammehead.py from the parent directory.")
