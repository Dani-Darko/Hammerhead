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

from PyQt5.QtCore import QPoint, Qt, QTimer                                     # PyQt5 -> Core Qt objects
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
                                 "buttonPP"   : {"object" : self.labelButtonPP},
                                 "buttonStart": {"object" : self.labelButtonStart}}
        
        # Process all button objects, setting default states, applying pixmaps and computing bounding boxes
        for label, button in self.buttonProperties.items():
            button["boundingBox"] = self.getHexBoundingBox(button["object"])    # Get hexagonal bounding box represented by the hexagonal pixmap
            button["state"] = "default"                                         # Set button state to default
            button["pixmap"] = {}                                               # Create empty pixmap dictionary, this will be filled in with all available button state pixmaps
            for state in ["default", "hovered", "pressed"]:                     # Iterate over all available button states
                button["pixmap"][state] = QPixmap(f"./assets/images/{label}{state.title()}.png")  # Load pixmap from assets directory (example: ./assets/img/buttonGraphDefault.png)
            button["object"].setPixmap(button["pixmap"][button["state"]])       # Set pixmap representing the current (default) button state
            
        # Setup sprite-related objects (sprite QPixmaps, animation timer, state counter)
        self.sprites = {state: QPixmap(f"./assets/images/HammerheadMascotSprite{state}.png") for state in range(2)}  # Load all relevant sprite images as QPixmaps
        self.spriteState = 0                                                    # Keeps track of current sprite state
        self.spriteTimer = QTimer(interval=1000)                                # Sprite animation timer, fires every 1 second
        self.spriteTimer.timeout.connect(self.updateSprite)                     # Timer fire triggers sprite update function
        self.updateSprite()                                                     # Call initial sprite update function (setting the label pixmap for the first time)
        
        ## Setup dialog-related objects (dialog box, dialog content label, incremental text update timer)
        self.labelDialogBox.setPixmap(QPixmap(f"./assets/images/HammerheadDialogBig.png"))  # Update dialog box with pixmap
        self.dialogText = (                                                     # Initial dialog text (welcome message with instructions)
            "Welcome to Hammerhead - a tool for CFD and ML-based optimisation "
            "of heat transfer in pipe flows. Hover over the buttons below for "
            "more information.")
        self.dialogTextIndex = 0                                                # Set initial incremental text index to zero (initially, no message is shown)
        self.dialogIncrementTimer = QTimer(interval=20)                         # Timer which triggers incremental update of text every 20ms
        self.dialogIncrementTimer.timeout.connect(self.incrementalUpdateDialogText)  # With every timer trigger, one more character is added to the dialog
        
        # Start all timers and show UI
        self.spriteTimer.start()                                                # Sprite animation timer (always active)
        self.dialogIncrementTimer.start()                                       # Incremental dialog text update timer (disabled when message is fully shown in dialog box)
        
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
                
    def updateSprite(self) -> None:
        """
        Updates animated sprite, periodically called by timer
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        self.labelAnimationMascot.setPixmap(self.sprites[self.spriteState])     # Set animation label pixmap corresponding to current sprite state
        self.spriteState = (self.spriteState + 1) % len(self.sprites)           # Increment sprite state (up to a maximum defined by the number of available sprite states, then restarts from zero)
        
    def incrementalUpdateDialogText(self) -> None:
        """
        Incrementally updates dialog text for a game-like text animation.
            Triggered by external timer which is stopped once entire message
            is visible in the dialog box.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        self.labelDialogContent.setText(f"<font color=#6b6b6b>{self.dialogText[:self.dialogTextIndex]}</font>")  # Update label with partial text (coloured using HTML)
        self.dialogTextIndex += 1                                               # Increment text index, such that next update draws one more character
        if self.dialogTextIndex > len(self.dialogText):                         # Once the target text index is greater than the total length of the text ...
            self.dialogIncrementTimer.stop()                                    # ... stop the timer, all text has been drawn


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
