##############################################################################################
#                                                                                            #
#       HAMMERHEAD - Harmonic MOO Model Expedient for the Reduction                          #
#                    Hydraulic-loss/Heat-transfer Enhancement Asset Design                   #
#       Author: Daniela Minerva Segura Galeana                                               #
#       Supervised by: Antonio J. Gil                                                        #
#                                                                                            #
##############################################################################################
#                                                                                            #
#       HFM Settings modal window                                                            #
#                                                                                            #
##############################################################################################


# IMPORTS: HAMMERHEAD files ###############################################################################################################

from layouts import layoutHFMSettings                                           # Layouts -> HFM Settings

# IMPORTS: PyQt5 ##########################################################################################################################

from PyQt5.QtCore import QTimer                                                 # PyQt5 -> Core Qt objects
from PyQt5.QtWidgets import QDialog                                             # PyQt5 -> Qt Widget objects

# IMPORTS: Others #########################################################################################################################

from typing import Any                                                          # Other -> Python type hinting

###########################################################################################################################################


class HFMSettings(QDialog, layoutHFMSettings.Ui_HFMSettingsDialog):

    def __init__(self, parent):
        super(HFMSettings, self).__init__(parent)
        self.setupUi(self)
                
    def showEvent(self, event):
        """
        Override default showEvent, performing additional layout tasks after
            widget positions have been computed
        """
        self.groupBoxHFMParameters.setMaximumSize(                              # Set the maximum size of the HFM Parameters group box
            self.groupBoxHFMParameters.width(),                                 # ... the maximum width is the current width
            self.groupBoxHFMParameters.maximumSize().height())                  # ... the maximum height remains unchanged
        self.horizontalLayoutSaveAndReturn.insertStretch(0, 1)                  # Add a stretchable object to the "Save and Return" layout, pushing button to the right without expanding the layout
        super(HFMSettings, self).showEvent(event)                               # Carry out rest of the default showEvent tasks