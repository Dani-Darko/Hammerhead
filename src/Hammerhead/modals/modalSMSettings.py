##############################################################################################
#                                                                                            #
#       HAMMERHEAD - Harmonic MOO Model Expedient for the Reduction                          #
#                    Hydraulic-loss/Heat-transfer Enhancement Asset Design                   #
#       Author: Daniela Minerva Segura Galeana                                               #
#       Supervised by: Antonio J. Gil                                                        #
#                                                                                            #
##############################################################################################
#                                                                                            #
#       SM Settings modal window                                                             #
#                                                                                            #
##############################################################################################


# IMPORTS: HAMMERHEAD files ###############################################################################################################

from layouts import layoutSMSettings                                            # Layouts -> SM Settings

# IMPORTS: PyQt5 ##########################################################################################################################

from PyQt5.QtCore import QTimer                                                 # PyQt5 -> Core Qt objects
from PyQt5.QtWidgets import QDialog                                             # PyQt5 -> Qt Widget objects

# IMPORTS: Others #########################################################################################################################

from typing import Any                                                          # Other -> Python type hinting

###########################################################################################################################################


class SMSettings(QDialog, layoutSMSettings.Ui_SMSettingsDialog):

    def __init__(self, parent):
        super(SMSettings, self).__init__(parent)
        self.setupUi(self)
