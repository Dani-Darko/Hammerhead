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

from utilities.dataProcessing import loadyaml                                   # Utilities -> File loading and manipulation from ./resources directory
from utilities.dbFunctions import computeUniqueCases, computeYCoords, filterUniqueCases  # Utilities -> HFM database population tools

# IMPORTS: PyQt5 ##########################################################################################################################

from PyQt5.QtCore import QTimer, QThread                                        # PyQt5 -> Core Qt objects
from PyQt5.QtGui import QShowEvent                                              # PyQt5 -> Qt GUI interaction and drawing objects
from PyQt5.QtWidgets import QDialog, QGridLayout, QSpinBox, QStyle              # PyQt5 -> Qt Widget objects

# IMPORTS: Others #########################################################################################################################

from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes  # Others -> Matpotlib inset axes tools
from pathlib import Path                                                        # Path manipulation
from subprocess import Popen, DEVNULL                                           # Others -> External process manipulation
from typing import Any                                                          # Others -> Python type hinting

import matplotlib.pyplot as plt                                                 # Others -> Plotting tools
import numpy as np                                                              # Others -> Array manipulation
import sys                                                                      # Others -> System-specific tools

###########################################################################################################################################


class HFMSettings(QDialog, layoutHFMSettings.Ui_HFMSettingsDialog):

    def __init__(self, parent, hfmParams):
        super(HFMSettings, self).__init__(parent)
        self.setupUi(self)
                
        # Set icons for all buttons in HFM summary groupBox
        for button, icon in zip([self.pushButtonOpenCaseDatabase, self.pushButtonOpenIgnoredCases, self.pushButtonReloadSummary],
                                [QStyle.SP_DirIcon,               QStyle.SP_FileIcon,              QStyle.SP_BrowserReload]):
            button.setIcon(self.style().standardIcon(icon))

        # Define some useful variables
        self.ignoreCaseListFile = Path("./resources/ignoreCaseList.txt").resolve()  # Full path to ignoreCaseList file
        self.defaultIgnoreCaseListContent = "\n".join([                         # Default contents of "empty" ingoreCaseList.txt file
            "# List of parameter combinations that will be ignored (no simulations will run for all specified cases or their equivalents)",
            "# Each line specifies a single combination to be ignored, and must contain 5 values separated by whitespace (A1, A2, k1, k2, Re)",
            "# All lines starting with \"#\" are ignored; example entry:",
            "# 0.002 0.001 32 16 300",
            ""])
        self.openExec = "open" if sys.platform == "darwin" else "xdg-open"      # System-dependant executable used to open files/folders
                
        # Dictionary of all parameter spinBox objects by parameter and type
        self.paramSpinBoxes = {"A1": {"min": self.doubleSpinBoxA1Min, "max": self.doubleSpinBoxA1Max, "num": self.spinBoxA1Num},
                               "k1": {"min": self.spinBoxk1Min,       "max": self.spinBoxk1Max,       "num": self.spinBoxk1Num},
                               "A2": {"min": self.doubleSpinBoxA2Min, "max": self.doubleSpinBoxA2Max, "num": self.spinBoxA2Num},
                               "k2": {"min": self.spinBoxk2Min,       "max": self.spinBoxk2Max,       "num": self.spinBoxk2Num},
                               "Re": {"min": self.spinBoxReMin,       "max": self.spinBoxReMax,       "num": self.spinBoxReNum}}
                               
        self.hfmParams = loadyaml("hfmParams", override = hfmParams)            # Load default HFM parameters from ./resources/hfmParams.yaml (hfmParams from args take priority)
         # Set the initial values of the GUI HFM parameter spinBoxes based on the parameters specified in the YAML config file and passed via args
        for prefix in ["A1", "k1", "A2", "k2", "Re"]:                           # Iterate over all "prefixes" (HFM parameters)
            for suffix in ["min", "max", "num"]:                                # Iterate over all "suffixes" (min, max, num fields)
                spinBox = self.paramSpinBoxes[prefix][suffix]                   # Retrieve the corresponding spinBox object
                # Set the spinBox value - ensure only ints are passed to QSpinBoxes, and only floats are passed to QDoubleSpinBoxes
                spinBox.setValue( (int if isinstance(spinBox, QSpinBox) else float)(self.hfmParams[f"{prefix}_{suffix}"]) )
                               
        # Dictionary of all visualisation slider objects by parameter
        self.visualisationSliders = {"A1": self.sliderVisualisationA1,
                                     "k1": self.sliderVisualisationk1,
                                     "A2": self.sliderVisualisationA2,
                                     "k2": self.sliderVisualisationk2}
        
        # Visualisation label template HTML, to be filled in with every label update
        self.visualisationLabelTemplate = (                                     # paramPrefix -> "A" or "k"; paramSuffix -> "1" or "2"; fmt -> format string
            """<html><head/>
                 <body>
                   <p>{paramPrefix}<span style=" vertical-align:sub;">{paramSuffix} </span><span style=" font-size:10pt;">= {value:{fmt}}</span>
                   </p>
                 </body>
               </html>""")
        
        # Dictionary of all visualisation label objects and their corresponding format string, by parameter
        self.visualisationLabels = {"A1": {"obj": self.labelVisualisationA1, "fmt": ".3f"},
                                    "k1": {"obj": self.labelVisualisationk1, "fmt": ".0f"},
                                    "A2": {"obj": self.labelVisualisationA2, "fmt": ".3f"},
                                    "k2": {"obj": self.labelVisualisationk2, "fmt": ".0f"}}
        
        # Define all plot-relevant arrays
        self.xStructure = np.linspace(0, self.hfmParams["L"], self.hfmParams["dx"], endpoint=True)  # Array of x-coordinates of the microstructured pipe region
        self.xBefore = np.array([-self.hfmParams["L"], 0])                      # X-coordinates of unstructured pipe regions before xStructure
        self.xAfter = np.array([self.hfmParams["L"], self.hfmParams["L"] * 2])  # X-coordinates of unstructured pipe regions after xStructure
        self.yBottom = np.full(2, self.hfmParams["r"])                          # Y-coordinates of unstructured inner pipe region (corresponding to xBefore or xAfter) 
        self.yTop = np.full(self.xStructure.shape[0] + 4, self.hfmParams["r"] + 0.06)  # Y-coordinates of entire outer pipe region (whole length)
        
        # Define and setup outer plotting axes
        self.axPipe = self.widgetGeometryVisualisation.figure.add_subplot(111)  # Outer axes showing entire structured pipe region
        self.axPipe.set_xlim(-self.hfmParams["L"] * 0.05, self.hfmParams["L"] * 1.05)  # Define x-limits slightly larger than structured pipe region, so that unstructured pipe is visible
        self.axPipe.set_ylim((self.hfmParams["r"] + 0.06) / 2, self.hfmParams["r"] + 0.065)  # Define y-limits such that whole segment of pipe is visible in the top half of the plot
        self.axPipe.set_xlabel("Microstructured pipe section [m]")
        self.axPipe.set_ylabel("Pipe radius [m]")
        
        # Define and setup inner plotting axes
        self.axZoom = zoomed_inset_axes(self.axPipe, zoom=7, loc='upper left',  # Inner axes showing small equal-aspect structured pipe segment
                                        bbox_to_anchor=(0.14, 1.9), bbox_transform=self.axPipe.transAxes)  # Values chosen as they are visually appealing
        self.axZoom.set_aspect("equal")                                         # Inner axes has equal aspect ratio to show real shape of microstructure
        self.axZoom.set_xlim(0, self.hfmParams["L"] / 8)                        # Set x-limits to show an eighth of the structured pipe region
        self.axZoom.set_ylim(self.hfmParams["r"] - 0.0025, self.hfmParams["r"] + 0.0625)  # Set y-limits slightly above and below visible pipe region
        mark_inset(self.axPipe, self.axZoom, loc1=1, loc2=3, fc="none", ec="k", alpha=0.5, zorder=5)  # Highlight inner axis region on the outer axis
        
        # Define and connect threads        
        self.summaryUpdateWorker = HFMSummaryUpdateWorker(self.parent().domain, self.hfmParams)  # Thread for updating HFM summary groupBox content
        self.summaryUpdateWorker.finished.connect(self.summaryUpdateComplete)   # On summary update thread finish, update summary labels

        # Dictionary that will contain an array of requested parameter values for each parameter
        self.paramValues = {param: None for param in ["A1", "k1", "A2", "k2", "Re"]}  # Create all keys, leave values as None, they will be replaced by arrays in the next step
        for param in self.paramSpinBoxes.keys():                                # Iterate over all possible parameters ...
            self.updateParamValues(param)                                       # ... updating the parameter value arrays based on current spinBox values

        # Connect all spinBoxes to the relevant valueChanged functions (the use of lambda is so that the update functions are aware of which parameter group triggered the update)
        for param, fields in self.paramSpinBoxes.items():                       # Iterate over all possible parameters ...
            for field, spinBoxObj in fields.items():                            # Iterate over all spinBoxes for this parameter ...
                if field != "num":  # The change of either "min" or "max" should trigger the recomputation of "min" and "max" spinBox limits in that parameter group
                    spinBoxObj.valueChanged.connect(lambda value, param=param: self.updateSpinBoxLimits(param))
                elif param != "Re":  # The change of the "num" spinBox for all parameters except "Re" should trigger the update of the visualisation slider limits
                    spinBoxObj.valueChanged.connect(lambda value, param=param: self.updateSliderLimits(param, value))
                # The change of ANY spinBox in a parameter group should trigger the recomputation of the parameter value array for that parameter, and the update of the HFM summary group box
                spinBoxObj.valueChanged.connect(lambda value, param=param: self.updateParamValues(param))
                spinBoxObj.valueChanged.connect(lambda value, param=param, field=field: self.updateThreadHFMParams(param, field, value))
                    
        # Connect all visualisation sliders to the relevant valueChanged function
        for param, slider in self.visualisationSliders.items():                 # Iterate over all parameters except for "Re"
            # The change of any slider should trigger the update of the corresponding label text, as well as the geometry plot
            slider.valueChanged.connect(lambda index, param=param: self.updateSliderLabel(param, index))
            slider.valueChanged.connect(lambda index: self.updatePlot())
            
        self.pushButtonOpenCaseDatabase.pressed.connect(self.openCaseDatabase)  # Pressing "folder icon" button opens caseDatabase directory
        self.pushButtonOpenIgnoredCases.pressed.connect(self.openIgnoredCases)  # Pressing "file icon" button opens ignoreCaseList.txt file
        self.pushButtonReloadSummary.pressed.connect(self.startSummaryUpdate)   # Pressing "refresh icon" button starts the HFM summary update process
        self.pushButtonSaveAndReturn.pressed.connect(self.hide)                 # Pressing "save and return" button hides the dialog window
                
    def showEvent(self, event: QShowEvent) -> None:
        """
        Override default showEvent, performing additional layout tasks after
            widget positions have been computed
        
        Parameters
        ----------
        event : QShowEvent              event triggered by main window calling show() on this dialog
    
        Returns
        -------
        None
        """
        for fields in self.paramSpinBoxes.values():                             # Iterate over all parameters ...
            for spinBoxObj in fields.values():                                  # Iterate over all spinBoxes for this parameter ...
                spinBoxObj.setFixedSize(spinBoxObj.size())                      # ... fix their size to their current size (so a change in limits does not resize them)
                
        for label in [self.labelAvailableCasesNum, self.labelIgnoredCasesNum, self.labelTotalQueuedCasesNum, self.labelUniqueQueuedCasesNum]:
            label.setFixedSize(label.size())                                    # For all summary labels, fix their size to their maximum size ...
            label.setText("...")                                                # ... then set their text as unknown, values will be recomputed later
            
        for child in self.groupBoxSummary.children():                           # Iterate over all objects inside summary groupBox
            if isinstance(child, QGridLayout):                                  # Ignore the GridLayout object as its size cannot be changed
                continue
            child.setFixedHeight(self.pushButtonOpenCaseDatabase.height())      # Set fixed width of all objects as the tallest item for consistent spacing
                
        for fields in self.visualisationLabels.values():                        # Iterate over all parameters except for "Re" ...
            fields["obj"].setFixedSize(fields["obj"].size())                    # ... fix the size of all visualisation labels (so that changing the text does not resize the label)
            
        self.summaryUpdateWorker.domain = self.parent().domain                  # Propagate current domain type from main window (can be changed while window is hidden)
        self.startSummaryUpdate()                                               # Start the HFM summary update thread
        super(HFMSettings, self).showEvent(event)                               # Carry out rest of the default showEvent tasks
        
    def updateSpinBoxLimits(self, param: str) -> None:
        """
        Updates spinBox limits for "min" and "max" spinBoxes in the parameter
            group that triggered this update. Triggered by changing any "min"
            or "max" spinBoxes.
        
        Parameters
        ----------
        param : str                     relevant HFM parameter for this update
    
        Returns
        -------
        None
        """
        fields = self.paramSpinBoxes[param]                                     # Define shorthand, select only spinBoxes for this parameter
        fields["min"].setMaximum(fields["max"].value())                         # Set the maximum limit of the "min" spinBox to the current value of the "max" spinBox
        fields["max"].setMinimum(fields["min"].value())                         # Set the minimum limit of the "max" spinBox to the current value of the "min" spinBox

    def updateParamValues(self, param: str) -> None:
        """
        Updates array of parameter values based on spinBox "min", "max" and
            "num" values. Triggered by changing any spinBox.
        
        Parameters
        ----------
        param : str                     relevant HFM parameter for this update
    
        Returns
        -------
        None
        """
        fields = self.paramSpinBoxes[param]                                     # Define shorthand, select only spinBoxes for this parameter
        self.paramValues[param] = np.round(                                     # Compute array of parameter values rounded to 3 decimal places
            np.linspace(fields["min"].value(), fields["max"].value(), fields["num"].value()),  # Array containing "num" values between "min" and "max"
            decimals=3)                                                         # ... rounded to 3 decimal places
        if param != "Re":                                                       # If this parameter is not Re, we should also trigger the update of the relevant visualisation label text
            self.updateSliderLabel(param, self.visualisationSliders[param].value())  # Update the label for this parameter, and provide the current array index based on corresponding slider position
            self.updatePlot()                                                   # As parameters have been updated (but sliders have not moved), also manually trigger the update of the geometry plot
            
    def updateThreadHFMParams(self, param: str, field: str, value: int) -> None:
        """
        Updates the relevant HFM parameter of the hfmParams dict inside the 
            HFM summary updater thread
            
        Parameters
        ----------
        param : str                     relevant HFM parameter for this update
        field : str                     relevant HFM field (min/max/num) for this update
        value : int                     target number of slider positions, and indices in paramValues
    
        Returns
        -------
        None
        """
        self.summaryUpdateWorker.hfmParams[f"{param}_{field}"] = value
        self.startSummaryUpdate()
        
    def updateSliderLimits(self, param: str, value: int) -> None:
        """
        Updates the maximum slider position, such that the total number of
            available positions matches the corresponding "num" spinBox value.
            Triggered by changing any "num" spinBox that is not "Re".
        
        Parameters
        ----------
        param : str                     relevant HFM parameter for this update
        value : int                     target number of slider positions, and indices in paramValues
    
        Returns
        -------
        None
        """
        self.visualisationSliders[param].setMaximum(value - 1)                  # Set maximum spinBox position as value - 1, as the "0" position is the minimum
        
    def updateSliderLabel(self, param: str, index: int) -> None:
        """
        Updates the text of a visualisation label. Triggered by updating
            paramValues or moving any visualisation slider.
        
        Parameters
        ----------
        param : str                     relevant HFM parameter for this update
        index : int                     current slider position, and paramValues index
    
        Returns
        -------
        None
        """
        self.visualisationLabels[param]["obj"].setText(                         # Set the text of the relevant label ...
            self.visualisationLabelTemplate.format(                             # ... based on a template containing HTML formatting
                paramPrefix=param[0],                                           # ... the param prefix is the first param char ("A" or "k")
                paramSuffix=param[1],                                           # ... the param suffix is the subscripted second char (1 or 2)
                value=self.paramValues[param][int(index)],                      # ... the label value is fetched from the corresponding paramValues, based on its index
                fmt=self.visualisationLabels[param]["fmt"]))                    # ... the corresponding value formatting string (number of decimal places) is available in the label dict
                
    def updatePlot(self) -> None:
        """
        Updates the plotting region. Triggered by updating paramValues or
            moving any visualisation slider.
        
        Parameters
        ----------
        None
    
        Returns
        -------
        None
        """
        try:                                                                    # This will fail if paramValues are not fully defined yet (during __init__)
            A1, A2, k1, k2 = [self.paramValues[param][self.visualisationSliders[param].value()] for param in ["A1", "A2", "k1", "k2"]]  # Identify parameter values to plot
            yStructure = computeYCoords(self.xStructure, A1, A2, k1, k2, self.hfmParams["r"])  # Compute y-coordinates of the structured pipe region based on the HFM parameters
            for collection in self.axPipe.collections + self.axZoom.collections:  # Iterate over all collections (fill_between objects) in both axes ...
                collection.remove()                                             # ... remove each one as it cannot be edited and has to be redrawn
            for ax in [self.axPipe, self.axZoom]:                               # Draw identical content in both axes ...
                ax.fill_between(np.concatenate((self.xBefore, self.xStructure, self.xAfter)),  # ... plot a fill_between -> x-data is a concatenation of the structured and two unstructured pipe regions
                                np.concatenate((self.yBottom, yStructure, self.yBottom)),  # ... y-data is also a concatenation of three regions, but the before and after are identical
                                self.yTop, facecolor="#5EA5BB" if np.all(yStructure < self.hfmParams["r"] + 0.06) else "#DF2020")  # ... upper y-region is precomputed; if regions intersect, colour plot in red
            self.widgetGeometryVisualisation.figure.canvas.draw()               # Re-render all changes
        except TypeError:                                                       # During __init__, some paramValues will still be None, but a plot refresh is triggered
            pass                                                                # ... as they cannot happen once window is fully composited, simply ignore them
            
    def openCaseDatabase(self) -> None:
        """
        Opens the caseDatabase directory with the system's file manager. The
            directory is created if it doesn't exist.
        
        Parameters
        ----------
        None
    
        Returns
        -------
        None
        """
        caseDatabaseDir = Path(f"./caseDatabase/{self.parent().domain}").resolve()  # Full path to case database directory
        caseDatabaseDir.mkdir(parents=True, exist_ok=True)                      # Create directory if it doesn't already exist
        Popen([self.openExec, caseDatabaseDir], stdout=DEVNULL, stderr=DEVNULL)  # Open the case database directory with the system's file browser

    def openIgnoredCases(self) -> None:
        """
        Opens the ignoreCaseList.txt file with the system's default text
            editor. The file is created with default content if it doesn't
            exist.
        
        Parameters
        ----------
        None
    
        Returns
        -------
        None
        """
        if not self.ignoreCaseListFile.is_file():                               # Check whether the file ignoreCaseList.txt file exists
            with open(self.ignoreCaseListFile, "w") as f:                       # ... if it doesn't, create it and fill it with default content
                f.write(self.defaultIgnoreCaseListContent)
        Popen([self.openExec, self.ignoreCaseListFile], stdout=DEVNULL, stderr=DEVNULL)  # Open the file with the system's default text editor
        
    def startSummaryUpdate(self) -> None:
        """
        Start the HFM summary update process. This starts the HFM summary
            update worker thread if it isn't already running; if it is, the
            thread is marked stale and is requested to stop.
            
        Parameters
        ----------
        None
    
        Returns
        -------
        None
        """
        if self.summaryUpdateWorker.isRunning():                                # If the thread is already running ...
            self.summaryUpdateWorker.stale = True                               # ... mark it as stale (will need recomputing)
            self.summaryUpdateWorker.exit(1)                                    # ... request it to exit after the current event-loop blocking call
        else:                                                                   # If the thread is not running ...
            self.progressBarSummary.setMaximum(0)                               # ... start the progress bar
            self.summaryUpdateWorker.start()                                    # ... start execution of the thread
        
    def summaryUpdateComplete(self) -> None:
        """
        Triggered when the HFM summary update worker thread is completes,
            whether successfully or after being requested to stop.
        
        Parameters
        ----------
        None
    
        Returns
        -------
        None
        """
        if self.summaryUpdateWorker.stale:                                      # If the thread is marked as stale, it was requested to stop early ...
            self.startSummaryUpdate()                                           # ... restart it (now with updated information), and do nothing else
            return
        
        self.progressBarSummary.setMaximum(1)                                   # If we are here, thread completed successfully, stop the progress bar
        for label, threadAttr in zip([self.labelAvailableCasesNum, self.labelIgnoredCasesNum, self.labelTotalQueuedCasesNum, self.labelUniqueQueuedCasesNum],
                                     ["availableCases",            "ignoredCases",            "queuedCasesTotal",            "queuedCasesUnique"]):
            label.setText(str(getattr(self.summaryUpdateWorker, threadAttr)))   # ... update labels in HFM summary groupBox with computed values

        
class HFMSummaryUpdateWorker(QThread):
    """
    Thread for computing counters used to updated labels in the HFM summary
        group box
    """

    def __init__(self, domain, hfmParams):
        super(HFMSummaryUpdateWorker, self).__init__()
                
        self.domain = domain
        self.hfmParams = hfmParams
        self.stale = True
        
        self.availableCases = 0                                                 # Number of existing cases in the caseDatabase
        self.ignoredCases = 0                                                   # Number of unique entries in the ignoreCaseList
        self.queuedCasesTotal = 0                                               # Total non-unique cases for database population
        self.queuedCasesUnique = 0                                              # Total unique cases for database population
    
    def run(self):
        self.stale = False                                                      # This flag will be set if anything changes before the thread can complete
        
        _, uniqueCases = computeUniqueCases(self.hfmParams)                     # Compute list of unique cases
        filteredUniqueCases, self.ignoredCases = filterUniqueCases(uniqueCases)  # Filter out unique cases based on ignoreCaseList file (and compute it's length)
        
        self.availableCases = len(list(Path(f"./caseDatabase/{self.domain}").glob("Re_*")))  # Update number of available cases as the number of valid folders in the caseDatabase directory
        self.queuedCasesTotal = sum([len(uniqueCase) - 1 for uniqueCase in filteredUniqueCases])  # Compute number of unique cases as the total number of parameter combination entries in filteredUniqueCases
        self.queuedCasesUnique = len(filteredUniqueCases)                       # Number of unique cases is simply the length of the filterUniqueCases list (each entry is a unique set of parameter combinations)
        
        for attr in ["availableCases", "ignoredCases", "queuedCasesTotal", "queuedCasesUnique"]:
            if getattr(self, attr) > 99999:                                     # Cap the maximum count for counter to 99,999
                setattr(self, attr, "99999+")                                   # ... otherwise replace it with the "maximum" string