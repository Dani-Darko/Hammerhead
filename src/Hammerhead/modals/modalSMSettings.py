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

from utilities.dataProcessing import loadyaml                                   # Utilities -> File loading and manipulation from ./resources directory

# IMPORTS: PyQt5 ##########################################################################################################################

from PyQt5.QtCore import Qt                                                     # PyQt5 -> Core Qt objects
from PyQt5.QtGui import QBrush, QColor, QShowEvent                              # PyQt5 -> Qt GUI interaction and drawing objects
from PyQt5.QtWidgets import QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QGroupBox, QListWidget, QListWidgetItem, QPushButton, QSpinBox  # PyQt5 -> Qt Widget objects

# IMPORTS: Others #########################################################################################################################

from pathlib import Path                                                        # Others -> Path manipulation
from typing import Any                                                          # Other -> Python type hinting

import torch                                                                    # Others -> Tensor manipulation functions

###########################################################################################################################################


class SMSettings(QDialog, layoutSMSettings.Ui_SMSettingsDialog):

    def __init__(self, parent, noTensor, train, search, trainingParams, availableKernelsGP):
        super(SMSettings, self).__init__(parent)
        self.setupUi(self)

        # Update UI elements based on command line arguments
        self.checkBoxTensorUpdate.setChecked(not noTensor)                      # enable/disable tensorUpdate groupBox based on --noTensor
        for model in train:                                                     # iterate over all models in --train
            getattr(self, f"checkBox{model}").setChecked(True)                  # ... enabling them as specified (all unspecified are disabled)
        self.checkBoxOptimalSearch.setChecked(len(search) > 0)                  # enable optimalSearch if any models in --search are specified

        self.comboBoxGPKernel.insertItems(0, sorted(availableKernelsGP))        # update comboBox of all GP kernels to include all previously discovered kernels
        
        self.trainingParams = loadyaml("trainingParams", override = trainingParams)  # Load default training parameters from ./resources/trainingParams.yaml (trainingParams from args take priority)
        # Set the initial values of the GUI training parameter fields based on the parameters specified in the YAML config file and passed via args
        for param in ["modes", "samples"]:                                      # Iterate over all int-type params
            getattr(self, f"spinBox{param.capitalize()}").setValue(int(self.trainingParams[param]))  # Set spinBox values based on config/cmd-args
        for param in ["valSplits", "RBFKernels", "NNLayers", "NNNeurons", "GPKernels"]:  # Iterate over all list-type params
            listWidget = getattr(self, f"listWidget{param[0].upper() + param[1:]}")  # Resolve listWidget object based on name (first letter uppercase)
            listWidget.clear()                                                  # Clear listWidget, all values will be replaced with values specified in config/cmd-args
            for val in self.trainingParams[param]:                              # Iterate over all items in the current param
                self.addToList(val, listWidget)                                 # ... add each item to the listWidget

        # For the following combination of buttons, sources and destinations, connect pushButtons to listWidget add/clear actions
        for buttonAdd, buttonClear, source, destination in [
            [self.pushButtonValSplitAddToList,  self.pushButtonValSplitClearList,  self.doubleSpinBoxValSplit, self.listWidgetValSplits ],
            [self.pushButtonRBFKernelAddToList, self.pushButtonRBFKernelClearList, self.comboBoxRBFKernel,     self.listWidgetRBFKernels],
            [self.pushButtonNNLayersAddToList,  self.pushButtonNNLayersClearList,  self.spinBoxNNLayers,       self.listWidgetNNLayers  ],
            [self.pushButtonNNNeuronsAddToList, self.pushButtonNNNeuronsClearList, self.spinBoxNNNeurons,      self.listWidgetNNNeurons ],
            [self.pushButtonGPKernelAddToList,  self.pushButtonGPKernelClearList,  self.comboBoxGPKernel,      self.listWidgetGPKernels ]]:
            buttonAdd.pressed.connect(lambda *args, source=source, destination=destination: self.addToList(source, destination))
            buttonClear.pressed.connect(lambda *args, destination=destination: destination.clear())

        # For each of the following objects, identify "update" actions for each object type and connect it to updateSummaryText function
        for obj in [self.groupBoxParameters, self.spinBoxModes, self.spinBoxSamples, self.pushButtonValSplitAddToList, self.pushButtonValSplitClearList,
                    self.checkBoxLRBF, self.checkBoxMRBF, self.checkBoxSRBF, self.pushButtonRBFKernelAddToList, self.pushButtonRBFKernelClearList,
                    self.checkBoxLNN, self.checkBoxMNN, self.checkBoxSNN, self.pushButtonNNLayersAddToList, self.pushButtonNNLayersClearList, self.pushButtonNNNeuronsAddToList, self.pushButtonNNNeuronsClearList,
                    self.checkBoxLGP, self.checkBoxMGP, self.checkBoxSGP, self.pushButtonGPKernelAddToList, self.pushButtonGPKernelClearList,
                    self.checkBoxTensorUpdate, self.checkBoxOptimalSearch]:
            match obj:
                case QCheckBox() | QGroupBox():
                    obj.toggled.connect(self.updateSummaryText)
                case QSpinBox():
                    obj.valueChanged.connect(self.updateSummaryText)
                case QPushButton():
                    obj.pressed.connect(self.updateSummaryText)

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
        for listWidget in [self.listWidgetValSplits, self.listWidgetRBFKernels, self.listWidgetNNLayers, self.listWidgetNNNeurons, self.listWidgetGPKernels]:
            listWidget.setSpacing(1)                                            # For each listWidget, set item spacing after compositing
        self.updateSummaryText()                                                # Write summary text, as UI has now been updated with values from config/cmd-args
        super(SMSettings, self).showEvent(event)                                # Carry out rest of the default showEvent tasks
        
    @staticmethod
    def enumerateCaseDatabse(domain: str) -> dict[int, int]:
        """
        Scans the caseDatabase directory, counting the number of cases for each
            Re in the active domain

        Parameters
        ----------
        domain : str                    active domain (2D or axisym)

        Returns
        -------
        databaseRe : dict[int, int]     number of cases per Re in the caseDatabase {Re: count}
        """
        databaseRe = {}                                                         # Empty dictionary for counting cases in caseDatabase {Re: count}
        for case in Path(f"./caseDatabase/{domain}").glob("Re_*"):              # Iterate over valid case directories in the caseDatabase
            Re = int(case.name.split("_")[1])                                   # Identify Re of the current case
            if Re in databaseRe:                                                # If Re exists in the dictionary
                databaseRe[Re] += 1                                             # ... increment the case counter
            else:                                                               # Otherwise, if Re doesn't exist
                databaseRe[Re] = 1                                              # ... create an entry with one count
        return databaseRe
        
    @staticmethod
    def enumerateTensors(domain: str, modes: int) -> dict[int, int]:
        """
        Scans the mlData directory, counting the number of cases in the tensor
            for each Re in the active domain and current number of modes

        Parameters
        ----------
        domain : str                    active domain (2D or axisym)
        modes : int                     number of modes

        Returns
        -------
        tensorsRe : dict[int, int]      number of cases per Re in mlData tensors {Re: count}
        """
        tensorsRe = {}                                                          # Empty dictionary for counting cases in mlData tensors {Re: count}
        for tensorDir in Path(f"./mlData/{domain}").glob(f"Re_*_modes_{modes}_harmonics_2"):  # iterate over each tensor directory matching the selected domain and number of modes
            try:                                                                # Attempt to extract case count from tensor (will fail for Re All, but that tensor is not needed)
                Re = int(tensorDir.name.split("_")[1])                          # Identify Re from tensorDir name
                tensorsRe[Re] = torch.load(tensorDir / "xData.pt")["xExpanded"].shape[0]  # Identify case count from xData tensor, add it to dictionary
            except TypeError:                                                   # This will fail for the Re All tensor
                pass                                                            # ... ignore this tensor, as it should only contain the cases already counted
        return tensorsRe
    
    @staticmethod
    def addToList(source: Any, destination: QListWidget) -> None:
        """
        Converts data from "source" into a listWidgetItem, and adds it to the
            "destination" listWidget

        Parameters
        ----------
        source : Any                    any object containing "source" data
        destination : QListWidget       "destination" listWidget for "source" item

        Returns
        -------
        None
        """
        match source:                                                           # Attempt to extract content from "source" object
            case QDoubleSpinBox() | QSpinBox():                                 # If "source" is any spinBox
                content = source.cleanText()                                    # ... get the value as text with any prefixes or suffixes
            case QComboBox():                                                   # If "source" is a comboBoc
                content = source.currentText()                                  # ... get the text of the currently selected item
            case _:                                                             # Otherwise
                content = str(source)                                           # ... convert object to string
                
        if content == "" or content in [destination.item(i).text() for i in range(destination.count())]:
            return                                                              # If "content" is empty or already exists in the listWidget, do nothing
                
        item = QListWidgetItem(content, destination)                            # Create listWidgetItem in the destination listWidget with the "content" as text
        item.setBackground(QBrush(QColor(107, 107, 107, 50), Qt.SolidPattern))  # Set item background to match the current item style
        
    def updateSummaryText(self) -> None:
        """
        Computes and updates the summary text

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        databaseRe = self.enumerateCaseDatabse(self.parent().domain)            # Get the number of cases in the caseDatabase
        tensorsRe = self.enumerateTensors(self.parent().domain, self.spinBoxModes.value())  # Get the number of cases in mlData tensors
        tensorUpdate = self.checkBoxTensorUpdate.isChecked()                    # True if tensor update is set to run
        optimalSearch = self.checkBoxOptimalSearch.isChecked()                  # True if optimal search is set to run
        noHFM = not self.parent().dialogHFMSettings.groupBoxHFMParameters.isChecked()  # True if database population in HFM modal is not set to run

        def _genSubtext(model: str,
                        modelName: str,
                        archParamCombn: int,
                        archParamCombnName: str
                       ) -> str:
            """
            Private helper function for generating the text for each model in
                the summary text

            Parameters
            ----------
            model : str                 model acronym (RBF, NN, GP)
            modelName : str             full model name
            archParamCombn : int        number of chosen parameter combinations for this model (from UI)
            archParamCombnName : str    name for parameter combinations of this model (kernel, layer-neuron combination)

            Returns
            -------
            subtext : str               generated text for the target model
            """
            # Identify which of the model architectures are enabled (lumped, modal or spatial) in the UI
            activeArch = [arch for arch in ["lumped", "modal", "spatial"] if getattr(self, f"checkBox{arch[0].upper()}{model}").isChecked()]
            return ("" if len(activeArch) == 0 else                             # String is empty if no architectures are enabled
                    f"""> {modelName} ({model}) will train with {archParamCombn} {archParamCombnName}{"" if archParamCombn == 1 else "s"} {
                          '(<b><span style="color: #E74856;">INVALID</span></b>) ' if archParamCombn == 0 else ''
                        }for the following dimension outputs (total of {
                          self.spinBoxSamples.value() * self.listWidgetValSplits.count() * len(activeArch) * archParamCombn
                        } tasks):<br>&nbsp;&nbsp;""" + str(activeArch) + "<br><br>")

        sm_string = (f"""> Surrogate model training is <b><span style="color: #E74856;">DISABLED</span></b> and is not queued to run. To enable it, check the "Run surrogate model training" checkbox."""
                     if not self.groupBoxParameters.isChecked() else
                     f"""> Surrogate model training is <b><span style="color: #13A10E;">ENABLED</b></span> with:
                         <br>
                         &nbsp;&nbsp;<b>{self.spinBoxSamples.value()} random sample{"" if self.spinBoxSamples.value() == 1 else "s"}</b> per architecture
                         <br>
                         &nbsp;&nbsp;<b>{self.listWidgetValSplits.count()} validation split{"" if self.listWidgetValSplits.count() == 1 else "s"}</b> per architecture
                         {'(<b><span style="color: #E74856;">INVALID</span></b>)' if self.listWidgetValSplits.count() == 0 else ""}""")

        header = f"""> Hammerhead is running in <b>{self.parent().domain}</b> mode.
                     <br><br>
                     > Tensors <b><span style="color: {"#13A10E" if tensorUpdate else "#E74856"};">WILL {"" if tensorUpdate else "NOT "}BE</b> recomputed from the case database.
                       To {"disable" if tensorUpdate else "enable"} this, {"uncheck" if tensorUpdate else "check"} the "Update tensors from HFM case database" checkbox.
                     <br><br>
                     {sm_string}
                     <br><br>
                     > Optimal search <b><span style="color: {"#13A10E" if optimalSearch else "#E74856"};">WILL {"" if optimalSearch else "NOT "}BE</b> performed
                        on existing models to find the shape parameters resulting in maximum THP.
                       To {"disable" if optimalSearch else "enable"} this, {"uncheck" if optimalSearch else "check"} the "Run optimal search" checkbox."""
        
        if self.groupBoxParameters.isChecked():                                 # Case 1: Surrogate model is enabled
            self.textEditSummary.setText(                                       # ... generate relevant text for training information and tasks to be executed
                f"""<html><head/>
                     <body style="color: #CCCCCC; background-color: #0C0C0C;">
                       <p>
                         {header}
                         <br><br>
                         {"-"*64}
                         <br><br>
                         > There are currently {sum(list(databaseRe.values()))} cases in the case database:
                         <br>
                         &nbsp;&nbsp;Re = {sorted(databaseRe.keys())}
                         <br>
                         > HFM case database population is <b><span style="color: {"#13A10E" if not noHFM else "#E74856"};">{"ENABLED" if not noHFM else "DISABLED"}</b>{
                            f" and will generate {self.parent().dialogHFMSettings.summaryUpdateWorker.queuedCasesTotal} cases " if not noHFM else ". No new cases will be generated"} prior to model training.
                         <br><br>
                         > There are currently {sum(list(tensorsRe.values()))} cases stored as tensors with <b>{self.spinBoxModes.value()} modes</b>.
                         <br>
                         &nbsp;&nbsp;Re = {sorted(tensorsRe.keys())}
                         <br><br>
                         {"-"*64}
                         <br><br>
                         {_genSubtext("RBF", "Radial Basis Function", self.listWidgetRBFKernels.count(), "kernel")}
                         {_genSubtext("NN", "Neural Network", self.listWidgetNNLayers.count() * self.listWidgetNNNeurons.count(), "neuron-layer combination")}
                         {_genSubtext("GP", "Gaussian Process", self.listWidgetGPKernels.count(), "kernel")}
                       </p>
                     </body>
                   </html>""")
        else:                                                                   # Case 2: Surrogate model training is disabled
            self.textEditSummary.setText(                                       # ... provide instructions on how to enable training if desired
                f"""<html><head/>
                     <body style="color: #CCCCCC; background-color: #0C0C0C;">
                       <p>
                         {header}
                       </p>
                     </body>
                   </html>""")
