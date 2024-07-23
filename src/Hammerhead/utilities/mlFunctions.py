##############################################################################################
#                                                                                            #
#       HAMMERHEAD - Harmonic MOO Model Expedient for the Reduction                          #
#                    Hydraulic-loss/Heat-transfer Enhancement Asset Design                   #
#       Author: Daniela Minerva Segura Galeana                                               #
#       Supervised by: Antonio J. Gil                                                        #
#                                                                                            #
##############################################################################################
#                                                                                            #
#       Script containing all Machine Learning (ML) parent operations                        #
#                                                                                            #
##############################################################################################

# IMPORTS: HAMMERHEAD files ###############################################################################################################

from utilities.dataProcessing import (genericPoolManager,                       # Utilities -> Generic function for managing multiprocessed task pools
                                      loadyaml,                                 # Utilities -> Configuration loading and manipulation from resources
                                      unstandardiseTensor)                      # Utilities -> Reverting standardised data into original (expanded) data
from utilities.mlPrediction import (predictTHP,                                 # Utilities -> Calculate THP from BCV predictions
                                    maximiseTHP)                                # Utilities -> Calculate optimal features using a torch optimiser
from utilities.plotFunctions import (predictionPlot,                            # Utilities -> Plotting model predictions
                                     lossPlot,                                  # Utilities -> Plotting per-variable training loss
                                     mlBenchmarkPlot,                           # Utilities -> Plotting per-NN-architecture loss benchmark
                                     historyPlot)

import utilities.mlTraining as train                                            # Utilities -> ML training process functions

# IMPORTS: Others #########################################################################################################################

from pathlib import Path                                                        # Others -> Path manipulation
from subprocess import call, DEVNULL                                            # Others -> External process manipulation
from tqdm import tqdm                                                           # Others -> Progress bar
from typing import Any                                                          # Others -> Python type hinting

import numpy as np                                                              # Others -> Array manipulation functions
import random                                                                   # Others -> Random sampling
import shutil                                                                   # Others -> File/directory manipulation
import torch                                                                    # Others -> Tensor manipulation functions

###########################################################################################################################################

bcvNames = ["inletU", "inletT", "inletp", "outletU", "outletT", "outletp"]      # Boundary Condition Variable (BCV) names
thpNames = ["lumpedT", "lumpedp"]                                               # Thermo-Hydraulic Performance (THP) variable names used by lumped model

models = {
    "lgp": {
        "dimensionType": "lumped",
        "dimensionSize": 1,
        "variables":     thpNames,
        "function":      "GP"},
    "lnn": {
        "dimensionType": "lumped",
        "dimensionSize": 1,
        "variables":     thpNames,
        "function":      "NN"},
    "lrbf": {
        "dimensionType": "lumped",
        "dimensionSize": 1,
        "variables":     thpNames,
        "function":      "RBF"},
    "mgp": {
        "dimensionType": "modal",
        "dimensionSize": None,
        "variables":     bcvNames,
        "function":      "GP"},
    "mnn": {
        "dimensionType": "modal",
        "dimensionSize": None,
        "variables":     bcvNames,
        "function":      "NN"},
    "mrbf": {
        "dimensionType": "modal",
        "dimensionSize": None,
        "variables":     bcvNames,
        "function":      "RBF"},
    "sgp": {
        "dimensionType": "spatial",
        "dimensionSize": 101,
        "variables":     bcvNames,
        "function":      "GP"},
    "snn": {
        "dimensionType": "spatial",
        "dimensionSize": 101,
        "variables":     bcvNames,
        "function":      "NN"},
    "srbf": {
        "dimensionType": "spatial",
        "dimensionSize": 101,
        "variables":     bcvNames,
        "function":      "RBF"}}

def mlTrain(domain: str, trainTasks: list[str], nProc: int) -> None:
    """
    Train the requested ML/RBF models
    
    Parameters
    ----------
    domain : str                        String specifing the domain type (either "2D" or "axisym")
    trainTasks : list                   List of models to train
    nProc : int                         Maximum number of concurrent training processes

    Returns
    -------
    None
    """
    if len(trainTasks) == 0:                                                    # Do not continue if no models were requested for training
        print("Training was not requested for any ML/RBF model")
        return
    
    torch.set_num_threads(nProc)                                                # Limit PyTorch to number of threads specified by nProc
    
    trainingParams = loadyaml("trainingParams")                                 # Load the training parameters from ./resources/trainingParams.yaml; edit this file to use different parameters
    for param in ["layers", "neurons", "validationSplit", "gpKernels", "maternNu"]:  # As the user has the option to provide a single value or a list of values for these params ...
        if isinstance(trainingParams[param], (int, float)):                     # ... if a single value is provided
            trainingParams[param] = [trainingParams[param]]                     # ... ensure that it is always a list (needs to be iterable for code below)
            
    valSplitMaxDP = max([len(str(valSplit).split(".")[-1]) for valSplit in trainingParams["validationSplit"]])  # compute maximum number of decimals places used by validationSplit
    layersPad = max([len(str(layers)) for layers in trainingParams["layers"]])  # compute layers string padding size to accommodate largest requested layer number
    neuronsPad = max([len(str(neurons)) for neurons in trainingParams["neurons"]])  # compute neurons string padding size to accommodate largest requested neuron number
    samplesPad = len(str(trainingParams["samples"] - 1))                        # compute padded string size for number of samples
    
    for model in ["mgp", "mnn", "mrbf"]:                                        # These models have parameters that depend on data loaded from trainingParams, iterate over them
        models[model]["dimensionSize"] = trainingParams["modes"]                # ... for each model, update the dimensionSize to be the number of modes specified in trainingParams
    
    tensorDirs = [tensorDir for tensorDir in Path(f"./mlData/{domain}").glob("*")  # Get list of paths for all tensor directories in ./mlData/{domain}
                  if (tensorDir.is_dir()                                        # ... filter out all non-directory entries
                      and int(tensorDir.stem.split("_")[-1]) == 1
                      and tensorDir.stem.split("_")[1] == "500"
                      and int(tensorDir.stem.split("_")[3]) == trainingParams["modes"])]  # ... only accept directories that contain tensors with the requested number of modes
    if len(tensorDirs) == 0:                                                    # If there are no tensor directories available for the requested domain and number of modes in ./mlData, exit
        print(f"No modes={trainingParams['modes']} tensor directories found in ./mlData/{domain} for model training process!")
        return
        
    def _callTrain(**kwargs):
        """
        Private helper function for calling ML training functions
        NOTE: all variables are taken from current scope of function call
        """
        try:
            getattr(train, models[modelName]["function"])(                      # get callable function knowing its name from model dictionary
                featureSize = featureSize,                                      # call it with model-relevant parameters
                dimensionSize = models[modelName]["dimensionSize"],
                xTrain = xTrain,
                outputTrain = outputTrain,
                xValid = xValid,
                outputValid = outputValid,
                varName = var,
                outputDir = tensorDir / modelName / targetDir,
                **kwargs)
        except KeyboardInterrupt:                                               # if process is terminated mid-training
            shutil.rmtree(tensorDir / modelName / targetDir, ignore_errors=True)  # ... remove the partial output directory
            raise                                                               # ... before propagating exception

    for tensorDir in sorted(tensorDirs):                                        # Iterate over all tensor directories mathching requirements loaded from trainingParams.yaml
        print(f"Loading tensors for Re={tensorDir.stem.split('_')[1]} and modes={trainingParams['modes']} from {tensorDir} for ML/RBF training")
        tensors = {tensor: torch.load(tensorDir / f"{tensor}Data.pt") for tensor in ["modal", "spatial", "lumped"]}  # Load the necessary tensors and store them in a dictionary by name
        tensors["x"] = torch.load(tensorDir / "xData.pt")["x"]                  # Additionally, load the standardised x tensor from xData.pt
        
        dataSize, featureSize = tensors["x"].shape                              # Shape of the full data set, where shape[0] is the number of processed cases and shape[1] is the number of features
        for n in tqdm(range(trainingParams["samples"]), desc="Processing samples", leave=False):  # Repeat training "samples" times, each time using a different random validationSplit seed and different initial hyperparameters
            for validationSplit in (valPBar := tqdm(trainingParams["validationSplit"], leave=False)):  # Iterate over all requested validation split sizes (with progress bar)
                for modelName in (modelPBar := tqdm(trainTasks, leave=False)):  # Iterate over all tasks specified in --train (with labelled progress bar)
                    modelPBar.set_description(f"Processing model={modelName}")  # Update progress bar with current model
                    for var in (varPBar := tqdm(models[modelName]["variables"], leave=False)):  # Also iterate over all variables available for this model (with labelled progress bar)
                        valPBar.set_description(f"Processing validationSplit={validationSplit}")  # Update progress bar with current validationSplit
                        featMaxIdx = [set(np.where(tensors["x"][:,i]==max(tensors["x"][:,i]))[0]) for i in range(tensors["x"].shape[1])]
                        featMinIdx = [set(np.where(tensors["x"][:,i]==min(tensors["x"][:,i]))[0]) for i in range(tensors["x"].shape[1])]
                        outMaxIdx = [set(np.where(tensors[ models[modelName]["dimensionType"] ][ var ][:,i]==max(tensors[ models[modelName]["dimensionType"] ][ var ][:,i]))[0]) for i in range(tensors[ models[modelName]["dimensionType"] ][ var ].shape[1])]
                        outMinIdx = [set(np.where(tensors[ models[modelName]["dimensionType"] ][ var ][:,i]==min(tensors[ models[modelName]["dimensionType"] ][ var ][:,i]))[0]) for i in range(tensors[ models[modelName]["dimensionType"] ][ var ].shape[1])]
                        BMIdx = []
                        for i in range(len(featMaxIdx)):
                            Idx = [item for item in featMaxIdx[i]]
                            BMIdx = BMIdx + Idx
                        for i in range(len(featMinIdx)):
                            Idx = [item for item in featMinIdx[i]]
                            BMIdx = BMIdx + Idx
                        for i in range(len(outMaxIdx)):
                            Idx = [item for item in outMaxIdx[i]]
                            BMIdx = BMIdx + Idx
                        for i in range(len(outMinIdx)):
                            Idx = [item for item in outMinIdx[i]]
                            BMIdx = BMIdx + Idx
                        BMIdx = list(set(BMIdx))
                        indexSequence = list(np.arange(dataSize))
                        indexSequence = [item for item in indexSequence if item not in (BMIdx)]
                        validSize = int(np.floor(validationSplit * dataSize))   # Compute the size of the validation data set, default is 35% of full data set for validation
                        indices = random.sample(indexSequence, k=(len(indexSequence))) + random.sample(BMIdx, k=(len(BMIdx)))  # Get a randomly arranged list of indices for the full data set
                        trainIdx, validIdx = indices[validSize:], indices[:validSize]  # Split all indices into two groups: indices that will address the training and the validation data sets
                
                        xTrain = tensors["x"][trainIdx, :]                      # Select the training cases from the xData tensor
                        xValid = tensors["x"][validIdx, :]                      # Select the validation cases from the xData tensor
                        
                        varPBar.set_description(f"Processing var={var}")        # Update progress bar with current flow variable
                        outputTrain = tensors[ models[modelName]["dimensionType"] ][ var ][trainIdx, :]  # ... compute the output training tensor
                        outputValid = tensors[ models[modelName]["dimensionType"] ][ var ][validIdx, :]  # ... compute the output validation tensor
                        if models[modelName]["function"] == "NN":               # If model is a neural network, we need to also iterate over requested layers/neurons
                            for layers in (layersPBar := tqdm(trainingParams["layers"], leave=False)):  # Iterate over all requested NN layer numbers (with labelled progress bar)
                                layersPBar.set_description(f"Processing layers={layers}")  # Update progress bar with current NN layer number
                                for neurons in (neuronsPBar := tqdm(trainingParams["neurons"], leave=False)):  # Iterate over all requested NN neuron numbers (with labelled progress bar)
                                    neuronsPBar.set_description(f"Processing neurons={neurons}")  # Update progress bar with current NN neuron number
                                    targetDir = "_".join(                       # construct string of target model state dictionary directory (innermost)
                                        [f"validationSplit_{validationSplit:.{valSplitMaxDP}f}",
                                         f"layers_{layers:0{layersPad}d}",
                                         f"neurons_{neurons:0{neuronsPad}d}",
                                         f"n_{n:0{samplesPad}d}"])
                                    _callTrain(layers = layers, neurons = neurons)  # call model-specific ML training function (layers and neurons are NN-specific kwargs)
                        elif models[modelName]["function"] == "GP":             # if model is a Gaussian Process, loop over kernels
                            for kernel in (kernelPBar := tqdm(trainingParams["gpKernels"], leave=False)):
                                kernelPBar.set_description(f"Processing kernel={kernel}")  # Update progress bar with current NN layer number
                                if kernel == "MaternKernel" or kernel == "PolynomialKernel":
                                    for maternNu in (nuPBar := tqdm(trainingParams["maternNu"], leave=False)):
                                        nuPBar.set_description(f"Processing MaternKernel nu={maternNu}")
                                        targetDir = "_".join(                   # construct string of target model state dictionary directory (innermost)
                                                    [f"validationSplit_{validationSplit:.{valSplitMaxDP}f}",
                                                     f"kernel_{kernel}_{maternNu}",
                                                     f"n_{n:0{samplesPad}d}"])
                                        _callTrain(kernel = kernel, maternNu = maternNu)  # call model-specific ML training function
                                else:
                                    targetDir = "_".join(                       # construct string of target model state dictionary directory (innermost)
                                                [f"validationSplit_{validationSplit:.{valSplitMaxDP}f}",
                                                 f"kernel_{kernel}",
                                                 f"n_{n:0{samplesPad}d}"])
                                    _callTrain(kernel = kernel, maternNu = 0)   # call model-specific ML training function
   
def mlPlot(domain: str, plotTasks: list[str], nProc: int) -> None:
    """
    ML/RBF prediction plotting
    
    Parameters
    ----------
    domain : str                        String specifing the domain type (either "2D" or "axisym")
    plotTasks : list                    List of models to train / plot
    nProc : int                         Maximum number of concurrent prediction/plotting processes

    Returns
    -------
    None
    """
    if len(plotTasks) == 0:                                                     # Do not continue if no models were requested for plotting
        print("Plotting was not requested for any ML/RBF model")
        return
    
    plotParams = loadyaml("plotParams")                                         # Load plotting parameters from plotParams.yaml
    plotParams["useTex"] = plotParams["useTex"] and not bool(call("tex --help", shell=True, stdout=DEVNULL, stderr=DEVNULL))  # If text rendering with TeX is enabled, check that TeX is present on the system and update the params dict accordingly
    trainingParams = loadyaml("trainingParams")                                 # Also load settings from the trainingParams.yaml
    for model in ["mgp", "mnn", "mrbf"]:                                        # These models have parameters that depend on data loaded from trainingParams, iterate over them
        models[model]["dimensionSize"] = trainingParams["modes"]                # ... for each model, update the dimensionSize to be the number of modes specified in trainingParams (if no other function has done this yet)
        
    tensorDirs = [tensorDir for tensorDir in Path(f"./mlData/{domain}").glob("*")  # Get list of paths for all tensor directories in ./mlData/{domain}
                  if (tensorDir.is_dir()                                        # ... filter out all non-directory entries
                      and int(tensorDir.stem.split("_")[3]) == trainingParams["modes"])]  # ... only accept directories that contain tensors with the requested number of modes
    if len(tensorDirs) == 0:                                                    # If there are no tensor directories available for the requested domain and number of modes in ./mlData, exit
        print(f"No modes={trainingParams['modes']} tensor directories found in ./mlData/{domain} for prediction/plotting process!")
        return
    
    xv, yv = np.linspace(0.001, 0.010, 30), np.linspace(2, 60, 30)              # X- and Y-value arrays used as input for the prediction tasks and for 3D surface plotting
    
    xPredExpandedArray1H = np.array(np.meshgrid(xv, yv), dtype=np.float32).T.reshape(-1, 2)  # Array of unstandardised (expanded) x-values used for prediction for a 1-harmonic scenario
    xPredExpandedArray2H = np.insert(xPredExpandedArray1H, [1, 2], [plotParams["A2"], plotParams["k2"]], axis=1)  # Array of unstandardised (expanded) x-values used for prediction for a 2-harmonic scenario
    
    tasksSingleRe, tensorDirsReAll = getPredPlotTasksSingleRe(plotTasks, tensorDirs, xPredExpandedArray1H, xPredExpandedArray2H)  # Get a partial list of task arguments for all single-Re scenarios, as well as a list of multi-Re tensor directories
    tasksMultiRe = getPredPlotTasksMultiRe(plotTasks, tensorDirsReAll, xPredExpandedArray1H, xPredExpandedArray2H)  # Get a partial list of task arguments for all multi-Re scenarios
    
    predPlotArgs = [[xv, yv, plotParams] + task for task in tasksSingleRe + tasksMultiRe]  # Merge the two lists of task arguments, adding other necessary parameters that exist locally 
    lossPlotArgs = getLossPlotTaskArgs(plotTasks, tensorDirs, plotParams)       # Get list of task arguments for plotting per-variable loss
    benchmarkPlotArgs = getBenchmarkPlotTaskArgs(plotTasks, tensorDirs, plotParams)  # Get list of task arguments for plotting per-architecture loss benchmark
    taskFuncs = sum([[fn for _ in args] for fn, args in zip([predictionPlot, lossPlot, mlBenchmarkPlot], [predPlotArgs, lossPlotArgs, benchmarkPlotArgs])], [])  # Create a list containing function objects for each element in all taskArgs lists
    taskArgs = predPlotArgs + lossPlotArgs + benchmarkPlotArgs                  # Combine all function arguments into a single list with the same size as taskFuncs, to be passed to pool manager
     
    genericPoolManager(taskFuncs, taskArgs, None, nProc, "Plotting", None)      # Send all tasks to the multi-threaded worker function
    print("Plotting process completed successfully")
    
def getPredPlotTasksSingleRe(plotTasks: list[str],
                             tensorDirs: list[Path],
                             xPredExpandedArray1H: np.ndarray,
                             xPredExpandedArray2H: np.ndarray) -> tuple[list[list[str, torch.tensor, Path]], list[Path]]:
    """
    ML/RBF prediction task collection function for a single-Re scenario
    
    Parameters
    ----------
    plotTasks : list                    List of models to train / plot
    tensorDirs : list                   List of tensor directories for which to predict / plot
    xPredExpandedArray1H : array-like   Array of unstandardised (expanded) x-values used for prediction for a 1-harmonic scenario
    xPredExpandedArray2H : array-like   Array of unstandardised (expanded) x-values used for prediction for a 2-harmonic scenario

    Returns
    -------
    tasksSingleRe : list                List of tasks, where each entry is a partial list of parameters to be passed to lumpedPlot
    tensorDirsReAll : list              List of paths corresponding to all relevant Re_All tensor directories (usually length 2, one for each harmonics value)
    """
    tasksSingleRe = []                                                          # List of tasks, where each entry will be a partial list of parameters to be passed to lumpedPlot
    tensorDirsReAll = []                                                        # List that will contain paths corresponding to all relevant Re_All tensor directories
    for tensorDir in tqdm(sorted(tensorDirs), desc="Predicting for single-Re scenarios"):  # Iterate over all tensor directories previously matching requirements loaded from trainingParams.yaml

        if tensorDir.stem.startswith("Re_All"):                                 # If tensor directory starts with Re_All, it is a multi-Re scenario that won't be processed here
            tensorDirsReAll.append(tensorDir)                                   # ... add it to a list that will be passed to the corresponding multi-Re function
            continue                                                            # ... and move on to the next tensor directory
            
        VTReduced = torch.load(tensorDir / "VTReduced.pt")                      # Each boundary condition variable has its own set of modes, and therefore its own set of left eigenvectors, load the corresponding dictionary of tensors
        xData = torch.load(tensorDir / "xData.pt")                              # Load the xData dictionary of tensors
        x, xMean, xStd = xData["x"], xData["xMean"], xData["xStd"]              # Extract the mean and standard deviation of the unstandardised xData, as well as the standardised xData
        xPredExpanded = torch.from_numpy(xPredExpandedArray1H if int(tensorDir.stem.split("_")[-1]) == 1 else xPredExpandedArray2H)  # Deduce the number of harmonics from the current tensorDir name, and convert the corresponding array to a tensor
        xPred = (xPredExpanded - xMean) / xStd                                  # The models are trained with normalised data, so the features need to be normalised with the stored mean and standard deviation values
        
        for modelName in plotTasks:                                             # Iterate over all tasks specified in --plot
            for stateDictDir in [path for path in (tensorDir / modelName).glob("*") if path.is_dir()]:  # Also iterate over all architecture-relevant state dictionary subdirectories
                try:
                    dataMean = torch.load(tensorDir / f"{models[modelName]['dimensionType']}Mean.pt")  # Load the tensor of output mean values for the current model's dimension type
                    dataStd = torch.load(tensorDir/ f"{models[modelName]['dimensionType']}Std.pt")  # Load the tensor of output standard deviation values for the current model's dimension type
                    predictedTHPQual = predictTHP(xPred, modelName, models[modelName], dataMean, dataStd, stateDictDir, VTReduced)  # Call the model's corresponding THP prediction function, passing it the collected arguments (for 3D qualitative plot)
                    predictedTHPQuant = predictTHP(x, modelName, models[modelName], dataMean, dataStd, stateDictDir, VTReduced)  # Also carry out prediction for the 2D quantitative plots
                    tasksSingleRe.append([predictedTHPQual, predictedTHPQuant, stateDictDir])  # Add the current model name, computed THP value, and the current tensor directory to the list of single-Re task parameters
                except FileNotFoundError as e:
                    pathParts = e.filename.split("/")
                    print(f"No available {pathParts[-1]} checkpoint found for prediction in {Path(*pathParts[:-1])} (try --train {pathParts[-3]})")
                
    return tasksSingleRe, tensorDirsReAll
            
def getPredPlotTasksMultiRe(plotTasks: list[str],
                            tensorDirsReAll: list[Path],
                            xPredExpandedArray1H: np.ndarray,
                            xPredExpandedArray2H: np.ndarray) -> list[list[str, torch.tensor, Path]]:
    """
    ML/RBF prediction task collection function for a multi-Re scenario
    
    Parameters
    ----------
    plotTasks : list                    List of models to train / plot
    tensorDirsReAll : list              List of Re_All tensor directories for which to predict / plot
    xPredExpandedArray1H : array-like   Array of unstandardised (expanded) x-values used for prediction for a 1-harmonic scenario
    xPredExpandedArray2H : array-like   Array of unstandardised (expanded) x-values used for prediction for a 2-harmonic scenario

    Returns
    -------
    tasksMultiRe : list                 List of tasks, where each entry is a partial list of parameters to be passed to lumpedPlot
    """
    tasksMultiRe = []                                                           # List of tasks, where each entry will be a partial list of parameters to be passed to lumpedPlot
    for tensorDirReAll in tqdm(tensorDirsReAll, desc="Predicting for multi-Re scenarios"):  # Iterate over all Re_All tensor directories previously identified by the corresponding single-Re function
        uniqueRe = torch.unique(torch.load(tensorDirReAll / "xData.pt")["xExpanded"][:, -1]).int()  # Get an integer tensor of unique Re values from the last column of the xExpanded tensor (where Re is a feature)
        VTReduced = torch.load(tensorDirReAll / "VTReduced.pt")                 # Each boundary condition variable has its own set of modes, and therefore its own set of left eigenvectors, load the corresponding dictionary of tensors
        xData = torch.load(tensorDirReAll / "xData.pt")                         # Load the xData dictionary of tensors
        x, xMean, xStd = xData["x"], xData["xMean"], xData["xStd"]              # Extract the mean and standard deviation of the unstandardised xData, as well as the standardised xData
        xPredExpandedArray = xPredExpandedArray1H if int(tensorDirReAll.stem.split("_")[-1]) == 1 else xPredExpandedArray2H  # Deduce the number of harmonics from the current tensorDir name, and set it as the chosen xPredExpanded array
        
        for modelName in plotTasks:                                             # Iterate over all tasks specified in --plot
            for stateDictDirReAll in [path for path in (tensorDirReAll / modelName).glob("*") if path.is_dir()]:  # Also iterate over all architecture-relevant state dictionary subdirectories
                try:
                    dataMean = torch.load(tensorDirReAll / f"{models[modelName]['dimensionType']}Mean.pt")  # Load the tensor of output mean values for the current model's dimension type
                    dataStd = torch.load(tensorDirReAll / f"{models[modelName]['dimensionType']}Std.pt")  # Load the tensor of output standard deviation values for the current model's dimension type
                    
                    for Re in uniqueRe:                                         # Iterate over all unique Re values (from the last column of the original xExpanded tensor)
                        xPredExpanded = torch.from_numpy(np.insert(xPredExpandedArray, xPredExpandedArray.shape[1], Re, axis=1))  # Insert the current Re value as the last column of the xPredExpanded array and convert it to a tensor
                        xPred = (xPredExpanded - xMean) / xStd                  # The models are trained with normalised data, so the features need to be normalised with the stored mean and standard deviation values
                        predictedTHPQual = predictTHP(xPred, modelName, models[modelName], dataMean, dataStd, stateDictDirReAll, VTReduced)  # Call the model's corresponding THP prediction function, passing it the collected arguments (for 3D qualitative plot)
                        predictedTHPQuant = predictTHP(x, modelName, models[modelName], dataMean, dataStd, stateDictDirReAll, VTReduced)  # Also carry out prediction for the 2D quantitative plots
                        tasksMultiRe.append([predictedTHPQual, predictedTHPQuant, stateDictDirReAll, Re])  # Add the current model name, computed THP value, the current tensor directory, and the current Reynolds number to the list of mulit-Re task parameters

                except FileNotFoundError as e:
                    pathParts = e.filename.split("/")
                    print(f"No available {pathParts[-2]} checkpoint found for prediction in {pathParts[-3]} (try --train {pathParts[-2]})")
                
    return tasksMultiRe
    
def getLossPlotTaskArgs(plotTasks: list[str],
                        tensorDirs: list[Path],
                        plotParams: dict[str, Any]) -> list[dict[str, Any], Path]:
    """
    Scans the mlData directory tree for valid paths for which loss plots need
        to be generated, and returns per-task loss plot function arguments
    
    Parameters
    ----------
    plotTasks : list                    List of models to train / plot
    tensorDirs : list                   List of tensor directories
    plotParams : dict                   Dictionary of plotting parameters from YAML file

    Returns
    -------
    lossPlotTaskArgs : list             List of per-task loss plot function arguments
    """
    lossPlotTaskArgs = []
    plotTasks = set(plotTasks) & {"lnn", "mnn", "snn", "lgp", "mgp", "sgp"}     # Set intersection of requested and possible plot tasks represents the set of models for which loss can and will be plotted
    for tensorDir in tqdm(tensorDirs, desc="Preparing loss plot data"):         # Iterate over all available tensor directories
        for plotTask in plotTasks:                                              # Iterate over all requested and available NN plot tasks
            for stateDictDir in (tensorDir / plotTask).glob("*"):               # Iterate also over all available subdirectories in each tensor directory (containing state dict directories)
                if stateDictDir.is_dir():                                       # If the state dict path is a valid directory
                    lossPlotTaskArgs.append([plotParams, stateDictDir])         # Add it to the list of arguments (each item in list represents one list of arguments, first is plotParams, second is stateDictDir path)
    return lossPlotTaskArgs
    
def getBenchmarkPlotTaskArgs(plotTasks: list[str],
                             tensorDirs: list[Path],
                             plotParams: dict[str, Any]) -> list[dict[str, Any], Path]:
    """
    Scans the mlData directory tree for valid paths for which ML benchmark
        plots need to be generated, and returns per-task benchmark plot
        function arguments
    
    Parameters
    ----------
    plotTasks : list                    List of models to train / plot
    tensorDirs : list                   List of tensor directories
    plotParams : dict                   Dictionary of plotting parameters from YAML file

    Returns
    -------
    lossPlotTaskArgs : list             List of per-task benchmark plot function arguments
    """
    lossPlotTaskArgs = []
    plotTasksNN = set(plotTasks) & {"lnn", "mnn", "snn"}                        # Set intersection of requested and possible plot tasks represents the set of models for which loss can and will be plotted
    for tensorDir in tqdm(tensorDirs, desc="Preparing ML benchmark plot data"):  # Iterate over all available tensor directories
        for plotTask in plotTasks:                                              # Iterate over all requested and available NN plot tasks
        
            lossTable = {}
            for stateDictDir in (tensorDir / plotTask).glob("*"):               # Iterate also over all available subdirectories in each tensor directory (containing state dict directories)
                if stateDictDir.is_dir():                                       # If the state dict path is a valid directory
                    if plotTask in plotTasksNN:
                        _, valSplit, _, layers, _, neurons, _, _ = stateDictDir.name.split("_")  # extract validation split, layers and neurons from state dict path name
                    else:
                        valSplit = float(stateDictDir.name.split("_")[1])
                    for varStateDict in stateDictDir.glob("*.pt"):              # for each variable inside the state dict folder
                        if plotTask in plotTasksNN:
                            key = (valSplit, layers, neurons, varStateDict.stem)  # set dictionary key as the current NN architecture and flow variable
                        elif stateDictDir.name.split("_")[3] == "MaternKernel":
                            key = (valSplit, stateDictDir.name.split("_")[3], float(stateDictDir.name.split("_")[4]), varStateDict.stem)
                        else:
                            key = (valSplit, stateDictDir.name.split("_")[3], varStateDict.stem)
                        loss = torch.load(varStateDict)["lossTrain"][-1]        # extract the last loss from the variable state dictionary
                        if key in lossTable:                                    # if this combination of NN architecture and flow variable has been seen before, a list of loss values already exists
                            lossTable[key].append(loss)                         # ... therefore, append to the list
                        else:                                                   # if this combination is new ...
                            lossTable[key] = [loss]                             # ... a list of per-sample losses needs to be created (including the now first-seen loss)
                            
            if lossTable:                                                       # if the loss table isn't empty, we can convert it to an array
                lossTableTemp = [[*key, np.array(value)] for key, value in lossTable.items() if len(value) > 0]  # convert dictionary to a list of lists with columns [valSplit, layers, neurons, var, dataArr] (discarding any empty entries)
                lossPlotTaskArgs.append([plotParams, tensorDir / plotTask, np.array(lossTableTemp, dtype=object)])  # the temporary loss table is converted to an unstructured array of objects and will be an argument passed to the plot function
    return lossPlotTaskArgs
    
def mlOptimalSearch(domain: str, searchTasks: list[str], nProc: int) -> None:
    """
    ML/RBF optimal feature search using THP prediction values
    
    Parameters
    ----------
    domain : str                        String specifing the domain type (either "2D" or "axisym")
    searchTasks : list                  List of models for which to perform optimal search
    nProc : int                         Maximum number of concurrent prediction processes

    Returns
    -------
    None
    """
    if len(searchTasks) == 0:                                                   # Do not continue if no models were requested for optimal search
        print("Optimal search was not requested for any ML/RBF model")
        return
    
    trainingParams = loadyaml("trainingParams")                                 # Load training parameters
    plotParams = loadyaml("plotParams")                                         # Load training parameters 
    for model in ["mgp", "mnn", "mrbf"]:                                        # These models have parameters that depend on data loaded from trainingParams, iterate over them
        models[model]["dimensionSize"] = trainingParams["modes"]                # ... for each model, update the dimensionSize to be the number of modes specified in trainingParams
    tensorDirs = [tensorDir for tensorDir in Path(f"./mlData/{domain}").glob("*")  # Get list of paths for all tensor directories in ./mlData/{domain}
                  if (tensorDir.is_dir() and int(tensorDir.stem.split("_")[3]) == trainingParams["modes"])]  # ... filter out all non-directory entries and only accept directories that contain tensors with the requested number of modes
    if len(tensorDirs) == 0:                                                    # If there are no tensor directories available for the requested domain and number of modes in ./mlData, exit
        print(f"No modes={trainingParams['modes']} tensor directories found in ./mlData/{domain} for optimal search process!")
        return                 

    taskArgs = []                                                               # ... corresponding list of function arguments
    for tensorDir in sorted(tensorDirs):                                        # Iterate over all tensor directories mathching requirements loaded from trainingParams.yaml
        for name, model in [(task, models[task]) for task in searchTasks]:      # Iterate over all tasks specified in --plot, returning the name of the model for training, and its respective entry in the dictionary
            stateDictDirs = [dictDir for dictDir in Path(f"{tensorDir}/{name}").glob("*")
                             if dictDir.is_dir()]
            for stateDictDir in stateDictDirs:
                taskArgs.append([model, name, stateDictDir, tensorDir])         # Create list of arguments that will be passed to the maximiseTHP function, and append it to the list of task arguments
    historyPlotArgs = getLossPlotTaskArgs(searchTasks, tensorDirs, plotParams)  # Get list of task arguments for plotting per-variable loss
    taskFuncs = sum([[fn for _ in args] for fn, args in zip([maximiseTHP], [taskArgs])], [])  # Create a list containing function objects for each element in all taskArgs lists
    taskPlotFuncs = sum([[fn for _ in args] for fn, args in zip([historyPlot], [historyPlotArgs])], [])  # Create a list containing function objects for each element in all taskArgs lists
    taskPlotArgs = historyPlotArgs                                              # Combine all function arguments into a single list with the same size as taskFuncs, to be passed to pool manager
    genericPoolManager(taskFuncs, taskArgs, None, nProc, "Searching for optimal feature values", "Completed optimal search using {} model with Re={} harmonics={}: {}")  # Send all tasks to the multi-threaded worker function
    genericPoolManager(taskPlotFuncs, taskPlotArgs, None, nProc, "Plotting", None)  # Send all tasks to the multi-threaded worker function
    print("Optimal search process completed successfully")
        
###############################################################################

if __name__ == "__main__":
    raise RuntimeError("This script is not intended for execution. Instead, execute hammerhead.py from the parent directory.")
