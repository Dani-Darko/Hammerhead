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
                                      normaliseTensor)                          # Utilities -> Normalising tensors
from utilities.mlPrediction import (predictTHP,                                 # Utilities -> Calculate THP from BCV predictions
                                    maximiseTHP)                                # Utilities -> Calculate optimal features using a torch optimiser
from utilities.plotFunctions import (predictionPlot,                            # Utilities -> Plotting model predictions
                                     varPlot,                                   # Utilities -> Plotting flow variable profiles
                                     lossPlot,                                  # Utilities -> Plotting per-variable training loss
                                     mlBenchmarkPlot,                           # Utilities -> Plotting per-NN-architecture loss benchmark
                                     historyPlot)                               # Utilities -> Plotting optimisation process gradient

import utilities.mlTraining as train                                            # Utilities -> ML training process functions

# IMPORTS: Others #########################################################################################################################

from gpytorch import kernels                                                    # Other -> All available gpytorch kernels
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

modelPrefix = {
    "L": {
        "dimensionType": "lumped",
        "dimensionSize": 1,
        "variables":     thpNames},
    "M": {
        "dimensionType": "modal",
        "dimensionSize": None,
        "variables":     bcvNames},
    "S": {
        "dimensionType": "spatial",
        "dimensionSize": 101,
        "variables":     bcvNames}
}

def mlTrain(domain: str, trainTasks: list[str], nProc: int, trainingParamsOverride: dict[str, Any]) -> None:
    """
    Train the requested ML/RBF models
    
    Parameters
    ----------
    domain : str                        String specifying the domain type (either "2D" or "axisym")
    trainTasks : list                   List of models to train
    nProc : int                         Maximum number of concurrent training processes
    trainingParamsOverride : dict[str, Any]  Dictionary of trainingParams that will supersede loaded values

    Returns
    -------
    None
    """
    if len(trainTasks) == 0:                                                    # Do not continue if no models were requested for training
        print("Training was not requested for any ML/RBF model")
        return
    
    torch.set_num_threads(nProc)                                                # Limit PyTorch to number of threads specified by nProc
    
    trainingParams = loadyaml("trainingParams", override = trainingParamsOverride)  # Load the training parameters from ./resources/trainingParams.yaml; edit this file to use different parameters
    trainingParams["kernelsGP"] = [getattr(kernels, kernel) for kernel in trainingParams["kernelsGP"]]  # Replace gpytorch kernel identifier strings with the kernel objects
    
    for param in ["layers", "neurons", "validationSplit", "kernelsGP", "kernelsRBF"]:  # As the user has the option to provide a single value or a list of values for these params ...
        if isinstance(trainingParams[param], (int, float)):                     # ... if a single value is provided
            trainingParams[param] = [trainingParams[param]]                     # ... ensure that it is always a list (needs to be iterable for code below)
            
    valSplitMaxDP = max([len(str(valSplit).split(".")[-1]) for valSplit in trainingParams["validationSplit"]])  # Compute maximum number of decimals places used by validationSplit
    layersPad = max([len(str(layers)) for layers in trainingParams["layers"]])  # Compute layers string padding size to accommodate largest requested layer number
    neuronsPad = max([len(str(neurons)) for neurons in trainingParams["neurons"]])  # Compute neurons string padding size to accommodate largest requested neuron number
    samplesPad = len(str(trainingParams["samples"] - 1))                        # Compute padded string size for number of samples
    
    modelPrefix["M"]["dimensionSize"] = trainingParams["modes"]                 # If model is M (modal), update the dimensionSize to be the number of modes specified in trainingParams
    
    tensorDirs = [tensorDir for tensorDir in Path(f"./mlData/{domain}").glob("*")  # Get list of paths for all tensor directories in ./mlData/{domain}
                  if (tensorDir.is_dir()                                        # ... filter out all non-directory entries
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
            getattr(train, modelName[1:])(                                      # Get callable function knowing its name (suffix of model string)
                featureSize = featureSize,                                      # Call it with model-relevant parameters
                dimensionSize = modelPrefix[ modelName[0] ]["dimensionSize"],
                xTrain = xTrain,
                outputTrain = outputTrain,
                xValid = xValid,
                outputValid = outputValid,
                varName = var,
                outputDir = tensorDir / modelName / targetDir,
                **kwargs)
        except KeyboardInterrupt:                                               # If process is terminated mid-training
            shutil.rmtree(tensorDir / modelName / targetDir, ignore_errors=True)  # ... remove the partial output directory
            raise                                                               # ... before propagating exception
    
    def _computeTensorMasks():
        """
        Private helper function to compute some useful tensor masks
        NOTE: all variables are taken from current scope of function call
        
        To discourage the surrogate models from extrapolating, the boundaries
            of our data range need to be part of the training set, therefore
            masks need to be computed that contain such information.
        featAllMask includes only the cases where all the features are on
            those boundaries, while featAnyMask contains cases that share
            boundaries with other values. Furthermore, to prevent regions of
            high and low density, where the surrogate models will perform 
            better and worse, respectively, the middle of the data range is
            provided by featMidMask.
        Additionally, the minimum and maximum values of the output tensor have
            to be present during training for the model to effectively predict
            the correct data extents. Theses are not computed in this function,
            as they rely on data from the per-variable output tensor.
        """
        featMaxMask = torch.eq(tensors["x"], 1)                                 # Bool mask where entries in normalised tensor are 1; 2D: [cases, features]
        feat0Mask = torch.eq(tensors["x"], 0)                                   # Bool mask where entries in normalised tensor are 0; 2D: [cases, features]
        
        # We now need to compute the non-zero minima (smooth pipe where minimum = 0 is not considered a minimum of the data range): best way without copying tensor is to temporarily set 0 to inf
        tensors["x"][feat0Mask] = float("inf")                                  # Set all 0 values to inf
        featMin, _ = torch.min(tensors["x"], 0)                                 # Find per-feature minima
        featMinMask = torch.eq(tensors["x"], featMin)                           # Bool mask where entries in normalised tensor are per-feature minima; 2D: [cases, features]
        tensors["x"][feat0Mask] = 0.0                                           # Return all inf values back to zero
                
        featOrMask = featMaxMask | featMinMask | feat0Mask                      # Bool OR of all three masks: where entries are 0, 1 or non-zero per-feature minima; 2D: [cases, features]
        featAllMask = torch.all(featOrMask, 1)                                  # Bool mask where ALL features in above mask are True; 1D: [cases]
        featAnyMask = torch.any(featOrMask, 1)                                  # Bool mask where ANY features in above mask are True; 1D: [cases]
        
        featCloseToMid = abs(tensors["x"] - torch.mean(tensors["x"], 0))        # Distance from mean value of each item in 2D tensor
        featMidVal, _ = torch.min(featCloseToMid, 0)                            # Per-feature minimum (closest value to mean)
        featMidMask = torch.all(torch.eq(featCloseToMid, featMidVal), 1)        # Bool mask where ALL features are closest to per-feature mean; 1D: [cases]
        
        return featAllMask, featAnyMask, featMidMask

    for tensorDir in sorted(tensorDirs):                                        # Iterate over all tensor directories matching requirements loaded from trainingParams.yaml
        print(f"Loading tensors for Re={tensorDir.stem.split('_')[1]} and modes={trainingParams['modes']} from {tensorDir} for ML/RBF training")
        tensors = {tensor: torch.load(tensorDir / f"{tensor}Data.pt", weights_only=True) for tensor in ["modal", "spatial", "lumped"]}  # Load the necessary tensors and store them in a dictionary by name
        tensors["x"] = torch.load(tensorDir / "xData.pt", weights_only=True)["x"]  # Additionally, load the standardised x tensor from xData.pt
        dataSize, featureSize = tensors["x"].shape                              # Shape of the full data set, where shape[0] is the number of processed cases and shape[1] is the number of features
        featAllMask, featAnyMask, featMidMask = _computeTensorMasks()

        for n in tqdm(range(trainingParams["samples"]), desc="Processing samples", leave=False):  # Repeat training "samples" times, each time using a different random validationSplit seed and different initial hyperparameters
            for validationSplit in (valPBar := tqdm(trainingParams["validationSplit"], leave=False)):  # Iterate over all requested validation split sizes (with progress bar)
                valPBar.set_description(f"Processing validationSplit={validationSplit}")  # Update progress bar with current validationSplit
                for modelName in (modelPBar := tqdm(trainTasks, leave=False)):  # Iterate over all tasks specified in --train (with labelled progress bar)
                    modelPBar.set_description(f"Processing model={modelName}")  # Update progress bar with current model
                    for var in (varPBar := tqdm(modelPrefix[ modelName[0] ]["variables"], leave=False)):  # Also iterate over all variables available for this model (with labelled progress bar)
                        varPBar.set_description(f"Processing var={var}")        # Update progress bar with current flow variable

                        outputTensor = tensors[ modelPrefix[ modelName[0] ]["dimensionType"] ][ var ]  # Output tensor for current variable
                        outputTensorAllMask = torch.all(torch.eq(outputTensor, 1) | torch.eq(outputTensor, 0), 1)  # Bool mask where ALL features are either 0 or 1; 1D: [cases]
                       
                        # Define mask of cases that will be placed at the END of our final tensor (will be prioritised for training): where cases are equivalent to the boundary values of the feature and output tensor
                        endMask = featAllMask | outputTensorAllMask             # Bool mask where each row in input and output tensor is all 0, 1 or min (per-column minimum not needed in output tensor as minimum = 0)
                        
                        startIdx = torch.arange(dataSize)[~featAnyMask & ~(endMask | featMidMask)]  # Indices of cases placed at the START of tensor (least priority for training); non-boundary cases, excluding middle-range cases
                        midIdx = torch.arange(dataSize)[featAnyMask & ~(endMask| featMidMask)]  # Indices of cases placed in the MIDDLE of the tensor (medium priority for training); partial boundary, excluding full boundary
                        endIdx = torch.arange(dataSize)[endMask | featMidMask]  # Indices of cases placed at the END of the tensor (maximum priority for training); full boundary, including middle-range cases

                        validSize = int(np.floor(validationSplit * dataSize))   # Compute the size of the validation data set, default is 35% of full data set for validation
                            
                        startIdx, midIdx = startIdx[torch.randperm(startIdx.shape[0])], midIdx[torch.randperm(midIdx.shape[0])]  # randomly shuffle low- and medium-priority case indices
                        indices = torch.hstack((startIdx, midIdx, endIdx))      # Stack all indices to get a single tensor of indices
                        assert(indices.shape[0] == tensors["x"].shape[0])       # Ensure that it is the case size in dimension 0 as x-tensor
                        
                        trainIdx, validIdx = indices[validSize:], indices[:validSize]  # Split case indices into training and validation data
                        xTrain = tensors["x"][trainIdx, :]                      # Select the training cases from the xData tensor
                        xValid = tensors["x"][validIdx, :]                      # Select the validation cases from the xData tensor
                        outputTrain = tensors[ modelPrefix[ modelName[0] ]["dimensionType"] ][ var ][trainIdx, :]  # ... compute the output training tensor
                        outputValid = tensors[ modelPrefix[ modelName[0] ]["dimensionType"] ][ var ][validIdx, :]  # ... compute the output validation tensor
                        
                        #######################################################
                        
                        if modelName[1:] == "NN":                               # If model is a neural network, we need to also iterate over requested layers/neurons
                            for layers in (layersPBar := tqdm(trainingParams["layers"], leave=False)):  # Iterate over all requested NN layer numbers (with labelled progress bar)
                                layersPBar.set_description(f"Processing layers={layers}")  # Update progress bar with current NN layer number
                                for neurons in (neuronsPBar := tqdm(trainingParams["neurons"], leave=False)):  # Iterate over all requested NN neuron numbers (with labelled progress bar)
                                    neuronsPBar.set_description(f"Processing neurons={neurons}")  # Update progress bar with current NN neuron number
                                    
                                    targetDir = "_".join(                       # Construct string of target model state dictionary directory (innermost)
                                        [f"validationSplit_{validationSplit:.{valSplitMaxDP}f}",
                                         f"layers_{layers:0{layersPad}d}",
                                         f"neurons_{neurons:0{neuronsPad}d}",
                                         f"n_{n:0{samplesPad}d}"])
                                    _callTrain(layers = layers, neurons = neurons)  # Call model-specific ML training function (layers and neurons are NN-specific kwargs)
                                    
                        ########################################################
                        
                        if modelName[1:] == "GP":                               # If model is a Gaussian Process, loop over GP kernels
                            for kernel in (kernelGPPBar := tqdm(trainingParams["kernelsGP"], leave=False)):  # Iterate over all requested GP kernels (with labelled progress bar)
                                kernelGPPBar.set_description(f"Processing kernel={kernel.__name__}")  # Update progress bar with current GP kernel name
                                
                                targetDir = "_".join(                           # Construct string of target model state dictionary directory (innermost)
                                            [f"validationSplit_{validationSplit:.{valSplitMaxDP}f}",
                                             f"kernel_{kernel.__name__}",
                                             f"n_{n:0{samplesPad}d}"])
                                _callTrain(kernel = kernel)                     # Call model-specific ML training function (kernel in a GP-specific kwarg)
                        
                        ########################################################
                        
                        if modelName[1:] == "RBF":                              # If model is a Radial Basis Function, loop over RBF kernels
                            for kernel in (kernelRBFPBar := tqdm(trainingParams["kernelsRBF"], leave=False)):  # Iterate over all requested RBF kernels (with labelled progress bar)
                                kernelRBFPBar.set_description(f"Processing kernel={kernel}")  # Update progress bar with current RBF kernel name
                                
                                targetDir = "_".join(                           # Construct string of target model state dictionary directory (innermost)
                                            [f"validationSplit_{validationSplit:.{valSplitMaxDP}f}",
                                             f"kernel_{''.join([s.capitalize() for s in kernel.split('_')])}",  # Remove underscores and convert to ClassCase
                                             f"n_{n:0{samplesPad}d}"])
                                _callTrain(kernel = kernel)                     # Call model-specific ML training function (kernel in a RBF-specific kwarg)
                        
def mlOptimalSearch(domain: str, searchTasks: list[str], nProc: int, trainingParamsOverride: dict[str, Any]) -> None:
    """
    ML/RBF optimal feature search using THP prediction values
    
    Parameters
    ----------
    domain : str                        String specifing the domain type (either "2D" or "axisym")
    searchTasks : list                  List of models for which to perform optimal search
    nProc : int                         Maximum number of concurrent prediction processes
    trainingParamsOverride : dict[str, Any]  Dictionary of trainingParams that will supersede loaded values

    Returns
    -------
    None
    """
    if len(searchTasks) == 0:                                                   # Do not continue if no models were requested for optimal search
        print("Optimal search was not requested for any ML/RBF model")
        return
    
    trainingParams = loadyaml("trainingParams", override = trainingParamsOverride)  # Also load settings from the trainingParams.yaml
    modelPrefix["M"]["dimensionSize"] = trainingParams["modes"]                 # If model is M (modal), update the dimensionSize to be the number of modes specified in trainingParams
    
    tensorDirs = [tensorDir for tensorDir in Path(f"./mlData/{domain}").glob("*")  # Get list of paths for all tensor directories in ./mlData/{domain}
                  if (tensorDir.is_dir()                                        # ... filter out all non-directory entries
                      and int(tensorDir.stem.split("_")[3]) == trainingParams["modes"])]  # ... only accept directories that contain tensors with the requested number of modes
    if len(tensorDirs) == 0:                                                    # If there are no tensor directories available for the requested domain and number of modes in ./mlData, exit
        print(f"No modes={trainingParams['modes']} tensor directories found in ./mlData/{domain} for optimal search process!")
        return                 

    optimTaskArgs = []                                                          # List of function arguments (each item is a list args for each function call) for the initial optimisation task
    for tensorDir in sorted(tensorDirs):                                        # Iterate over all tensor directories mathching requirements loaded from trainingParams.yaml
        for name, model in [(task, modelPrefix[task[0]]) for task in searchTasks]:  # Iterate over all tasks specified in --plot, returning the name of the model for training, and its respective entry in the dictionary
            for stateDictDir in Path(f"{tensorDir}/{name}").glob("*"):          # Iterate over all children in parent path tensorDir/name
                if not stateDictDir.is_dir():                                   # If child path is not a directory ...
                    continue                                                    # ... ignore child path
                optimTaskArgs.append([name, model, stateDictDir, tensorDir])    # Create list of arguments that will be passed to the maximiseTHP function, and append it to the list of task arguments
                
    optimTaskFuncs = sum([[fn for _ in args] for fn, args in zip([maximiseTHP], [optimTaskArgs])], [])  # Create a list containing optimisation function objects for each element in all optimTaskArgs lists
    genericPoolManager(optimTaskFuncs, optimTaskArgs, None, nProc,              # Send all tasks to the multi-threaded worker function
                       "Searching for optimal feature values",
                       "Completed optimal search using {} model with {}: Re={} harmonics={}: {}")
    
    print("Optimal search process completed successfully")
   
def mlPlot(domain: str, plotTasks: list[str], nProc: int, trainingParamsOverride: dict[str, Any]) -> None:
    """
    ML/RBF prediction plotting
    
    Parameters
    ----------
    domain : str                        String specifing the domain type (either "2D" or "axisym")
    plotTasks : list                    List of models to train / plot
    nProc : int                         Maximum number of concurrent prediction/plotting processes
    trainingParamsOverride : dict[str, Any]  Dictionary of trainingParams that will supersede loaded values

    Returns
    -------
    None
    """
    if len(plotTasks) == 0:                                                     # Do not continue if no models were requested for plotting
        print("Plotting was not requested for any ML/RBF model")
        return
    
    plotParams = loadyaml("plotParams")                                         # Load plotting parameters from plotParams.yaml
    plotParams["useTex"] = plotParams["useTex"] and not bool(call("tex --help", shell=True, stdout=DEVNULL, stderr=DEVNULL))  # If text rendering with TeX is enabled, check that TeX is present on the system and update the params dict accordingly
    trainingParams = loadyaml("trainingParams", override = trainingParamsOverride)  # Also load settings from the trainingParams.yaml
    modelPrefix["M"]["dimensionSize"] = trainingParams["modes"]                 # If model is M (modal), update the dimensionSize to be the number of modes specified in trainingParams
        
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
    varPlotArgs = getVarPlotTaskArgs(plotTasks, tensorDirs, plotParams)         # Get list of task arguments for plotting per-variable profiles
    lossPlotArgs = getLossPlotTaskArgs(plotTasks, tensorDirs, plotParams)       # Get list of task arguments for plotting per-variable loss
    benchmarkPlotArgs = getBenchmarkPlotTaskArgs(plotTasks, tensorDirs, plotParams)  # Get list of task arguments for plotting per-architecture loss benchmark
    historyPlotArgs = getLossPlotTaskArgs(plotTasks, tensorDirs, plotParams)    # Get list of task arguments for plotting evolution of loss for the optimisation process

    taskFuncs = sum([[fn for _ in args] for fn, args in zip([predictionPlot, varPlot, lossPlot, mlBenchmarkPlot, historyPlot],  # Create a list containing function objects for each element in all taskArgs lists
                                                            [predPlotArgs, varPlotArgs, lossPlotArgs, benchmarkPlotArgs, historyPlotArgs])], [])
    taskArgs = predPlotArgs + varPlotArgs + lossPlotArgs + benchmarkPlotArgs + historyPlotArgs  # Combine all function arguments into a single list with the same size as taskFuncs, to be passed to pool manager
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
    plotTasks : list                    List of models to plot
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
            
        VTReduced = torch.load(tensorDir / "VTReduced.pt", weights_only=True)   # Each boundary condition variable has its own set of modes, and therefore its own set of left eigenvectors, load the corresponding dictionary of tensors
        xData = torch.load(tensorDir / "xData.pt", weights_only=True)           # Load the xData dictionary of tensors
        x, xMin, xMax = xData["x"], xData["xMin"], xData["xMax"]                # Extract the min and max values of the denormalised xData, as well as the normalised xData
        xPredExpanded = torch.from_numpy(xPredExpandedArray1H if int(tensorDir.stem.split("_")[-1]) == 1 else xPredExpandedArray2H)  # Deduce the number of harmonics from the current tensorDir name, and convert the corresponding array to a tensor
        xPred, _, _ = normaliseTensor(xPredExpanded)                            # The models are trained with normalised data, so the features need to be normalised with the stored min and max values
        
        for modelName in plotTasks:                                             # Iterate over all tasks specified in --plot
            for stateDictDir in [path for path in (tensorDir / modelName).glob("*") if path.is_dir()]:  # Also iterate over all architecture-relevant state dictionary subdirectories
                try:
                    dataMin = torch.load(tensorDir / f"{modelPrefix[ modelName[0] ]['dimensionType']}Min.pt", weights_only=True)  # Load the tensor of output maximum values for the current model's dimension type
                    dataMax = torch.load(tensorDir / f"{modelPrefix[ modelName[0] ]['dimensionType']}Max.pt", weights_only=True)  # Load the tensor of output minimum values for the current model's dimension type
                    predictedTHPQual = predictTHP(xPred, modelName, modelPrefix[ modelName[0] ], dataMin, dataMax, stateDictDir, VTReduced)  # Call the model's corresponding THP prediction function, passing it the collected arguments (for 3D qualitative plot)
                    predictedTHPQuant = predictTHP(x, modelName, modelPrefix[ modelName[0] ], dataMin, dataMax, stateDictDir, VTReduced)  # Also carry out prediction for the 2D quantitative plots
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
    plotTasks : list                    List of models to plot
    tensorDirsReAll : list              List of Re_All tensor directories for which to predict / plot
    xPredExpandedArray1H : array-like   Array of unstandardised (expanded) x-values used for prediction for a 1-harmonic scenario
    xPredExpandedArray2H : array-like   Array of unstandardised (expanded) x-values used for prediction for a 2-harmonic scenario

    Returns
    -------
    tasksMultiRe : list                 List of tasks, where each entry is a partial list of parameters to be passed to lumpedPlot
    """
    tasksMultiRe = []                                                           # List of tasks, where each entry will be a partial list of parameters to be passed to lumpedPlot
    for tensorDirReAll in tqdm(tensorDirsReAll, desc="Predicting for multi-Re scenarios"):  # Iterate over all Re_All tensor directories previously identified by the corresponding single-Re function
        uniqueRe = torch.unique(torch.load(tensorDirReAll / "xData.pt", weights_only=True)["xExpanded"][:, -1]).int()  # Get an integer tensor of unique Re values from the last column of the xExpanded tensor (where Re is a feature)
        VTReduced = torch.load(tensorDirReAll / "VTReduced.pt", weights_only=True)  # Each boundary condition variable has its own set of modes, and therefore its own set of left eigenvectors, load the corresponding dictionary of tensors
        xData = torch.load(tensorDirReAll / "xData.pt", weights_only=True)      # Load the xData dictionary of tensors
        x, xMin, xMax = xData["x"], xData["xMin"], xData["xMax"]                # Extract the minimum and maximum values of the denormalised xData, as well as the normalised xData
        xPredExpandedArray = xPredExpandedArray1H if int(tensorDirReAll.stem.split("_")[-1]) == 1 else xPredExpandedArray2H  # Deduce the number of harmonics from the current tensorDir name, and set it as the chosen xPredExpanded array
        
        for modelName in plotTasks:                                             # Iterate over all tasks specified in --plot
            for stateDictDirReAll in [path for path in (tensorDirReAll / modelName).glob("*") if path.is_dir()]:  # Also iterate over all architecture-relevant state dictionary subdirectories
                try:
                    dataMin = torch.load(tensorDirReAll / f"{modelPrefix[ modelName[0] ]['dimensionType']}Min.pt", weights_only=True)  # Load the tensor of output min values for the current model's dimension type
                    dataMax = torch.load(tensorDirReAll / f"{modelPrefix[ modelName[0] ]['dimensionType']}Max.pt", weights_only=True)  # Load the tensor of output max values for the current model's dimension type
                    
                    for Re in uniqueRe:                                         # Iterate over all unique Re values (from the last column of the original xExpanded tensor)
                        xPredExpanded = torch.from_numpy(np.insert(xPredExpandedArray, xPredExpandedArray.shape[1], Re, axis=1))  # Insert the current Re value as the last column of the xPredExpanded array and convert it to a tensor
                        xPredQual, _, _ = normaliseTensor(xPredExpanded)        # The models are trained with normalised data, so the features need to be normalised with the stored mean and standard deviation values
                        xPredQuant = x[xData["xExpanded"][:, -1] == Re]         # Feature values for current Re to predict THP against corresponding HFM data
                        predictedTHPQual = predictTHP(xPredQual, modelName, modelPrefix[ modelName[0] ], dataMin, dataMax, stateDictDirReAll, VTReduced)  # Call the model's corresponding THP prediction function, passing it the collected arguments (for 3D qualitative plot)
                        predictedTHPQuant = predictTHP(xPredQuant, modelName, modelPrefix[ modelName[0] ], dataMin, dataMax, stateDictDirReAll, VTReduced)  # Also carry out prediction for the 2D quantitative plots
                        tasksMultiRe.append([predictedTHPQual, predictedTHPQuant, stateDictDirReAll, Re])  # Add the current model name, computed THP value, the current tensor directory, and the current Reynolds number to the list of mulit-Re task parameters

                except FileNotFoundError as e:
                    pathParts = e.filename.split("/")
                    print(f"No available {pathParts[-2]} checkpoint found for prediction in {pathParts[-3]} (try --train {pathParts[-2]})")
                
    return tasksMultiRe
    
def getVarPlotTaskArgs(plotTasks: list[str],
                       tensorDirs: list[Path],
                       plotParams: dict[str, Any]) -> list[dict[str, Any], Path]:
    """
    Scans the mlData directory tree for valid paths for which var plots need
        to be generated, and returns per-task var plot function arguments
    
    Parameters
    ----------
    plotTasks : list                    List of models to plot
    tensorDirs : list                   List of tensor directories
    plotParams : dict                   Dictionary of plotting parameters from YAML file

    Returns
    -------
    varPlotTaskArgs : list              List of per-task var plot function arguments
    """
    varPlotTaskArgs = []            
    plotTasks = [plotTask for plotTask in plotTasks if plotTask[0] != "L"]      # Subset of requested and possible plot tasks, which represents the models for which loss can and will be plotted
    for tensorDir in tqdm(tensorDirs, desc="Preparing variable profile plot data"):  # Iterate over all available tensor directories
        if (tensorDir / "variableProfiles.pt").is_file():                       # If outer variable profiles (HFM) tensor does not exist, ignore this tensor dir
            for plotTask in plotTasks:                                          # Iterate over all requested and available plot tasks
                for stateDictDir in (tensorDir / plotTask).glob("*"):           # Iterate also over all available subdirectories in each tensor directory (containing state dict directories)
                    if (stateDictDir / "optimalSearchResults.pt").is_file():    # If inner variable profile tensor (SM) exists, we can create a plot task
                        varPlotTaskArgs.append([plotParams, tensorDir / "variableProfiles.pt", stateDictDir / "optimalSearchResults.pt"])
    return varPlotTaskArgs

def getLossPlotTaskArgs(plotTasks: list[str],
                        tensorDirs: list[Path],
                        plotParams: dict[str, Any]) -> list[dict[str, Any], Path]:
    """
    Scans the mlData directory tree for valid paths for which loss plots need
        to be generated, and returns per-task loss plot function arguments
    
    Parameters
    ----------
    plotTasks : list                    List of models to plot
    tensorDirs : list                   List of tensor directories
    plotParams : dict                   Dictionary of plotting parameters from YAML file

    Returns
    -------
    lossPlotTaskArgs : list             List of per-task loss plot function arguments
    """
    lossPlotTaskArgs = []
    plotTasks = [plotTask for plotTask in plotTasks if plotTask[1:] in ["NN", "GP"]]  # Subset of requested and possible plot tasks, which represents the models for which loss can and will be plotted
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
    plotTasks : list                    List of models to plot
    tensorDirs : list                   List of tensor directories
    plotParams : dict                   Dictionary of plotting parameters from YAML file

    Returns
    -------
    lossPlotTaskArgs : list             List of per-task benchmark plot function arguments
    """
    lossPlotTaskArgs = []
    for tensorDir in tqdm(tensorDirs, desc="Preparing ML benchmark plot data"):  # Iterate over all available tensor directories
        for plotTask in plotTasks:                                              # Iterate over all requested and available NN plot tasks
        
            lossTable = {}
            for stateDictDir in (tensorDir / plotTask).glob("*"):               # Iterate also over all available subdirectories in each tensor directory (containing state dict directories)
                if stateDictDir.is_dir():                                       # If the state dict path is a valid directory
                                       
                    if plotTask[1:] == "NN":                                    # If current plotTask is a Neural Network, the lossTable key has to be formatted differently from other tasks
                        _, valSplit, _, layers, _, neurons, _, _ = stateDictDir.name.split("_")  # Extract validation split, layers and neurons from state dict path name
                        keyEntries = [valSplit, layers, neurons]                # First entries in lossTable key will be all extracted variables
                    else:                                                       # Key layout of lossTable for GP and RBF
                        _, valSplit, _, kernels, _, _ = stateDictDir.name.split("_")  # Extract validation split and kernels from state dict path name
                        keyEntries = [valSplit, kernels]                        # First entries in lossTable key will be all extracted variables
                        
                    for varStateDict in stateDictDir.glob("*.pt"):              # For each variable inside the state dict folder
                        if varStateDict.name == "optimalSearchResults.pt": 
                            continue
                        key = tuple(keyEntries + [varStateDict.stem])           # Dictionary key is the current keyEntries prefix with varStateDict stem as suffix
                        loss = torch.load(varStateDict)["lossTrain"][-1]        # Extract the last loss from the variable state dictionary
                        if key in lossTable:                                    # If this combination of architecture and flow variable has been seen before, a list of loss values already exists
                            lossTable[key].append(loss)                         # ... therefore, append to the list
                        else:                                                   # If this combination is new ...
                            lossTable[key] = [loss]                             # ... a list of per-sample losses needs to be created (including the now first-seen loss)
                                
            if lossTable:                                                       # If the loss table isn't empty, we can convert it to an array
                lossTableTemp = [[*key, np.array(value)] for key, value in lossTable.items() if len(value) > 0]  # Convert dictionary to a list of lists with columns [valSplit, layers, neurons, var, dataArr] (discarding any empty entries)
                lossPlotTaskArgs.append([plotParams, tensorDir / plotTask, np.array(lossTableTemp, dtype=object)])  # The temporary loss table is converted to an unstructured array of objects and will be an argument passed to the plot function
    return lossPlotTaskArgs
        
###############################################################################

if __name__ == "__main__":
    raise RuntimeError("This script is not intended for execution. Instead, execute hammerhead.py from the parent directory.")
