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
from utilities.plotFunctions import lumpedPlot                                  # Utilities -> Plotting tools

import utilities.mlTraining as train                                            # Utilities -> ML training process functions

# IMPORTS: Others #########################################################################################################################

from pathlib import Path                                                        # Others -> Path manipulation
from subprocess import call, DEVNULL                                            # Others -> External process manipulation
from tqdm import tqdm                                                           # Others -> Progress bar

import numpy as np                                                              # Others -> Array manipulation functions
import random                                                                   # Others -> Random sampling
import torch                                                                    # Others -> Tensor manipulation functions

###########################################################################################################################################

bcvNames = ["inletU", "inletT", "inletp", "outletU", "outletT", "outletp"]      # Boundary Condition Variable (BCV) names
thpNames = ["lumpedT", "lumpedp"]                                               # Thermo-Hydraulic Performance (THP) variable names used by lumped model

models = {
    "kriging": {
        "dimensionType": "modal",
        "dimensionSize": None,
        "variables":     bcvNames,
        "function":      "GP"},
    "lumped": {
        "dimensionType": "lumped",
        "dimensionSize": 1,
        "variables":     thpNames,
        "function":      "NN"},
    "modal": {
        "dimensionType": "modal",
        "dimensionSize": None,
        "variables":     bcvNames,
        "function":      "NN"},
    "rbf": {
        "dimensionType": "modal",
        "dimensionSize": None,
        "variables":     bcvNames,
        "function":      "RBF"},
    "spatial": {
        "dimensionType": "spatial",
        "dimensionSize": 101,
        "variables":     bcvNames,
        "function":      "NN"}}

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
    
    trainingParams = loadyaml("trainingParams")                                 # Load the training parameters from ./resources/trainingParams.yaml; edit this file to use different parameters
    for model in ["kriging", "modal", "rbf"]:                                   # These models have parameters that depend on data loaded from trainingParams, iterate over them
        models[model]["dimensionSize"] = trainingParams["modes"]                # ... for each model, update the dimensionSize to be the number of modes specified in trainingParams
    
    tensorDirs = [tensorDir for tensorDir in Path(f"./mlData/{domain}").glob("*")  # Get list of paths for all tensor directories in ./mlData/{domain}
                  if (tensorDir.is_dir()                                        # ... filter out all non-directory entries
                      and int(tensorDir.stem.split("_")[3]) == trainingParams["modes"])]  # ... only accept directories that contain tensors with the requested number of modes
    if len(tensorDirs) == 0:                                                    # If there are no tensor directories available for the requested domain and number of modes in ./mlData, exit
        print(f"No modes={trainingParams['modes']} tensor directories found in ./mlData/{domain} for model training process!")
        return

    taskFuncs = []                                                              # List of training function callables that will be passed to the parallelised trainging worker
    taskKwargsList = []                                                         # ... corresponding list of function keyword arguments stored in dictionaries
    for tensorDir in sorted(tensorDirs):                                        # Iterate over all tensor directories mathching requirements loaded from trainingParams.yaml
        print(f"Loading tensors for Re={tensorDir.stem.split('_')[1]} and modes={trainingParams['modes']} from {tensorDir} for ML/RBF training")
        tensors = {tensor: torch.load(tensorDir / f"{tensor}Data.pt") for tensor in ["modal", "spatial", "lumped"]}  # Load the necessary tensors and store them in a dictionary by name
        tensors["x"] = torch.load(tensorDir / "xData.pt")["x"]                  # Additionally, load the standardised x tensor from xData.pt
        
        dataSize, featureSize = tensors["x"].shape                              # Shape of the full data set, where shape[0] is the number of processed cases and shape[1] is the number of features
        validSize = int(np.floor(trainingParams["validationSplit"] * dataSize))  # Compute the size of the validation data set, default is 35% of full data set for validation
        indices = random.sample(range(dataSize), k=dataSize)                    # Get a randomly arranged list of indices for the full data set
        trainIdx, validIdx = indices[validSize:], indices[:validSize]           # Split all indices into two groups: indices that will address the training and the validation data sets

        xTrain = tensors["x"][trainIdx, :]                                      # Select the training cases from the xData tensor
        xValid = tensors["x"][validIdx, :]                                      # Select the validation cases from the xData tensor

        for name, model in [(task, models[task]) for task in trainTasks]:       # Iterate over all tasks specified in --train, returning the name of the model for training, and its respective entry in the dictionary
            for var in model["variables"]:                                      # Also iterate over all variables available for this model
                outputTrain = tensors[ model["dimensionType"] ][ var ][trainIdx, :]  # ... compute the output training tensor
                outputValid = tensors[ model["dimensionType"] ][ var ][validIdx, :]  # ... compute the output validation tensor
                taskFuncs.append(getattr(train, model["function"]))             # ... get the model function callable and the keyword argument dictionary that will be passed to it, and store them in their respective task lists
                taskKwargsList.append({"name": name, "featureSize": featureSize, "dimensionSize": model["dimensionSize"],
                                       "layers": trainingParams["layers"], "neurons": trainingParams["neurons"],
                                       "xTrain": xTrain, "outputTrain": outputTrain, "xValid": xValid, "outputValid": outputValid,
                                       "varName": var, "tensorDir": tensorDir})
    genericPoolManager(taskFuncs, None, taskKwargsList, nProc, "Training requested ML/RBF models", "Completed training of {} on Re={} harmonics={} {} variable")  # Send all tasks to the multi-threaded worker function
   
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
    for model in ["kriging", "modal", "rbf"]:                                   # These models have parameters that depend on data loaded from trainingParams, iterate over them
        models[model]["dimensionSize"] = trainingParams["modes"]                # ... for each model, update the dimensionSize to be the number of modes specified in trainingParams (if no other function has done this yet)
        
    tensorDirs = [tensorDir for tensorDir in Path(f"./mlData/{domain}").glob("*")  # Get list of paths for all tensor directories in ./mlData/{domain}
                  if (tensorDir.is_dir() and int(tensorDir.stem.split("_")[3]) == trainingParams["modes"])]  # ... filter out all non-directory entries and only accept directories that contain tensors with the requested number of modes
    if len(tensorDirs) == 0:                                                    # If there are no tensor directories available for the requested domain and number of modes in ./mlData, exit
        print(f"No modes={trainingParams['modes']} tensor directories found in ./mlData/{domain} for prediction/plotting process!")
        return
    
    plotDir = Path(f"./plots/{domain}")                                         # Directory where all plots are stored
    xv, yv = np.linspace(0.001, 0.010, 30), np.linspace(2, 60, 30)              # X- and Y-value arrays used as input for the prediction tasks and for 3D surface plotting
    
    xPredExpandedArray1H = np.array(np.meshgrid(xv, yv), dtype=np.float32).T.reshape(-1, 2)  # Array of unstandardised (expanded) x-values used for prediction for a 1-harmonic scenario
    xPredExpandedArray2H = np.insert(xPredExpandedArray1H, [1, 2], [plotParams["A2"], plotParams["k2"]], axis=1)  # Array of unstandardised (expanded) x-values used for prediction for a 2-harmonic scenario
    
    tasksSingleRe, tensorDirsReAll = getPlotTasksSingleRe(plotTasks, tensorDirs, xPredExpandedArray1H, xPredExpandedArray2H,
                                                          trainingParams["layers"], trainingParams["neurons"])  # Get a partial list of task arguments for all single-Re scenarios, as well as a list of multi-Re tensor directories
    tasksMultiRe = getPlotTasksMultiRe(plotTasks, tensorDirsReAll, xPredExpandedArray1H, xPredExpandedArray2H, trainingParams["layers"], trainingParams["neurons"])  # Get a partial list of task arguments for all multi-Re scenarios
    
    taskArgs = [[xv, yv, plotDir, plotParams] + task for task in tasksSingleRe + tasksMultiRe]  # Merge the two lists of task arguments, adding other necessary parameters that exist locally 
    taskFuncs = [lumpedPlot for _ in taskArgs]                                  # All function calls are made to a single function, create a list containing only that function for each element in taskArgs
    genericPoolManager(taskFuncs, taskArgs, None, nProc, "Plotting", "Completed plotting of {} with Re={} harmonics={}")  # Send all tasks to the multi-threaded worker function
    print("Plotting process completed successfully")
    
def getPlotTasksSingleRe(plotTasks: list[str],
                         tensorDirs: list[Path],
                         xPredExpandedArray1H: np.ndarray,
                         xPredExpandedArray2H: np.ndarray,
                         layers: int,
                         neurons: int) -> tuple[list[list[str, torch.tensor, Path]], list[Path]]:
    """
    ML/RBF prediction task collection function for a single-Re scenario
    
    Parameters
    ----------
    plotTasks : list                    List of models to train / plot
    tensorDirs : list                   List of tensor directories for which to predict / plot
    xPredExpandedArray1H : array-like   Array of unstandardised (expanded) x-values used for prediction for a 1-harmonic scenario
    xPredExpandedArray2H : array-like   Array of unstandardised (expanded) x-values used for prediction for a 2-harmonic scenario
    layers: int                         Number of NN hidden layers
    neurons: int                        Number of NN neurons per layer

    Returns
    -------
    tasksSingleRe : list                List of tasks, where each entry is a partial list of parameters to be passed to lumpedPlot
    tensorDirsReAll : list              List of paths corresponding to all relevant Re_All tensor directories (usually length 2, one for each harmonics value)
    """
    tasksSingleRe = []                                                          # List of tasks, where each entry will be a partial list of parameters to be passed to lumpedPlot
    tensorDirsReAll = []                                                        # List that will contain paths corresponding to all relevant Re_All tensor directories
    for tensorDir in tqdm(sorted(tensorDirs), desc="Predicting for single-Re scenarios"):  # Iterate over all tensor directories previously matching requirements loaded from trainingParams.yaml

        if tensorDir.stem.startswith("Re_All"):                                 # If tensor directory starts with Re_All, it is a multi-Re scenario that won't be processed here
            tensorDirsReAll.append(tensorDir)                                   # ... add it to a list that will be passed to the corresonding multi-Re function
            continue                                                            # ... and move on to the next tensor directory
            
        VTReduced = torch.load(tensorDir / "VTReduced.pt")                      # Each boundary condition variable has its own set of modes, and therefore its own set of left eigenvectors, load the corresponding dictionary of tensors
        xData = torch.load(tensorDir / "xData.pt")                              # Load the xData dictionary of tensors
        xMean, xStd = xData["xMean"], xData["xStd"]                             # Extract the mean and standard deviation of the unstandardised xData
        xPredExpanded = torch.from_numpy(xPredExpandedArray1H if int(tensorDir.stem.split("_")[-1]) == 1 else xPredExpandedArray2H)  # Deduce the number of harmonics from the current tensorDir name, and convert the corresponding array to a tensor
        xPred = (xPredExpanded - xMean) / xStd                                  # The models are trained with normalised data, so the features need to be normalised with the stored mean and standard deviation values
        
        for name, model in [(task, models[task]) for task in plotTasks]:        # Iterate over all tasks specified in --plot, returning the name of the model for training, and its respective entry in the dictionary
            dataMean = torch.load(tensorDir / f"{model['dimensionType']}Mean.pt")  # Load the tensor of output mean values for the current model's dimension type
            dataStd = torch.load(tensorDir/ f"{model['dimensionType']}Std.pt")  # Load the tensor of output standard deviation values for the current model's dimension type
            lumpedPred = predictTHP(model, name, layers, neurons, xPred, dataMean, dataStd, tensorDir, VTReduced)  # Call the model's corresponding THP prediction function, passing it the collected arguments
            tasksSingleRe.append([name, lumpedPred, tensorDir])                 # Add the current model name, computed THP value, and the current tensor directory to the list of single-Re task parameters
    return tasksSingleRe, tensorDirsReAll
            
def getPlotTasksMultiRe(plotTasks: list[str],
                        tensorDirsReAll: list[Path],
                        xPredExpandedArray1H: np.ndarray,
                        xPredExpandedArray2H: np.ndarray,
                        layers: int,
                        neurons: int) -> list[list[str, torch.tensor, Path]]:
    """
    ML/RBF prediction task collection function for a multi-Re scenario
    
    Parameters
    ----------
    plotTasks : list                    List of models to train / plot
    tensorDirsReAll : list              List of Re_All tensor directories for which to predict / plot
    xPredExpandedArray1H : array-like   Array of unstandardised (expanded) x-values used for prediction for a 1-harmonic scenario
    xPredExpandedArray2H : array-like   Array of unstandardised (expanded) x-values used for prediction for a 2-harmonic scenario
    layers: int                         Number of NN hidden layers
    neurons: int                        Number of NN neurons per layer

    Returns
    -------
    tasksMultiRe : list                 List of tasks, where each entry is a partial list of parameters to be passed to lumpedPlot
    """
    tasksMultiRe = []                                                           # List of tasks, where each entry will be a partial list of parameters to be passed to lumpedPlot
    for tensorDirReAll in tqdm(tensorDirsReAll, desc="Predicting for multi-Re scenarios"):  # Iterate over all Re_All tensor directories previously identified by the corresponding single-Re function
        uniqueRe = torch.unique(torch.load(tensorDirReAll / "xData.pt")["xExpanded"][:, -1])  # Get a tensor of unique Re values from the last column of the xExpanded tensor (where Re is a feature)
        VTReduced = torch.load(tensorDirReAll / "VTReduced.pt")                 # Each boundary condition variable has its own set of modes, and therefore its own set of left eigenvectors, load the corresponding dictionary of tensors
        xData = torch.load(tensorDirReAll / "xData.pt")                         # Load the xData dictionary of tensors
        xMean, xStd = xData["xMean"], xData["xStd"]                             # Extract the mean and standard deviation of the unstandardised xData
        xPredExpandedArray = xPredExpandedArray1H if int(tensorDirReAll.stem.split("_")[-1]) == 1 else xPredExpandedArray2H  # Deduce the number of harmonics from the current tensorDir name, and set it as the chosen xPredExpanded array
        
        for name, model in [(task, models[task]) for task in plotTasks]:        # Iterate over all tasks specified in --plot, returning the name of the model for training, and its respective entry in the dictionary
            dataMean = torch.load(tensorDirReAll / f"{model['dimensionType']}Mean.pt")  # Load the tensor of output mean values for the current model's dimension type
            dataStd = torch.load(tensorDirReAll / f"{model['dimensionType']}Std.pt")  # Load the tensor of output standard deviation values for the current model's dimension type
            
            for Re in uniqueRe:                                                 # Iterate over all unique Re values (from the last column of the original xExpanded tensor)
                xPredExpanded = torch.from_numpy(np.insert(xPredExpandedArray, xPredExpandedArray.shape[1], Re, axis=1))  # Insert the current Re value as the last column of the xPredExpanded array and convert it to a tensor
                xPred = (xPredExpanded - xMean) / xStd                          # The models are trained with normalised data, so the features need to be normalised with the stored mean and standard deviation values
                lumpedPred = predictTHP(model, name, layers, neurons, xPred, dataMean, dataStd, tensorDirReAll, VTReduced)  # Call the model's corresponding THP prediction function, passing it the collected arguments
                tasksMultiRe.append([name, lumpedPred, tensorDirReAll, Re])     # Add the current model name, computed THP value, the current tensor directory, and the current Reynolds number to the list of mulit-Re task parameters
    return tasksMultiRe

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
    for model in ["kriging", "modal", "rbf"]:                                   # These models have parameters that depend on data loaded from trainingParams, iterate over them
        models[model]["dimensionSize"] = trainingParams["modes"]                # ... for each model, update the dimensionSize to be the number of modes specified in trainingParams
        
    tensorDirs = [tensorDir for tensorDir in Path(f"./mlData/{domain}").glob("*")  # Get list of paths for all tensor directories in ./mlData/{domain}
                  if (tensorDir.is_dir() and int(tensorDir.stem.split("_")[3]) == trainingParams["modes"])]  # ... filter out all non-directory entries and only accept directories that contain tensors with the requested number of modes
    if len(tensorDirs) == 0:                                                    # If there are no tensor directories available for the requested domain and number of modes in ./mlData, exit
        print(f"No modes={trainingParams['modes']} tensor directories found in ./mlData/{domain} for optimal search process!")
        return                 
    
    taskFuncs = []                                                              # List of functions that will be passed to the parallelised pool manager
    taskArgs = []                                                               # ... corresponding list of function arguments
    for tensorDir in sorted(tensorDirs):                                        # Iterate over all tensor directories mathching requirements loaded from trainingParams.yaml
        for name, model in [(task, models[task]) for task in searchTasks]:      # Iterate over all tasks specified in --plot, returning the name of the model for training, and its respective entry in the dictionary
            taskFuncs.append(maximiseTHP)                                       # Every task will call the same maximiseTHP function
            taskArgs.append([model, name, tensorDir])                           # Create list of arguments that will be passed to the maximiseTHP function, and append it to the list of task arguments
    genericPoolManager(taskFuncs, taskArgs, None, nProc, "Searching for optimal feature values", "Completed optimal search using {} model with Re={} harmonics={}: {}")  # Send all tasks to the multi-threaded worker function
    print("Optimal search process completed successfully")
        
###############################################################################

if __name__ == "__main__":
    raise RuntimeError("This script is not intended for execution. Instead, execute hammehead.py from the parent directory.")
