##############################################################################################
#                                                                                            #
#       HAMMERHEAD - Harmonic MOO Model Expedient for the Reduction                          #
#                    Hydraulic-loss/Heat-transfer Enhancement Asset Design                   #
#       Author: Daniela Minerva Segura Galeana                                               #
#       Supervised by: Antonio J. Gil                                                        #
#                                                                                            #
##############################################################################################
#                                                                                            #
#       Script containing Machine Learning (ML) prediction functions                         #
#                                                                                            #
##############################################################################################

# IMPORTS: HAMMERHEAD files ###############################################################################################################

from utilities.dataProcessing import (denormaliseTensor,                        # Utilities -> Reverting normalised data into original (expanded) data
                                      lumpedDataCalculation)                    # Utilities -> THP computation
from utilities.mlClasses import (Network,                                       # Utilities -> Neural Network class
                                 Kriging)                                       # Utilities -> Gaussian Process class

# IMPORTS: Others #########################################################################################################################

from pathlib import Path                                                        # Others -> Path manipulation tools
from scipy.optimize import differential_evolution                               # Others -> Stochastic minimisation algorithm
from typing import Any, Optional                                                # Others -> Python type hinting

import gpytorch                                                                 # Others -> Gaussian Process tools
import torch                                                                    # Others -> Neural Network tools

###########################################################################################################################################

def NN(modelName: str,
       featureSize: int,
       dimensionSize: int,
       xPred: torch.tensor,
       outputMin: torch.tensor,
       outputMax: torch.tensor,
       varName: str,
       stateDictDir: Path,
       VTReduced: torch.tensor = None,
       **kwargs) ->  tuple[torch.tensor, None, None]:
    """
    Neural Network (NN) prediction process
    
    Parameters
    ----------
    modelName : str                     Name of model (defines output size)
    featureSize : int                   Number of features in input tensor
    dimensionSize : int                 Dimension of the output tensor
    xPred  : torch.tensor               Feature tensor used for prediction
    outputMin : torch.tensor            Min values of the output tensor
    outputMax : torch.tensor            Max values of the output tensor
    varName : str                       Variable name (used for labelling stored training data)
    stateDictDir : str                  Trained data storage directory
    VTReduced : torch.tensor            PCA left eigenvector data

    Returns
    -------
    outputDataTensor : torch.tensor     Prediction output tensor
    outputUpperTensor : None            Upper confidence limit not computed by this model 
    outputLowerTensor : None            Lower confidence limit not computed by this model
    """
    checkpoint = torch.load(stateDictDir / f"{varName}.pt", weights_only=True)  # Load the checkpoint of the model
    _, _, _, layers, _, neurons, _, _ = stateDictDir.name.split("_")            # Infer number of layers and neurons from state dict path
    
    network = Network(featureSize, dimensionSize, int(neurons), int(layers))    # Create instance of neural network class
    network.load_state_dict(checkpoint['modelState'])                           # Load the latest state of the model
    network.eval()                                                              # Evaluate current state of the network to enable prediction
    
    with torch.no_grad():                                                       # Disable gradient calculation for prediction purposes
        data = network(xPred)                                                   # Produce the prediction
    dataExpanded = denormaliseTensor(data, outputMin, outputMax)                # Denormalise (expand) the data
    
    # Expand the output to spatial data if running modal prediction, otherwise (lumped or spatial) return the data as-is
    return torch.mm(dataExpanded, VTReduced) if modelName[0] == "M" else dataExpanded, None, None 

def RBF(modelName: str,
        xPred: torch.tensor,
        outputMin: torch.tensor,
        outputMax: torch.tensor,
        varName: str,
        stateDictDir: Path,
        VTReduced: torch.tensor,
        **kwargs) ->  tuple[torch.tensor, None, None]:
    """
    Radial Basis Function (RBF) interpolator prediction process
    
    Parameters
    ----------
    modelName : str                     Name of model (defines output size)
    xPred  : torch.tensor               Feature tensor used for prediction
    outputMin : torch.tensor            Min values of the output tensor
    outputMax : torch.tensor            Max values of the output tensor
    varName : str                       Variable name (used for labelling stored training data)
    stateDictDir : str                  Trained data storage directory
    VTReduced : torch.tensor            PCA left eigenvector data

    Returns
    -------
    outputDataTensor : torch.tensor     Prediction output tensor
    outputUpperTensor : None            Upper confidence limit not computed by this model
    outputLowerTensor : None            Lower confidence limit not computed by this model
    """
    rbfi = torch.load(stateDictDir / f"{varName}.pt", weights_only=False)['modelState']  # Load the stored trained RBFI object
    
    data = torch.from_numpy(rbfi(xPred.detach())).float()                       # Produce the prediction
    dataExpanded = denormaliseTensor(data, outputMin, outputMax)                # Denormalise (expand) the data
    
    # Expand the output to spatial data if running modal prediction, otherwise (lumped or spatial) return the data as-is
    return torch.mm(dataExpanded, VTReduced) if modelName[0] == "M" else dataExpanded, None, None 

    
def GP(modelName: str,
       featureSize: int,
       dimensionSize: int,
       xPred: torch.tensor,
       outputMin: torch.tensor,
       outputMax: torch.tensor,
       varName: str,
       stateDictDir: Path,
       VTReduced: torch.tensor = None,
       **kwargs) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    Gaussian Process (GP) prediction process
    
    Parameters
    ----------
    modelName : str                     Name of model (defines output size)
    featureSize : int                   Number of features in input tensor
    dimensionSize : int                 Dimension of the output tensor
    xPred  : torch.tensor               Feature tensor used for prediction
    outputMin : torch.tensor            Min values of the output tensor
    outputMax : torch.tensor            Max values of the output tensor
    varName : str                       Variable name (used for labelling stored training data)
    stateDictDir : str                  Trained data storage directory
    VTReduced : torch.tensor            PCA left eigenvector data

    Returns
    -------
    outputDataTensor : torch.tensor     Prediction output mean tensor
    outputUpperTensor : torch.tensor    Prediction output upper confidence limit tensor
    outputLowerTensor : torch.tensor    Prediction output lower confidence limit tensor
    """
    trainingData = torch.load(stateDictDir / f"{varName}.pt", weights_only=False)  # Load the dictionary of model parameters and training data
    xTrain, outputTrain = trainingData["xTrain"], trainingData["outputTrain"]   # Load the features and output that the model was trained with from the file
    kernel = getattr(gpytorch.kernels, (stateDictDir.name.split("_"))[3])       # Infer the kernel the model was trained with from the directory name
    
    likelihoodPrev = [gpytorch.likelihoods.GaussianLikelihood(noise=torch.tensor(1e-7)) for _ in range(dimensionSize)]  # GPyTorch works with a single output, a list of likelihoods is produced for a multi-output model
    modelPrev = [Kriging(xTrain, outputTrain[:, i], likelihoodPrev[i], kernel, featureSize, dimensionSize) for i in range(dimensionSize)]  # GPyTorch works with a single output, a list of models is produced for a multi-output model
    modelLikelihood = [independentModel.likelihood for independentModel in modelPrev]  # Link each output point likelihood function to an output point model
    
    model = gpytorch.models.IndependentModelList(*modelPrev)                    # Create a Gaussian Process model from modelPrev (list of sub-models)
    likelihood = gpytorch.likelihoods.LikelihoodList(*modelLikelihood)          # Create likelihood object from list of previously created list of likelihoods 
    
    model.load_state_dict(torch.load(stateDictDir / f"{varName}.pt", weights_only=False)["modelState"])  # Load the model state from the trained data tensors
    
    model.eval()                                                                # Evaluate current state of the model to enable prediction
    likelihood.eval()                                                           # Evaluate current state of the likelihood to enable prediction

    with torch.no_grad(), gpytorch.settings.lazily_evaluate_kernels(state=False), gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False):  # Disable gradient calculation for prediction purposes
        predictions = likelihood(*model(*[xPred for _ in range(dimensionSize)]))  # Produce the multi-output prediction including confidence region

    data = torch.column_stack([prediction.mean for prediction in predictions])  # Extract the mean value from the prediction
    confidence = [torch.column_stack([prediction.confidence_region()[i] for prediction in predictions]) for i in range(2)]  # Extract the standardised upper and lower confidence limit tensors
    dataExpanded = [denormaliseTensor(tensor.float(), outputMin, outputMax) for tensor in [data, *confidence]]  # Denormalise (expand) the data and confidence limits
    
    # Expand the output to spatial data if running modal prediction, otherwise (lumped or spatial) return the data as-is
    return tuple(torch.mm(tensor, VTReduced) if modelName[0] == "M" else tensor for tensor in dataExpanded)

def predictTHP(xPred: torch.tensor,
               modelName: str,
               modelDict: dict[str, Any],
               outputMin: dict[str, torch.tensor],
               outputMax: dict[str, torch.tensor],
               stateDictDir: Path,
               VTReduced: dict[str, torch.tensor],
               maximise: Optional[bool] = False,
               varProfile: Optional[bool] = False) -> list[Optional[torch.tensor]]:
    """
    Compute a Thermo-Hydraulic Performance (THP) prediction with the boundary
        conditions variables (BCV) of the specified model

    Parameters
    ----------
    xPred : torch.tensor                Feature tensor used for prediction
    modelName : str                     Name of current model
    modelDict : dict                    Dictionary of current model attributes
    outputMin : dict                    Minimum of the output tensor
    outputMax : dict                    Maximum of the output tensor
    stateDictDir : pathlib.Path         Trained data storage directory
    VTReduced : dict                    Dictionary of PCA left eigenvectors, one for each BCV
    maximise : bool                     If True, this function is being called as part of differential evolution
    varProfile : bool                   If True, returns flow variable profiles instead of lumped THP

    Returns
    -------
    predictedTHPData : torch.tensor     Thermo-Hydraulic Performance evaluation
    predictedTHPUpper : torch.tensor    Thermo-Hydraulic Performance evaluation upper confidence limit
    predictedTHPLower : torch.tensor    Thermo-Hydraulic Performance evaluation lower confidence limit
    """
    if maximise:                                                                # If running as part of a maximisation process, passed xPred is a 1D numpy array
        xPred = torch.from_numpy(xPred).float().unsqueeze(0)                    # Convert 1D numpy array to a 2D tensor, for compatibility with existing behaviour

    mlPrediction = [globals()[modelName[1:]](                                   # Call the model's corresponding BCV prediction function, passing it the collected arguments and constructing a list of BCV
                        modelName=modelName, featureSize=xPred.shape[1], dimensionSize=modelDict["dimensionSize"],
                        xPred=xPred, outputMin=outputMin[var], outputMax=outputMax[var],
                        varName=var, stateDictDir=stateDictDir, VTReduced=VTReduced.get(var))
                    for var in modelDict["variables"]]
    mlPrediction = list(zip(*mlPrediction))                                     # Unwrap list [[data1, upper1, lower1], [data2, upper2, lower2], ...] -> [[data1, data2, ...], [upper1, upper2, ...], [lower1, lower2, ...]]
    mlPrediction = [(None if data[0] is None else data) for data in mlPrediction]  # Convert any lists of None to a single None for ease of processing downstream
    
    if modelName[0] != "L" and not varProfile:                                  # If the current model isn't lumped (prefix is not L), and we are not returning flow variables ...
        mlPrediction = [(None if data is None else lumpedDataCalculation(data)) for data in mlPrediction]  # ... compute the Thermo-Hydraulic Performance (integral of advected heat flux and dissipation rate)
    
    if maximise:                                                                # If running as part of a maximisation process, need to return a single value
        return -(mlPrediction[0][0]-mlPrediction[0][1])[0, 0]                   # Compute and return lumped THP evaluation to maximisation process
        
    if varProfile:                                                              # If we are returning variable profile, we know we are already NOT lumped (no need to perform additional check)
        return [(None if data is None else data) for data in mlPrediction]      # Return flow variable profiles
        
    return [(None if data is None else data[0] - data[1]) for data in mlPrediction]  # Return the THP evaluation (subtracting dissipation rate from advected heat flux)

def maximiseTHP(modelName: str,
                modelDict: dict,
                stateDictDir: Path,
                tensorDir: Path) -> tuple[str, str, str, str, str]:
    """
    Find optimal features by minimising predicted THP

    Parameters
    ----------
    modelName : str                     Name of current model
    modelDict : dict                    Dictionary of current model attributes
    tensorDir : pathlib.Path            Trained data storage directory

    Returns
    -------
    modelName : str                     Name of model (used for post-search report, needs to be returned because of asynchronous search)
    archString : str                    Architecture string (used for post-search report)
    Re : str                            Reynolds number (used for post-search report)
    harmonics : str                     Number of harmonics that were used for search
    optimalFeatureVals : str            Summary string of optimal feature values
    """
    
    xData = torch.load(tensorDir / "xData.pt", weights_only=True)               # Load the xData dictionary of tensors
    xMin, xMax = xData["xMin"], xData["xMax"]                                   # Extract the min and max values of the denormalised xData
    outputMin = torch.load(tensorDir / f"{modelDict['dimensionType']}Min.pt", weights_only=True)  # Load the tensor of output min values for the current model's dimension type
    outputMax = torch.load(tensorDir / f"{modelDict['dimensionType']}Max.pt", weights_only=True)  # Load the tensor of output max values for the current model's dimension type
    VTReduced = torch.load(tensorDir / "VTReduced.pt", weights_only=True)       # Each boundary condition variable has its own set of modes, and therefore its own set of left eigenvectors, load the corresponding dictionary of tensors
    
    thpHistory = []                                                             # Store intermediate results from THP minimisation
    xObjective = differential_evolution(predictTHP,                             # Perform THP minimisation by differential evolution of predictTHP function
                                        args=(modelName, modelDict, outputMin, outputMax, stateDictDir, VTReduced, True),  # Arguments for predictTHP function
                                        bounds = torch.stack((torch.zeros_like(xMin), torch.ones_like(xMax))).T,  # Bounds are the per-feature minima and maxima (0 and 1)
                                        callback=lambda intermediate_result: thpHistory.append(intermediate_result.fun),  # Callback to append intermediate minimisation results to thpHistory list
                                        polish=True)                            # Improve post-minimisation result
    xPredMax = torch.from_numpy(xObjective["x"]).float()                        # Convert the 1D numpy array (optimised shape features) to a 2D tensor
    xPredExpanded = denormaliseTensor(xPredMax, xMin, xMax)                     # ... and denormalise (expand) it

    if modelName[0] != "L":                                                     # If the current model isn't lumped, we can also compute variable profiles
        xPredBaseline = torch.zeros(xMin.shape[0])                              # Shape features of baseline case (all zeros)
        if xPredBaseline.shape[0] in [3, 5]:                                    # If Re_All (number of features is 3 or 5), final entry is Re, which is NOT zero for baseline case
            xPredBaseline[-1] = xPredMax[-1]                                    # Set Re baseline feature to optimised Re value
        profileBaseline = predictTHP(xPredBaseline.unsqueeze(0), modelName, modelDict, outputMin, outputMax, stateDictDir, VTReduced, varProfile=True)  # Compute baseline variable profiles
        profileOptimised = predictTHP(xPredMax.unsqueeze(0), modelName, modelDict, outputMin, outputMax, stateDictDir, VTReduced, varProfile=True)  # Compute optimised variable profiles
    
    torch.save({'xPredExpanded': xPredExpanded,
                'thpHistory': thpHistory,
                'profileBaseline': None if modelName[0] == "L" else [profileBaseline[0][i].squeeze() for i in [2, 3, 4]],  # (ignore upper and lower limits, just store list of variable tensors)
                'profileOptimised': None if modelName[0] == "L" else [profileOptimised[0][i].squeeze() for i in [2, 3, 4]]},  # (select only: outletU [2], outletT [3], inletp [4])
                Path(stateDictDir) / "optimalSearchResults.pt")                 # Store optimal search results
                
    _, Re, _, _, _, harmonics = tensorDir.stem.split("_")                       # Extract the Reynolds number and number of harmonics from the tensorDir name
    featureLabels = ["A1", "A2", "k1", "k2", "Re"][:(5 if Re == "All" else 4):(2 if harmonics == "1" else 1)]  # Match feautre positions with their corresponding labels
    labelledFeatureValues = [f"{label}={value.item():.4f}" for label, value in zip(featureLabels, xPredExpanded)]  # Construct of a list of strings where each entry is a labelled feature with its optimal value
    archParts =  stateDictDir.name.split("_")[:-2]                              # List of architecture parts [name1, value1, name2, value2, ...]
    archString = ", ".join([f"{archParts[i]}={archParts[i+1]}" for i in range(0, len(archParts), 2)])  # Compute architecture string as "name1=value1, name2=value2, ..."
        
    return modelName, archString, Re, harmonics, ", ".join(labelledFeatureValues)
    
###############################################################################

if __name__ == "__main__":
    raise RuntimeError("This script is not intended for execution. Instead, execute hammerhead.py from the parent directory.")
