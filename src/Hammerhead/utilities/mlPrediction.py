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

from utilities.dataProcessing import (unstandardiseTensor,                      # Utilities -> Reverting standardised data into original (expanded) data
                                      lumpedDataCalculation)                    # Utilities -> THP computation
from utilities.mlClasses import (Network,                                       # Utilities -> Neural Network class
                                 Kriging)                                       # Utilities -> Gaussian Process class
from utilities.plotFunctions import varPlot

# IMPORTS: Others #########################################################################################################################

from pathlib import Path                                                        # Others -> Path manipulation tools
from scipy.optimize import minimize, line_search, least_squares, differential_evolution, nnls, basinhopping
from typing import Any, Optional                                                # Others -> Python type hinting

import gpytorch                                                                 # Others -> Gaussian Process tools
import torch                                                                    # Others -> Neural Network tools

###########################################################################################################################################

def NN(name: str,
       featureSize: int,
       dimensionSize: int,
       xPred: torch.tensor,
       outputMean: torch.tensor,
       outputStd: torch.tensor,
       varName: str,
       stateDictDir: Path,
       VTReduced: torch.tensor = None,
       **kwargs) ->  tuple[torch.tensor, None, None]:
    """
    Neural Network (NN) prediction process
    
    Parameters
    ----------
    name : str                          Name of model (used for loading training data)
    featureSize : int                   Number of features in input tensor
    dimensionSize : int                 Dimension of the output tensor
    xPred  : torch.tensor               Feature tensor used for prediction
    outputMean : torch.tensor           Mean of output tensor
    outputStd : torch.tensor            Stadard deviation of the output tensor
    varName : str                       Variable name (used for labelling stored training data)
    stateDictDir : str                  Trained data storage directory
    VTReduced : torch.tensor            PCA left eigenvector data

    Returns
    -------
    outputDataTensor : torch.tensor     Prediction output tensor
    outputUpperTensor : None            Upper confidence limit not computed by this model 
    outputLowerTensor : None            Lower confidence limit not computed by this model
    """
    checkpoint = torch.load(stateDictDir / f"{varName}.pt")                     # Load the checkpoint of the model
    _, _, _, layers, _, neurons, _, _ = stateDictDir.name.split("_")            # Infer number of layers and neurons from state dict path
    
    network = Network(featureSize, dimensionSize, int(neurons), int(layers))    # Create instance of neural network class
    network.load_state_dict(checkpoint['modelState'])                           # Load the latest state of the model
    network.eval()                                                              # Evaluate current state of the network to enable prediction
    
    with torch.no_grad():                                                       # Disable gradient calculation for prediction purposes
        data = network(xPred)                                                   # Produce the prediction
    dataExpanded = unstandardiseTensor(data, outputMean, outputStd)             # Unstandardise (expand) the data
    
    return torch.mm(dataExpanded, VTReduced) if name == "mnn" else dataExpanded, None, None  # Expand the output to spatial data if running modal prediction, otherwise (lumped or spatial) return the data as-is

def RBF(name: str,
        xPred: torch.tensor,
        outputMean: torch.tensor,
        outputStd: torch.tensor,
        varName: str,
        stateDictDir: Path,
        VTReduced: torch.tensor,
        **kwargs) ->  tuple[torch.tensor, None, None]:
    """
    Radial Basis Function (RBF) interpolator prediction process
    
    Parameters
    ----------
    name : str                          Name of model (used for loading training data)
    xPred  : torch.tensor               Feature tensor used for prediction
    outputMean : torch.tensor           Mean of output tensor
    outputStd : torch.tensor            Stadard deviation of the output tensor
    varName : str                       Variable name (used for labelling stored training data)
    stateDictDir : str                  Trained data storage directory
    VTReduced : torch.tensor            PCA left eigenvector data

    Returns
    -------
    outputDataTensor : torch.tensor     Prediction output tensor
    outputUpperTensor : None            Upper confidence limit not computed by this model
    outputLowerTensor : None            Lower confidence limit not computed by this model
    """
    rbfi = torch.load(stateDictDir / f"{varName}.pt")                           # Load the stored trained RBFI object (interally uses pickle)
    
    data = torch.from_numpy(rbfi(xPred.detach())).float()                       # Produce the prediction
    dataExpanded = unstandardiseTensor(data, outputMean, outputStd)             # Unstandardise (expand) the data
    return torch.mm(dataExpanded, VTReduced) if name == "mrbf" else dataExpanded, None, None  # Expand the output to spatial data if running modal prediction, otherwise (lumped or spatial) return the data as-is

    
def GP(name: str,
       featureSize: int,
       dimensionSize: int,
       xPred: torch.tensor,
       outputMean: torch.tensor,
       outputStd: torch.tensor,
       varName: str,
       stateDictDir: Path,
       VTReduced: torch.tensor = None,
       **kwargs) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    Gaussian Process (GP) prediction process
    
    Parameters
    ----------
    featureSize : int                   Number of features in input tensor
    dimensionSize : int                 Dimension of the output tensor
    xPred  : torch.tensor               Feature tensor used for prediction
    outputMean : torch.tensor           Mean of output tensor
    outputStd : torch.tensor            Stadard deviation of the output tensor
    varName : str                       Variable name (used for labelling stored training data)
    stateDictDir : str                  Trained data storage directory
    VTReduced : torch.tensor            PCA left eigenvector data

    Returns
    -------
    outputDataTensor : torch.tensor     Prediction output mean tensor
    outputUpperTensor : torch.tensor    Prediction output upper confidence limit tensor
    outputLowerTensor : torch.tensor    Prediction output lower confidence limit tensor
    """
    trainingData = torch.load(stateDictDir / f"{varName}_trainingData.pt")
    xTrain, outputTrain = trainingData["xTrain"], trainingData["outputTrain"]   # Load the features and output that the model was trained with
    
    #likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=dimensionSize)
    #model = Kriging(xTrain, outputTrain, likelihood, featureSize, dimensionSize)
    
    likelihoodPrev = [gpytorch.likelihoods.GaussianLikelihood() for _ in range(dimensionSize)]  # Gpytorch works with a single output, a list of likelihoods is produced for a multi-output model
    modelPrev = [Kriging(xTrain, outputTrain[:, i], likelihoodPrev[i], featureSize) for i in range(dimensionSize)]  # Gpytorch works with a single output, a list of models is produced for a multi-output model
    modelLikelihood = [independentModel.likelihood for independentModel in modelPrev]  # Link each output point likelihood function to an output point model
    
    model = gpytorch.models.IndependentModelList(*modelPrev)                    # Create a Gaussian Process model from modelPrev (list of sub-models)
    likelihood = gpytorch.likelihoods.LikelihoodList(*modelLikelihood).train()  # Create likelihood object from list of previously created list of likelyhoods 
    
    model.load_state_dict(torch.load(stateDictDir / f"{varName}.pt")["modelState"])  # Load the model state from the trained data tensors
    
    model.eval()                                                               # Evaluate current state of the model to enable prediction
    likelihood.eval()                                                          # Evaluate current state of the likelihood to enable prediction

    with torch.no_grad(), gpytorch.settings.fast_pred_var(state=True), gpytorch.settings.fast_pred_samples(state=True):  # Disable gradient calculation for prediction purposes
        #predictions = likelihood(model(xPred))                                 # Produce the multi-output prediction including confidence region
        predictions = likelihood(*model(*[xPred for _ in range(dimensionSize)]))  # Produce the multi-output prediction including confidence region

    data = torch.column_stack([prediction.mean for prediction in predictions])  # Extract the mean value from the prediction
    confidence = [torch.column_stack([prediction.confidence_region()[i] for prediction in predictions]) for i in range(2)]  # Extract the standardised upper and lower confidence limit tensors
    dataExpanded = [unstandardiseTensor(tensor, outputMean, outputStd) for tensor in [data, *confidence]]  # Unstandardise (expand) the data and confidence limits
    
    return tuple(torch.mm(tensor, VTReduced) if name == "mgp" else tensor for tensor in dataExpanded) # Expand the output to spatial data if running modal prediction, otherwise (lumped or spatial) return the data as-is

def predictTHP(xPred: torch.tensor,
               name: str,
               model: dict[str, Any],
               outputMean: dict[str, torch.tensor],
               outputStd: dict[str, torch.tensor],
               stateDictDir: Path,
               VTReduced: dict[str, torch.tensor],
               thpHistory: list[Optional[torch.tensor]] = [],
               method: Optional[str] = None) -> list[Optional[torch.tensor]]:
    """
    Compute a Thermo-Hydraulic Performance (THP) prediction with the boundary
        conditions variables (BCV) of the specified model

    Parameters
    ----------
    name : str                          Name of current model
    model : dict                        Dictionary of current model attributes
    xPred : torch.tensor                Feature tensor used for prediction
    outputMean : dict                   Mean of output tensor
    outputStd : dict                    Stadard deviation of the output tensor
    stateDictDir : pathlib.Path         Trained data storage directory
    VTReduced : dict                    Dictionary of PCA left eigenvectors, one for each BCV

    Returns
    -------
    predictedTHPData : torch.tensor     Thermo-Hydraulic Performance evaluation
    predictedTHPUpper : torch.tensor    Thermo-Hydraulic Performance evaluation upper confidence limit
    predictedTHPLower : torch.tensor    Thermo-Hydraulic Performance evaluation lower confidence limit
    """
    if method:
        xPred = torch.from_numpy(xPred).float().unsqueeze(0)
    mlPrediction = [globals()[model["function"]](                           # Call the model's corresponding BCV prediction function, passing it the collected arguments and constructing a list of BCV
                        name=name, featureSize=xPred.shape[1], dimensionSize=model["dimensionSize"],
                        xPred=xPred, outputMean=outputMean[var], outputStd=outputStd[var],
                        varName=var, stateDictDir=stateDictDir, VTReduced=VTReduced.get(var))
                    for var in model["variables"]]
    mlPrediction = list(zip(*mlPrediction))                                 # Unwrap list [[data1, upper1, lower1], [data2, upper2, lower2], ...] -> [[data1, data2, ...], [upper1, upper2, ...], [lower1, lower2, ...]]
    mlPrediction = [(None if data[0] is None else data) for data in mlPrediction]  # Convert any lists of None to a single None for ease of processing downstream
    
    if name != "lgp" and name != "lnn" and name != "lrbf":                    # If the current model isn't lumped ...
        mlPrediction = [(None if data is None else lumpedDataCalculation(data)) for data in mlPrediction]  # Compute the Thermo-Hydraulic Performance (integral of advected heat flux and dissipation rate)
    
    if method:
        thpHistory.append(-(mlPrediction[0][0]-mlPrediction[0][1]).squeeze(0).squeeze(0))
        return thpHistory[-1]
    return [(None if data is None else data[0] - data[1]) for data in mlPrediction]  # Return the THP evaluation (subtracting dissipation rate from advected heat flux)

def predictVar(xPred: torch.tensor,
               name: str,
               model: dict[str, Any],
               outputMean: dict[str, torch.tensor],
               outputStd: dict[str, torch.tensor],
               stateDictDir: Path,
               VTReduced: dict[str, torch.tensor]) -> list[Optional[torch.tensor]]:
    """
    Compute a Thermo-Hydraulic Performance (THP) prediction with the boundary
        conditions variables (BCV) of the specified model

    Parameters
    ----------
    name : str                          Name of current model
    model : dict                        Dictionary of current model attributes
    xPred : torch.tensor                Feature tensor used for prediction
    outputMean : dict                   Mean of output tensor
    outputStd : dict                    Stadard deviation of the output tensor
    stateDictDir : pathlib.Path         Trained data storage directory
    VTReduced : dict                    Dictionary of PCA left eigenvectors, one for each BCV

    Returns
    -------
    predictedTHPData : torch.tensor     Thermo-Hydraulic Performance evaluation
    predictedTHPUpper : torch.tensor    Thermo-Hydraulic Performance evaluation upper confidence limit
    predictedTHPLower : torch.tensor    Thermo-Hydraulic Performance evaluation lower confidence limit
    """
    mlPrediction = [globals()[model["function"]](                           # Call the model's corresponding BCV prediction function, passing it the collected arguments and constructing a list of BCV
                        name=name, featureSize=xPred.shape[1], dimensionSize=model["dimensionSize"],
                        xPred=xPred, outputMean=outputMean[var], outputStd=outputStd[var],
                        varName=var, stateDictDir=stateDictDir, VTReduced=VTReduced.get(var))
                    for var in model["variables"]]
    mlPrediction = list(zip(*mlPrediction))                                 # Unwrap list [[data1, upper1, lower1], [data2, upper2, lower2], ...] -> [[data1, data2, ...], [upper1, upper2, ...], [lower1, lower2, ...]]
    mlPrediction = [(None if data[0] is None else data) for data in mlPrediction]  # Convert any lists of None to a single None for ease of processing downstream
    return [(None if data is None else data) for data in mlPrediction]  # Return the THP evaluation (subtracting dissipation rate from advected heat flux)

def maximiseTHP(model: dict,
                name: str,
                stateDictDir: Path,
                tensorDir: Path,
                epochs: int = int(10)) -> tuple[str, str, str, str]:
    """
    Find optimal features by minimising predicted THP

    Parameters
    ----------
    model : dict                        Dictionary of current model attributes
    name : str                          Name of current model
    tensorDir : pathlib.Path            Trained data storage directory
    epochs : int                        Number of optimisation iterations

    Returns
    -------
    name : str                          Name of model (used for post-training report, needs to be returned because of asynchronous training)
    Re : str                            Reynolds number (used for post-training report)
    harmonics : str                     Number of harmonics that were used for training
    optimalFeatureVals : str            Summary string of optimal feature values
    """
    
    xData = torch.load(tensorDir / "xData.pt")                                  # Load the xData dictionary of tensors
    xMean, xStd = xData["xMean"], xData["xStd"]                                 # Extract the mean and standard deviation of the unstandardised xData
    outputMean = torch.load(tensorDir / f"{model['dimensionType']}Mean.pt")     # Load the tensor of output mean values for the current model's dimension type
    outputStd = torch.load(tensorDir/ f"{model['dimensionType']}Std.pt")        # Load the tensor of output standard deviation values for the current model's dimension type
    VTReduced = torch.load(tensorDir / "VTReduced.pt")                          # Each boundary condition variable has its own set of modes, and therefore its own set of left eigenvectors, load the corresponding dictionary of tensors
    
    xPred = torch.ones(xMean.shape[0])                                          # Create an initial tensor with the same number of features as xMean and filled with zeros
    bounds = tuple([(min(xData["x"][:,i]),max(xData["x"][:,i])) for i in range(xMean.shape[0])])
    thpHistory = []
                                                                               
    xObjective = differential_evolution(predictTHP, bounds = bounds,           # line_search, least_squares, differential_evolution, nnls, basinhopping
                                        args=(name, model, outputMean, outputStd, stateDictDir, VTReduced, thpHistory, "maximize"),
                                        maxiter=int(1))
    xPredExpanded = unstandardiseTensor(torch.from_numpy(xObjective["x"]).float(), xMean, xStd)  # Convert the list of 1D feature tensors to a 2D feature tensor history, and unstandardise (expand) it
    torch.save({'xPredExpanded': xPredExpanded,
                'thpHistory': thpHistory},
                stateDictDir / "optimalFeatureHistory.pt")                      # Store the expanded optimal feature history
    
    if name != "lgp" and name != "lnn" and name != "lrbf":                        # If the current model isn't lumped ...
        xPred = torch.zeros(xMean.shape[0])
        if xPred.shape[0] == 3 or xPred.shape[0] == 5:
            xPred[-1] = xPredExpanded[-1]
        xPred0 = ((xPred - xMean) / xStd).float().unsqueeze(0)                  # Standardise tensor based on its mean and standard deviation (compute the Z-score)
        predictedBaseline = predictVar(xPred0, name, model, outputMean, outputStd, stateDictDir, VTReduced)
        predictedMax = predictVar(torch.from_numpy(xObjective["x"]).float().unsqueeze(0), name, model, outputMean, outputStd, stateDictDir, VTReduced)
        varPlot(predictedMax, predictedBaseline, stateDictDir)
    
    _, Re, _, _, _, harmonics = tensorDir.stem.split("_")                       # Extract the Reynolds number and number of harmonics from the tensorDir name
    featureLabels = ["A1", "A2", "k1", "k2", "Re"][:(5 if Re == "All" else 4):(2 if harmonics == "1" else 1)]  # Match feautre positions with their corresponding labels
    labelledFeatureValues = [f"{label}={value.item():.4f}" for label, value in zip(featureLabels, xPredExpanded)]  # Construct of a list of strings where each entry is a labelled feature with its optimal value
    with open(stateDictDir / "optimalFeatures.txt", "w") as file:               # Open a text file for writing (old content will be overwritten)
        file.write("\n".join(labelledFeatureValues))                            # Join labelled feature using newlines and store in a human-readable format
        
    return name, Re, harmonics, ", ".join(labelledFeatureValues)
    
###############################################################################

if __name__ == "__main__":
    raise RuntimeError("This script is not intended for execution. Instead, execute hammerhead.py from the parent directory.")
