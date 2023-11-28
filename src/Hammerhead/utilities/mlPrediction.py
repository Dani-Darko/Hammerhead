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

# IMPORTS: Others #########################################################################################################################

from pathlib import Path                                                        # Others -> Path manipulation tools
from typing import Any                                                          # Others -> Python type hinting

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
       tensorDir: Path,
       VTReduced: torch.tensor = None,
       **kwargs) -> torch.tensor:
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
    tensorDir : str                     Trained data storage directory
    VTReduced : torch.tensor            PCA left eigenvector data

    Returns
    -------
    outputTensor : torch.tensor         Prediction output tensor
    """
    checkpoint = torch.load(tensorDir / name / f"{name}_{varName}.pt")          # Load the checkpoint of the model
    
    network = Network(featureSize, dimensionSize)                               # Create instance of neural network class
    network.load_state_dict(checkpoint['modelState'])                           # Load the latest state of the model
    network.eval()                                                              # Evaluate current state of the network to enable prediction
    
    with torch.no_grad():                                                       # Disable gradient calculation for prediction purposes
        data = network(xPred)                                                   # Produce the prediction
    dataExpanded = unstandardiseTensor(data, outputMean, outputStd)             # Unstandardise (expand) the data
    
    if name == "modal":                                                         # If running modal model prediction
        return torch.mm(dataExpanded, VTReduced)                                # ... expand the output to spatial data
    return dataExpanded                                                         # ... otherwise return the data as-is (lumped or spatial)

def RBF(name: str,
        xPred: torch.tensor,
        outputMean: torch.tensor,
        outputStd: torch.tensor,
        varName: str,
        tensorDir: Path,
        VTReduced: torch.tensor,
        **kwargs) ->  torch.tensor:
    """
    Radial Basis Function (RBF) interpolator prediction process
    
    Parameters
    ----------
    name : str                          Name of model (used for loading training data)
    xPred  : torch.tensor               Feature tensor used for prediction
    outputMean : torch.tensor           Mean of output tensor
    outputStd : torch.tensor            Stadard deviation of the output tensor
    varName : str                       Variable name (used for labelling stored training data)
    tensorDir : str                     Trained data storage directory
    VTReduced : torch.tensor            PCA left eigenvector data

    Returns
    -------
    outputTensor : torch.tensor         Prediction output tensor
    """
    rbfi = torch.load(tensorDir / name / f"{name}_{varName}.pickle")            # Load the stored trained RBFI object (interally uses pickle)
    
    data = torch.from_numpy(rbfi(xPred.detach())).float()                       # Produce the prediction
    dataExpanded = unstandardiseTensor(data, outputMean, outputStd)             # Unstandardise (expand) the data
    return torch.mm(dataExpanded, VTReduced)                                    # Expand the output to spatial data
    
def GP(name: str,
       featureSize: int,
       dimensionSize: int,
       xPred: torch.tensor,
       outputMean: torch.tensor,
       outputStd: torch.tensor,
       varName: str,
       tensorDir: Path,
       VTReduced: torch.tensor = None,
       **kwargs) -> torch.tensor:
    """
    Gaussian Process (GP) prediction process
    
    Parameters
    ----------
    name : str                          Name of model (used for loading training data)
    featureSize : int                   Number of features in input tensor
    dimensionSize : int                 Dimension of the output tensor
    xPred  : torch.tensor               Feature tensor used for prediction
    outputMean : torch.tensor           Mean of output tensor
    outputStd : torch.tensor            Stadard deviation of the output tensor
    varName : str                       Variable name (used for labelling stored training data)
    tensorDir : str                     Trained data storage directory
    VTReduced : torch.tensor            PCA left eigenvector data

    Returns
    -------
    outputTensor : torch.tensor     Prediction output tensor
    """
    trainingData = torch.load(tensorDir / name / f"{name}_{varName}_trainingData.pt")
    xTrain, outputTrain = trainingData["xTrain"], trainingData["outputTrain"]   # Load the features and output that the model was trained with
    
    likelihoodPrev = [gpytorch.likelihoods.GaussianLikelihood() for _ in range(dimensionSize)]  # Gpytorch works with a single output, a list of likelihoods is produced for a multi-output model
    modelPrev = [Kriging(xTrain, outputTrain[:, i], likelihoodPrev[i], featureSize) for i in range(dimensionSize)]  # Gpytorch works with a single output, a list of models is produced for a multi-output model
    modelLikelihood = [independentModel.likelihood for independentModel in modelPrev]  # Link each output point likelihood function to an output point model
    
    model = gpytorch.models.IndependentModelList(*modelPrev)                    # Create a Gaussian Process model from modelPrev (list of sub-models)
    likelihood = gpytorch.likelihoods.LikelihoodList(*modelLikelihood).train()  # Create likelihood object from list of previously created list of likelyhoods 
    
    model.load_state_dict(torch.load(tensorDir / name / f"{name}_{varName}.pt")["modelState"])  # Load the model state from the trained data tensors
    
    model.eval()                                                               # Evaluate current state of the model to enable prediction
    likelihood.eval()                                                          # Evaluate current state of the likelihood to enable prediction

    with torch.no_grad(), gpytorch.settings.fast_pred_var():                   # Disable gradient calculation for prediction purposes
        predictions = likelihood(*model(*[xPred for _ in range(dimensionSize)]))  # Produce the multi-output prediction including confidence region

    data = torch.column_stack([prediction.mean for prediction in predictions])  # Extract the mean value from the prediction
    dataExpanded = unstandardiseTensor(data, outputMean, outputStd)             # Unstandardise (expand) the data

    # stanlow = torch.transpose(torch.stack([prediction.confidence_region()[0]
    #                                                for submodel, prediction in zip(model.models, predictions)]),1,0)
    # stanup = torch.transpose(torch.stack([prediction.confidence_region()[1]
    #                                                for submodel, prediction in zip(model.models, predictions)]),1,0)
    # up = stanup * datastd + datamean
    # low = stanlow * datastd + datamean
    # upExpanded = np.dot(up.detach().numpy(), VTReduced)
    # lowExpanded = np.dot(low.detach().numpy(), VTReduced)
    
    return torch.mm(dataExpanded, VTReduced)                                    # Expand the output to spatial data

def predictTHP(model: dict[str, Any],
               name: str,
               xPred: torch.tensor,
               outputMean: dict[str, torch.tensor],
               outputStd: dict[str, torch.tensor],
               tensorDir: Path,
               VTReduced: dict[str, torch.tensor]) -> torch.tensor:
    """
    Compute a Thermo-Hydraulic Performance (THP) prediction with the boundary
        conditions variables (BCV) of the specified model

    Parameters
    ----------
    model : dict                        Dictionary of current model attributes
    name : str                          Name of current model
    xPred : torch.tensor                Feature tensor used for prediction
    outputMean : dict                   Mean of output tensor
    outputStd : dict                    Stadard deviation of the output tensor
    tensorDir : pathlib.Path            Trained data storage directory
    VTReduced : dict                    Dictionary of PCA left eigenvectors, one for each BCV

    Returns
    -------
    lumped : torch.tensor               Thermo-Hydraulic Performance evaluation
    """
    mlPrediction = [globals()[model["function"]](                           # Call the model's corresponding BCV prediction function, passing it the collected arguments and constructing a list of BCV
                        name=name, featureSize=xPred.shape[1], dimensionSize=model["dimensionSize"],
                        xPred=xPred, outputMean=outputMean[var], outputStd=outputStd[var],
                        varName=var, tensorDir=tensorDir, VTReduced=VTReduced.get(var))
                    for var in model["variables"]]
    if name != "lumped":                                                    # If the current model isn't lumped ...
            mlPrediction = lumpedDataCalculation(mlPrediction)              # Compute the Thermo-Hydraulic Performance (integral of advected heat flux and dissipation rate)
    return mlPrediction[0] - mlPrediction[1]                                # Return the THP evaluation (subtracting dissipation rate from advected heat flux)

def maximiseTHP(model: dict,
                name: str,
                tensorDir: Path,
                epochs: int = 10) -> tuple[str, str, str, str]:
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
    dataMean = torch.load(tensorDir / f"{model['dimensionType']}Mean.pt")       # Load the tensor of output mean values for the current model's dimension type
    dataStd = torch.load(tensorDir/ f"{model['dimensionType']}Std.pt")          # Load the tensor of output standard deviation values for the current model's dimension type
    VTReduced = torch.load(tensorDir / "VTReduced.pt")                          # Each boundary condition variable has its own set of modes, and therefore its own set of left eigenvectors, load the corresponding dictionary of tensors
    
    xPred = torch.zeros(1, xMean.shape[0])                                      # Create an initial tensor with the same number of features as xMean and filled with zeros
    xPred.requires_grad_()                                                      # Enable computation of gradients during backpropagation for this tensor
    xHistory = [xPred[0].detach().clone()]                                      # List that will contain history of all xPred tensors during optimisation
    
    optimizer = torch.optim.Adam([xPred], lr=0.1)                               # Create an instance of the omptimizer, using initial xPred as the starting tensor
    for _ in range(epochs):                                                     # Iterate over the specified number of epochs (optimisation steps)
        THP = -predictTHP(model, name, xPred, dataMean, dataStd, tensorDir, VTReduced)  # Calculate the THP of the current xPred (negative for maximisation, we want lowest THP)
        THP.requires_grad_()                                                    # Enable computation of graidents during backpropagation for this new tensor
        optimizer.zero_grad()                                                   # Reset the gradient of the optimiser
        THP.backward()                                                          # Perform backpropagation
        optimizer.step()                                                        # Step the optimiser object forward
        xHistory.append(xPred[0].detach().clone())                              # Detach the current xPred tensor state and store it in the history
        
    xPredExpanded = unstandardiseTensor(torch.stack(xHistory, dim=0), xMean, xStd)  # Convert the list of 1D feature tensors to a 2D feature tensor history, and unstandardise (expand) it
    torch.save(xPredExpanded, tensorDir / name / "optimalFeatureHistory.pt")    # Store the expanded optimal feature history
    
    _, Re, _, _, _, harmonics = tensorDir.stem.split("_")                       # Extract the Reynolds number and number of harmonics from the tensorDir name
    featureLabels = ["A1", "A2", "k1", "k2", "Re"][:(5 if Re == "All" else 4):(2 if harmonics == "1" else 1)]  # Match feautre positions with their corresponding labels
    labelledFeatureValues = [f"{label}={value.item():.4f}" for label, value in zip(featureLabels, xPredExpanded[-1])]  # Construct of a list of strings where each entry is a labelled feature with its optimal value
    with open(tensorDir / name / "optimalFeatures.txt", "w") as file:           # Open a text file for writing (old content will be overwritten)
        file.write("\n".join(labelledFeatureValues))                            # Join labelled feature using newlines and store in a human-readable format
        
    return name, Re, harmonics, ", ".join(labelledFeatureValues)
    
###############################################################################

if __name__ == "__main__":
    raise RuntimeError("This script is not intended for execution. Instead, execute hammehead.py from the parent directory.")