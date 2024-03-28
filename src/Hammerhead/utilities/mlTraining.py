##############################################################################################
#                                                                                            #
#       HAMMERHEAD - Harmonic MOO Model Expedient for the Reduction                          #
#                    Hydraulic-loss/Heat-transfer Enhancement Asset Design                   #
#       Author: Daniela Minerva Segura Galeana                                               #
#       Supervised by: Antonio J. Gil                                                        #
#                                                                                            #
##############################################################################################
#                                                                                            #
#       Script containing Machine Learning (ML) training functions                           #
#                                                                                            #
##############################################################################################

# IMPORTS: HAMMERHEAD files ###############################################################################################################

from utilities.mlClasses import (Network,                                       # Utilities -> Neural Network class
                                 Kriging)                                       # Utilities -> Gaussian Process class

# IMPORTS: Others #########################################################################################################################

from pathlib import Path                                                        # Others -> Path manipulation tools
from scipy.interpolate import RBFInterpolator                                   # Others -> Radial Basis Function Interpolator

import gpytorch                                                                 # Others -> Gaussian Process tools/settings
import torch                                                                    # Others -> Neural Network tools

###########################################################################################################################################

def NN(name: str,
       featureSize: int,
       dimensionSize: int,
       layers: int,
       neurons: int,
       xTrain: torch.tensor,
       outputTrain: torch.tensor,
       xValid: torch.tensor,
       outputValid: torch.tensor,
       varName: str,
       tensorDir: Path,
       maxEpochs: int = int(1e4),
       lossTarget: float = 1e-9,
       **kwargs) -> tuple[str, str, str, str]:
    """
    Neural Network (NN) training process
    
    Parameters
    ----------
    name : str                          Name of model (used for labelling stored training data)
    featureSize : int                   Number of features in input tensor
    dimensionSize : int                 Dimension of the output tensor
    layers: int                         Number of NN hidden layers
    neurons: int                        Number of NN neurons per layer
    xTrain : torch.tensor               Feature tensor used for training
    outputTrain : torch.tensor          Output data tensor used for training
    xValid : torch.tensor               Feature tensor used for validation (overfitting)
    outputValid : torch.tensor          Output data tensor used for validation (overfitting)
    varName : str                       Variable name (used for labelling stored training data)
    tensorDir : str                     Trained data output (storage) directory
    maxEpochs : int                     Maximum number of training iterations (if lossTarget is not reached)
    lossTarget : float                  Loss target to terminate training

    Returns
    -------
    name : str                          Name of model (used for post-training report, needs to be returned because of asynchronous training)
    Re : str                            Reynolds number (used for post-training report)
    harmonics : str                     Number of harmonics that were used for training
    varName : str                       Variable name (used for post-training report)
    """
    network = Network(featureSize, dimensionSize, layers, neurons)              # Create instance of neural network class
    optimizer = torch.optim.RMSprop(                                            # Object to hold and update hyperparameter state of the model throughout training
        network.parameters(), momentum=0.75, lr=1e-4, weight_decay=1e-5)        # These parameters seem to work for this particular problem
    lossFunc = torch.nn.MSELoss()                                               # Loss function utilising mean square error (MSE)
    lossTrainList, lossValidList = [], []                                       # History of all computed losses during training    
    
    for epoch in range(maxEpochs):                                              # Start the training, for a maximum of maxEpoch iterations
        yPred = network(xTrain)                                                 # Produce a prediction from training features
        lTrain = lossFunc(yPred, outputTrain)                                   # Compare prediction with training data through MSE
        lossTrainList.append(lTrain.item())                                     # Store the loss from training error
        v = network(xValid)                                                     # Produce a prediction from validation features
        lVal = lossFunc(v, outputValid)                                         # Compare prediction with validation data through MSE
        lossValidList.append(lVal.item())                                       # Store the loss from validation error
        optimizer.zero_grad()                                                   # Reset the gradient for the backpropagation
        lTrain.backward()                                                       # Compute the gradient on the training prediction
        optimizer.step()                                                        # Hyperparameter update from the computed gradient
        if lossTrainList[-1] < lossTarget:                                      # If loss target has been reached ...
            break                                                               # ... stop the training process

    outputDir = tensorDir / name                                                # Model state will be stored in a tensorDir subdirectory labelled by the model name
    outputDir.mkdir(parents=True, exist_ok=True)                                # Create this directory if it does not yet exist
    torch.save({'epoch': epoch,
                'modelState': network.state_dict(),
                'optimizerState': optimizer.state_dict(),
                'lossTrain': lossTrainList,
                'lossValid': lossValidList},
                outputDir / f"{name}_{varName}.pt")                             # Store the obtained network checkpoint
    
    _, Re, _, _, _, harmonics = tensorDir.stem.split("_")
    return name, Re, harmonics, varName

def RBF(name: str,
        xTrain: torch.tensor,
        outputTrain: torch.tensor,
        varName: str,
        tensorDir: Path,
        **kwargs) -> tuple[str, str, str, str]:
    """
    Radial Basis Function (RBF) interpolator training process
    
    Parameters
    ----------
    name : str                          Name of model (used for labelling stored training data)
    xTrain : torch.tensor               Feature tensor used for training
    outputTrain : torch.tensor          Output data tensor used for training
    varName : str                       Variable name (used for labelling stored training data)
    tensorDir : str                     Trained data output (storage) directory
    
    Returns
    -------
    name : str                          Name of model (used for post-training report, needs to be returned because of asynchronous training)
    Re : str                            Reynolds number (used for post-training report)
    harmonics : str                     Number of harmonics that were used for training
    varName : str                       Variable name (used for post-training report)
    """
    rbfi = RBFInterpolator(xTrain, outputTrain, kernel='inverse_multiquadric', epsilon=3)  # Apply Radial Basis Function Interpolator (RBFI) from scipy onto the feature and output training tensors
    outputDir = tensorDir / name                                                # Model state will be stored in a tensorDir subdirectory labelled by the model name
    outputDir.mkdir(parents=True, exist_ok=True)                                # Create this directory if it does not yet exist
    torch.save(rbfi, outputDir / f"{name}_{varName}.pt")                        # Store the trained RBFI object (internally uses pickle)
    
    _, Re, _, _, _, harmonics = tensorDir.stem.split("_")
    return name, Re, harmonics, varName
    
def GP(name: str,
       featureSize: int,
       dimensionSize: int,
       xTrain: torch.tensor,
       outputTrain: torch.tensor,
       varName: str,
       tensorDir: Path,
       epochs: int = 2,  # 40
       **kwargs) -> tuple[str, str, str, str]:
    """
    Gaussian Process (GP) training process
    
    Parameters
    ----------
    name : str                          Name of model (used for labelling stored training data)
    featureSize : int                   Number of features in input tensor
    dimensionSize : int                 Dimension of the output tensor
    xTrain : torch.tensor               Feature tensor used for training
    outputTrain : torch.tensor          Output data tensor used for training
    varName : str                       Variable name (used for labelling stored training data)
    tensorDir : str                     Trained data output (storage) directory
    epochs : int                        Number of training iterations

    Returns
    -------
    name : str                          Name of model (used for post-training report, needs to be returned because of asynchronous training)
    Re : str                            Reynolds number (used for post-training report)
    harmonics : str                     Number of harmonics that were used for training
    varName : str                       Variable name (used for post-training report)
    epochs : int                        Number of training iterations performed
    finalLossTrain : float              Training loss evaluated at the final iteration
    """
    likelihoodPrev = [gpytorch.likelihoods.GaussianLikelihood() for _ in range(dimensionSize)]  # Gpytorch trains for a single output, a list of likelihoods is produced for a multi-output model
    modelPrev = [Kriging(xTrain, outputTrain[:, i], likelihoodPrev[i], featureSize) for i in range(dimensionSize)]  # Gpytorch trains for a single output, a list of models is produced for a multi-output model
    modelLikelihood = [independentModel.likelihood for independentModel in modelPrev]  # Link each output point likelihood function to an output point model
    
    model = gpytorch.models.IndependentModelList(*modelPrev)                    # Create a Gaussian Process model from modelPrev (list of sub-models)
    likelihood = gpytorch.likelihoods.LikelihoodList(*modelLikelihood).train()  # Create likelihood object from list of previously created list of likelyhoods 
    model.train()                                                               # Initialize the training process for the model
    likelihood.train()                                                          # Initialize the training process for the likelihood
    
    # optimizer = torch.optim.RMSprop(
    #     model.parameters(), momentum=0.9, lr=1e-10)
    optimizer = torch.optim.RMSprop(                                            # Object to hold and update hyperparameter state of the model throughout training
        model.parameters(), momentum=0.9, lr=1e-1)                              # These parameters seem to work for this particular problem

    lossFunc = gpytorch.mlls.SumMarginalLogLikelihood(likelihood, model)        # Loss function utilising marginal log-likelihood (MLL)
    lossList = []                                                               # History of all computed losses during training
        
    with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False) and gpytorch.settings.lazily_evaluate_kernels(state=False):
        for _ in range(epochs):                                                 # Start the training
            optimizer.zero_grad()                                               # Reset the gradient for the backpropagation
            output = model(*model.train_inputs)                                 # Produce a prediction from training features
            loss = -lossFunc(output, model.train_targets)                       # Compare prediction with training data through MLL
            loss.backward()                                                     # Compute the gradient on the training prediction
            optimizer.step()                                                    # Hyperparameter update from the computed gradient
            lossList.append(loss)                                               # Store the loss from training error
    
    outputDir = tensorDir / name                                                # Model state will be stored in a tensorDir subdirectory labelled by the model name
    outputDir.mkdir(parents=True, exist_ok=True)                                # Create this directory if it does not yet exist
    torch.save({'epoch': epochs,
                'modelState': model.state_dict(),
                'optimizerState': optimizer.state_dict(),
                'lossTrain': lossList},
               outputDir / f"{name}_{varName}.pt")                              # Store the obtained model checkpoint
    
    torch.save({'xTrain': xTrain,
                'outputTrain': outputTrain},
               outputDir / f"{name}_{varName}_trainingData.pt")                 # Store training tensors for evaluation
    
    _, Re, _, _, _, harmonics = tensorDir.stem.split("_")
    return name, Re, harmonics, varName
    
###############################################################################

if __name__ == "__main__":
    raise RuntimeError("This script is not intended for execution. Instead, execute hammehead.py from the parent directory.")
