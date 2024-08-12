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
from torch.linalg import svd
import numpy as np

###########################################################################################################################################

def NN(featureSize: int,
       dimensionSize: int,
       layers: int,
       neurons: int,
       xTrain: torch.tensor,
       outputTrain: torch.tensor,
       xValid: torch.tensor,
       outputValid: torch.tensor,
       varName: str,
       outputDir: Path,
       maxEpochs: int = int(9e4),
       lossTarget: float = 1e-9,
       **kwargs) -> None:
    """
    Neural Network (NN) training process
    
    Parameters
    ----------
    featureSize : int                   Number of features in input tensor
    dimensionSize : int                 Dimension of the output tensor
    layers: int                         Number of NN hidden layers
    neurons: int                        Number of NN neurons per layer
    xTrain : torch.tensor               Feature tensor used for training
    outputTrain : torch.tensor          Output data tensor used for training
    xValid : torch.tensor               Feature tensor used for validation (overfitting)
    outputValid : torch.tensor          Output data tensor used for validation (overfitting)
    varName : str                       Variable name (used for labelling stored training data)
    outputDir : str                     Trained data output (storage) directory
    maxEpochs : int                     Maximum number of training iterations (if lossTarget is not reached)
    lossTarget : float                  Loss target to terminate training

    Returns
    -------
    None
    """
    network = Network(featureSize, dimensionSize, neurons, layers)              # Create instance of neural network class
    optimizer = torch.optim.AdamW(                                              # Object to hold and update hyperparameter state of the model throughout training
        network.parameters(), lr=1e-2)                                          # These parameters seem to work for this particular problem
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

    outputDir.mkdir(parents=True, exist_ok=True)                                # Create this directory if it does not yet exist
    torch.save({'epoch': epoch,
                'modelState': network.state_dict(),
                'optimizerState': optimizer.state_dict(),
                'lossTrain': lossTrainList,
                'lossValid': lossValidList},
                outputDir / f"{varName}.pt")                                    # Store the obtained network checkpoint

def RBF(xTrain: torch.tensor,
        outputTrain: torch.tensor,
        varName: str,
        outputDir: Path,
        **kwargs) -> None:
    """
    Radial Basis Function (RBF) interpolator training process
    
    Parameters
    ----------
    xTrain : torch.tensor               Feature tensor used for training
    outputTrain : torch.tensor          Output data tensor used for training
    varName : str                       Variable name (used for labelling stored training data)
    outputDir : str                     Trained data output (storage) directory
    
    Returns
    -------
    None
    """
    rbfi = RBFInterpolator(xTrain, outputTrain, kernel='inverse_multiquadric', epsilon=3)  # Apply Radial Basis Function Interpolator (RBFI) from scipy onto the feature and output training tensors
    outputDir.mkdir(parents=True, exist_ok=True)                                # Create this directory if it does not yet exist
    torch.save(rbfi, outputDir / f"{varName}.pt")                               # Store the trained RBFI object (internally uses pickle)
    
def GP(featureSize: int,
       dimensionSize: int,
       kernel: str,
       maternNu: float,
       xTrain: torch.tensor,
       outputTrain: torch.tensor,
       xValid: torch.tensor,
       outputValid: torch.tensor,
       varName: str,
       outputDir: Path,
       epochs: int = int(7e3),  # 40
       lossTarget: float = 1e-4,
       **kwargs) -> None:
    """
    Gaussian Process (GP) training process
    
    Parameters
    ----------
    featureSize : int                   Number of features in input tensor
    dimensionSize : int                 Dimension of the output tensor
    kernel: str                         Kernel type of the covariance matrix
    maternNu: float                     Smoother for the MaternKernel
    xTrain : torch.tensor               Feature tensor used for training
    outputTrain : torch.tensor          Output data tensor used for training
    xValid : torch.tensor               Feature tensor used for validation (overfitting)
    outputValid : torch.tensor          Output data tensor used for validation (overfitting)
    varName : str                       Variable name (used for labelling stored training data)
    outputDir : str                     Trained data output (storage) directory
    epochs : int                        Number of training iterations
    lossTarget : float                  Loss target to terminate training

    Returns
    -------
    None
    """
    
    likelihoodPrev = [gpytorch.likelihoods.GaussianLikelihood(noise=torch.tensor(1e-7)) for _ in range(dimensionSize)]  # Gpytorch trains for a single output, a list of likelihoods is produced for a multi-output model
    modelPrev = [Kriging(xTrain, outputTrain[:, i], likelihoodPrev[i], kernel, maternNu, featureSize, dimensionSize) for i in range(dimensionSize)]  # Gpytorch trains for a single output, a list of models is produced for a multi-output model
    modelLikelihood = [independentModel.likelihood for independentModel in modelPrev]  # Link each output point likelihood function to an output point model
    
    model = gpytorch.models.IndependentModelList(*modelPrev)                    # Create a Gaussian Process model from modelPrev (list of sub-models)
    likelihood = gpytorch.likelihoods.LikelihoodList(*modelLikelihood).train()  # Create likelihood object from list of previously created list of likelyhoods 
    
    model.train()                                                               # Initialize the training process for the model
    likelihood.train()                                                          # Initialize the training process for the likelihood
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1)

    mll = gpytorch.mlls.SumMarginalLogLikelihood(likelihood, model)               # Loss function utilising marginal log-likelihood (MLL)
    lossTrainList = []                                                            # History of all computed losses during training
    
    with gpytorch.settings.lazily_evaluate_kernels(state=False), gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False):
        for i in range(epochs):                                                 # Start the training
            optimizer.zero_grad()                                               # Reset the gradient for the backpropagation
            distTrain = likelihood(*model(*[xTrain for _ in range(dimensionSize)]))
            mseTrain = [gpytorch.metrics.mean_squared_error(distTrain[i], outputTrain[:,i], squared=True) for i in range(dimensionSize)]
            output = model(*model.train_inputs)                                 # Produce a prediction from training features
            loss = -mll(output, model.train_targets)                            # Compare prediction with training data through MLL
            loss.backward()                                                     # Compute the gradient on the training prediction
            optimizer.step()                                                    # Hyperparameter update from the computed gradient
            lossTrainList.append(max(mseTrain).item())                               # Store the loss from training error
            if model.likelihood.likelihoods[0].noise.item() < lossTarget:                                  # If loss target has been reached ...
                break                                                           # ... stop the training process
    
    outputDir.mkdir(parents=True, exist_ok=True)                                # Create this directory if it does not yet exist
    torch.save({'epoch': epochs,
                'modelState': model.state_dict(),
                'optimizerState': optimizer.state_dict(),
                'xTrain': xTrain,
                'outputTrain': outputTrain,
                'lossTrain': lossTrainList},
               outputDir / f"{varName}.pt")                                     # Store the obtained model checkpoint
    
###############################################################################

if __name__ == "__main__":
    raise RuntimeError("This script is not intended for execution. Instead, execute hammerhead.py from the parent directory.")
