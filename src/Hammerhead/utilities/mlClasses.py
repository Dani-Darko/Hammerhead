##############################################################################################
#                                                                                            #
#       HAMMERHEAD - Harmonic MOO Model Expedient for the Reduction                          #
#                    Hydraulic-loss/Heat-transfer Enhancement Asset Design                   #
#       Author: Daniela Minerva Segura Galeana                                               #
#       Supervised by: Antonio J. Gil                                                        #
#                                                                                            #
##############################################################################################
#                                                                                            #
#       Script containing all Machine Learning classes instantiated throughout Hammerhead    #
#                                                                                            #
##############################################################################################

# IMPORTS: HAMMERHEAD files ###############################################################################################################

# IMPORTS: Others #########################################################################################################################

import torch                                                                    # Others -> Neural Network functions
import gpytorch                                                                 # Others -> Gaussian Process tools

###########################################################################################################################################

class Network(torch.nn.Module):
    def __init__(self, featureSize: int, dimensionSize: int, neurons: int = 32, hiddenLayers: int = 1) -> None:
        """
        Neural network class used by modal, spatial and lumped models
        
        Parameters
        ----------
        featureSize : int                   Number of features in input tensor
        dimensionSize : int                 Dimension of the output tensor
        neurons : int                       Number of input and output features for all hidden layers
        hiddenLayers : int                  Number of layers excluding the input and output layer

        Returns
        -------
        None
        """
        super().__init__()
        inputs = [featureSize] + [neurons for _ in range(hiddenLayers + 1)]     # List of input features for each layer (except for the input layer which uses featureSize
        outputs = [neurons for _ in range(hiddenLayers + 1)] + [dimensionSize]  # List of output features for each layer (except for the output layer which uses dimensionSize)
        
        linearSteps = [torch.nn.Linear(i, o) for i, o in zip(inputs, outputs)]  # List of all linear modules constructed using the known number of input and output features per layer
        sigmoidStep = torch.nn.Sigmoid()                                        # Intermedate function executed between each linear layer, taking input from each linear layer and passing its output to the next linear layer
        
        layers = sum([[linearStep, sigmoidStep] for linearStep in linearSteps], [])[:-1]  # Construct a list of all modules that will be called in order, with a sigmoidStep between each linearStep
        self.model = torch.nn.Sequential(*layers)                               # Construct a sequential module container, where the output of each module is passed as an input for the next module
            
    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Computation performed for every NN call (once per epoch)
        
        Parameters
        ----------
        x : torch.tensor                    Input tensor
        
        Returns:
        --------
        out : torch.tensor                  Output tensor
        """
        return self.model(x)

class Kriging(gpytorch.models.ExactGP):
    def __init__(self, xTrain: torch.tensor, yTrain: torch.tensor, likelihood: gpytorch.likelihoods.GaussianLikelihood, featureSize: int) -> None:
        """
        Gaussian Process class used by the kriging model only
        
        Parameters
        ----------
        xTrain : torch.tensor               Features tensor
        yTrain : torch.tensor               Modes tensor
        likelihood : gpytorch.likelihoods   Log Marginal Likelihood for regression
        featureSize : int                   Amount of features

        Returns
        -------
        None
        """
        super().__init__(xTrain, yTrain, likelihood)
        self.modalMean = gpytorch.means.ZeroMean()                              # Function that computes the distribution mean for each step (in this case, always zero)
        self.modalCovar = gpytorch.kernels.ScaleKernel(                         # Function that will compute a covariance matrix for each step ... 
            gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=featureSize)     # ... using a 'kernel' based on a Scale of the sum of a Matern and a Polynomial kernel
            + gpytorch.kernels.PolynomialKernel(power=1, num_dimentions=featureSize)) # ... nu values can be one of [0.5, 1.5, 2.5], higher is smoother
        
    def forward(self, x: torch.tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Compute a multivariate normal random variable, from a distribution
            based on the computed model mean (zero) and covariance (based on
            the kernel) at this step; called once per epoch
            
        Parameters
        ----------
        x : torch.tensor                    Input tensor
        
        Returns:
        --------
        out : MultivariateNormal            Multivariate distribution at current state
        """
        return gpytorch.distributions.MultivariateNormal(self.modalMean(x), self.modalCovar(x))
    
###############################################################################

if __name__ == "__main__":
    raise RuntimeError("This script is not intended for execution. Instead, execute hammehead.py from the parent directory.")
