##############################################################################################
#                                                                                            #
#       HAMMERHEAD - Harmonic MOO Model Expedient for the Reduction                          #
#                    Hydraulic-loss/Heat-transfer Enhancement Asset Design                   #
#       Author: Daniela Minerva Segura Galeana                                               #
#       Supervised by: Antonio J. Gil                                                        #
#                                                                                            #
##############################################################################################
#                                                                                            #
#       Script containing miscellaneous plotting routines                                    #
#                                                                                            #
##############################################################################################

# IMPORTS: HAMMERHEAD files ###############################################################################################################

# IMPORTS: Others #########################################################################################################################

from matplotlib import cm, rc                                                   # Others -> Matplotlib formatting tools
from pathlib import Path                                                        # Others -> Path manipulation
from typing import Optional, Union                                              # Others -> Python type hinting

import torch                                                                    # Others -> Tensor manipulation functions
import numpy as np                                                              # Others -> Array manipulation functions
import matplotlib.pyplot as plt                                                 # Others -> Plotting tools

###########################################################################################################################################

def lumpedPlot(xv: np.ndarray,
               yv: np.ndarray,
               plotDir: Path,
               plotParams: dict[str, Union[float, bool]],
               name: str,
               lumpedPred: torch.tensor,
               tensorDir: Path,
               Re: Optional[int] = None) -> tuple[str, int, int]:
    """
    Machine learning prediction surface vs high-fidelity scatter data plot
        of the computed Thermo-Hydraulic Performance over two fixed shape
        parameters and two varying shape parameters
    
    Parameters
    ----------
    xv : array_like                 Array of feature 1 (A1) values for plotting purposes
    yv : array_like                 Array of feature 2 (k1) values for plotting purposes
    plotDir : Path                  Plot output storage directory
    plotParams : dict               Dictionary of plotting parameters
    name : str                      Name of model (used for labelling stored plots)
    lumpedPred : torch.tensor       Tensor of predicted THP
    tensorDir : Path                Trained tensor data storage directory
    Re : int                        Reynolds number, only passed when Re is a feature

    Returns
    -------
    name : str                      Name of model (used for formatting return output)
    Re : int                        Reynolds number (used for formatting return output)
    harmonics : int                 Number of harmonics
    """
    plotDir = plotDir / tensorDir.stem / name                                   # Construct plot directory of the form ./caseDatabase/{domain} / Re_{Re}_modes_{modes}_harmonics_{harmonics} / {name}
    plotDir.mkdir(parents=True, exist_ok=True)                                  # Create the directory where plots will be stored (if it doesn't yet exist)
    
    xExpanded = torch.load(tensorDir / "xData.pt")["xExpanded"]                 # Load xExpanded data from xData.pt dictionary of tensors
    lumpedDataExpanded = torch.load(tensorDir / "lumpedDataExpanded.pt")        # Load the unstandardised lumped data dictionary of tensors
    harmonics = int(tensorDir.stem.split("_")[-1])                              # Deduce whether harmonics is 1 or 2 from the tensorDir name
    A2, k2 = plotParams["A2"], plotParams["k2"]                                 # Create shorthands for A2 and k2 parameter values from plotParams.yaml
    
    if Re is not None:                                                          # If Re was passed, Re is a feature
        ReIdx = xExpanded[:, -1] == Re                                          # ... find all indices where the last column of xExpanded (Re) matches the current Re
        xExpanded = xExpanded[ReIdx, :-1]                                       # ... and keep only the matchin Re rows
        for key in lumpedDataExpanded.keys():                                   # ... also iterate over all tensors in the lumpedDataExpanded directory
            lumpedDataExpanded[key] = lumpedDataExpanded[key][ReIdx]            # ... and also only keep the matching Re rows
        
    if Re is None:                                                              # If Re was not passed, Re is not a feature
        Re = int(tensorDir.stem.split("_")[1])                                  # ... we still need to deduce it for plot labelling, do that from the tensorDir name
        
    if harmonics == 2:                                                          # If harmonics=2, dimensions are too big for plotting, they will need to be shortened to match the form of harmonics=1
        plotIdx = (xExpanded[:, 1] == A2) & (xExpanded[:, 3] == k2)             # ... get a boolean tensor used to select row indices where columns 1 and 3 match the specified A2 and k2 values in plotParams.yaml
        xExpanded = torch.index_select(xExpanded[plotIdx], 1, torch.tensor([0, 2]))  # ... keep only the rows where A2 and k2 match the specified parameters, and then remove their respective (now redundant) columns (keep 0->A1 and 2->k1 only)
        for key in lumpedDataExpanded.keys():                                   # ... also iterate over all tensors in the lumpedDataExpanded directory
            lumpedDataExpanded[key] = lumpedDataExpanded[key][plotIdx]          # ... and select only the relevant rows (but keeping columns unmodified, as they are only a single column)
      
    baselineIdx = torch.all(xExpanded == torch.zeros(2), axis=1)                # Get the row index where A1=0 and k1=0 to use as a baseline value
    lumpedBaseline = lumpedDataExpanded["lumpedT"][baselineIdx] - lumpedDataExpanded["lumpedp"][baselineIdx]  # Calculate the Thermo-Hydraulic Performance of the baseline case (A1=0, k1=0)    
    lumpedReal = (lumpedDataExpanded["lumpedT"] - lumpedDataExpanded["lumpedp"]) / lumpedBaseline  # Calculate the Thermo-Hydraulic Performance of the high-fidelity data, normalised against the baseline case
    lumpedPred = lumpedPred / lumpedBaseline                                    # Normalise predicted THP against the baseline case
    
    rc('font', **{'family': 'sans-serif', 'serif': ['Computer Modern Sans Serif']})  # Plot font settings to match default LaTeX style
    rc('text', usetex=plotParams["useTex"])                                     # Use TeX for rendering text if available and requested in plotParams.yaml

    X, Y = np.meshgrid(xv, yv)                                                  # Transform the x, y values into a set of coordinates for a surface plot
    Z = lumpedPred.reshape(len(xv), len(yv))                                    # Reshape lumpedPred to match the X, Y grid shape
    
    fig = plt.figure()                                                          # Create a figure (this will contain a single 3D axis for plotting)
    ax = fig.add_subplot(projection="3d")                                       # Add an axis with a 3D projection to the figure
    ax.scatter(*xExpanded.T, lumpedReal, c="k", label=f"$Re={Re},A_2={A2},k_2={k2}$")  # Plot the high fidelity data as black dots and set the fixed parameters label
    surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, alpha=0.65, lw=0, antialiased=False)  # Plot the predicted lumped data as a surface with a colour gradient
    bar = fig.colorbar(surf, ax=ax, extend="min", shrink=0.75)                  # Show a colourbar for the predicted lumped data surface colour gradient values
    
    bar.set_label(r'$Q$', rotation=0, fontsize=14)                              # Add a label to the colourbar (Q = Thermo-Hydraulic Performance final evaluation)
    ax.set_xlabel("$A_1$")                                                      # Set the x-axis label as A1
    ax.set_ylabel("$k_1$")                                                      # Set the y-axis label as k1
    ax.tick_params(axis='both', labelsize=10)                                   # Modify the tick label size
    ax.legend()                                                                 # Add a legend, containing the fixed parameter values label
    
    for azim in np.arange(20, 360, 45):                                         # Iterate over a range of azimuthal view angles, spanning a full circle in 45 degree increments, starting at 20 degrees
        ax.view_init(30, azim)                                                  # Set the 3D-axis viewing angle (vertical / elevation, horizontal / azimuthal)
        fig.savefig(plotDir / f'Re_{Re}_A2_{A2}_k2_{k2}_azim_{azim}.pdf', bbox_inches='tight')  # Save the generated figure as a PDF
    plt.close(fig)                                                              # Close the figure and free up resources
    return name, Re, harmonics
    
###############################################################################

if __name__ == "__main__":
    raise RuntimeError("This script is not intended for execution. Instead, execute hammehead.py from the parent directory.")