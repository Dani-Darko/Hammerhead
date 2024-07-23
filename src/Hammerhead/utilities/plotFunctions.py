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

import matplotlib.pyplot as plt                                                 # Others -> Plotting tools
import matplotlib as mpl                                                        # Others -> Low-level matplotlib objects
import numpy as np                                                              # Others -> Array manipulation functions
import pickle                                                                   # Others -> Storing matplotlib figure objects
import random
import torch                                                                    # Others -> Tensor manipulation functions

###########################################################################################################################################

def predictionPlot(xv: np.ndarray,
                   yv: np.ndarray,
                   plotParams: dict[str, Union[float, bool]],
                   predictedTHPQual: list[Optional[torch.tensor]],
                   predictedTHPQuant: list[Optional[torch.tensor]],
                   stateDictDir: Path,
                   Re: Optional[int] = None) -> None:
    """
    Machine learning prediction surface vs high-fidelity scatter data plot
        of the computed Thermo-Hydraulic Performance over two fixed shape
        parameters and two varying shape parameters
    
    Parameters
    ----------
    xv : array_like                 Array of feature 1 (A1) values for plotting purposes
    yv : array_like                 Array of feature 2 (k1) values for plotting purposes
    plotParams : dict               Dictionary of plotting parameters
    predictedTHPQual : torch.tensor     List of predicted THP tensors [data, upper, lower] for qualitative 3D plot
    predictedTHPQuant : torch.tensor    List of predicted THP tensors [data, upper, lower] for quantitative 2D plot
    stateDictDir : Path             Trained tensor data storage directory
    Re : int                        Reynolds number, only passed when Re is a feature

    Returns
    -------
    None
    """
    pivotIdx = stateDictDir.parts.index("mlData")
    plotDir = Path(*stateDictDir.parts[:pivotIdx], "mlPlots", *stateDictDir.parts[pivotIdx + 1:])  # Construct plot directory (same format as state dict path, but mlData is now mlPlots)
    plotDir.mkdir(parents=True, exist_ok=True)                                  # Create the directory where plots will be stored (if it doesn't yet exist)
    tensorDir = Path(*stateDictDir.parts[:pivotIdx + 3])                        # Reconstruct outer tensorDir from state dict path
    
    lumpedPred3D = predictedTHPQual[0]                                          # Extract THP data from list of predicted qualitative data for convenience (for 3D plot)
    lumpedPred2D = predictedTHPQuant[0]                                         # Also extract THP data from list of predicted quantitative data (for 2D plot)
    lumpedLimits3D = None if predictedTHPQual[1] is None else predictedTHPQual[1:]  # Extract limits (as a single None if none are available) from list of predicted qualitative data
    lumpedLimits2D = None if predictedTHPQuant[1] is None else predictedTHPQuant[1:]  # Also extract limits from list of predicted quantitative data
    xExpanded = torch.load(tensorDir / "xData.pt")["xExpanded"]                 # Load xExpanded data from xData.pt dictionary of tensors
    xExpanded2D = torch.load(tensorDir / "xData.pt")["xExpanded"]                 # Load xExpanded data from xData.pt dictionary of tensors
    lumpedDataExpanded = torch.load(tensorDir / "lumpedDataExpanded.pt")        # Load the unstandardised lumped data dictionary of tensors
    lumpedDataExpanded2D = torch.load(tensorDir / "lumpedDataExpanded.pt")      # Load the unstandardised lumped data dictionary of tensors
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
    lumpedReal = (lumpedDataExpanded["lumpedT"] - lumpedDataExpanded["lumpedp"]) / lumpedBaseline  # Calculate the Thermo-Hydraulic Performance of the high-fidelity data for the 3D plot, normalised against the baseline case
    lumpedReal2D = (lumpedDataExpanded2D["lumpedT"] - lumpedDataExpanded2D["lumpedp"])  # Calculate the Thermo-Hydraulic Performance of the high-fidelity data for the 2D plot
    lumpedRealNorm2D = lumpedReal2D / lumpedBaseline                            # Normalise real qualitative THP against the baseline case for 2D plot
    lumpedPred3D = lumpedPred3D / lumpedBaseline                                # Normalise predicted qualitative THP against the baseline case for 3D plot
    lumpedPredNorm2D = lumpedPred2D / lumpedBaseline                            # Normalise predicted quantitative THP against the baseline case for 2D plot
    lumpedLimits3D = None if lumpedLimits3D is None else [limit / lumpedBaseline for limit in lumpedLimits3D]  # Also normalise both qualitative limits if they exist
    lumpedLimits2D = None if lumpedLimits2D is None else [limit / lumpedBaseline for limit in lumpedLimits2D]  # Also normalise both qualitative limits if they exist

    rc('font', **{'family': 'sans-serif', 'serif': ['Computer Modern Sans Serif']})  # Plot font settings to match default LaTeX style
    rc('text', usetex=plotParams["useTex"])                                     # Use TeX for rendering text if available and requested in plotParams.yaml
    
    # 3D (qualitative) prediction plot:

    X, Y = np.meshgrid(xv, yv)                                                  # Transform the x, y values into a set of coordinates for a surface plot
    Zpred = lumpedPred3D.reshape(len(xv), len(yv)).T                            # Reshape lumpedPred to match the X, Y grid shape
    Zlims = None if lumpedLimits3D is None else [limit.reshape(len(xv), len(yv)).T for limit in lumpedLimits3D]  # If limits exist, also reshape them to same shape as ZPred
    
    fig = plt.figure()                                                          # Create a figure (this will contain a single 3D axis for plotting)
    ax = fig.add_subplot(projection="3d")                                       # Add an axis with a 3D projection to the figure
    ax.scatter(*xExpanded.T, lumpedReal, c="k", label=f"$Re={Re},A_2={A2},k_2={k2}$")  # Plot the high fidelity data as black dots and set the fixed parameters label
    surf = ax.plot_surface(X, Y, Zpred, cmap=cm.jet, alpha=0.65, lw=0, antialiased=False)  # Plot the predicted lumped data as a surface with a colour gradient
    bar = fig.colorbar(surf, ax=ax, extend="min", shrink=0.75)                  # Show a colourbar for the predicted lumped data surface colour gradient values
    if Zlims is not None:                                                       # If surfaces for limits also exist ...
        for Zlim in Zlims:                                                      # ... iterate over both upper and lower limits ...
            ax.plot_surface(X, Y, Zlim, color="k", alpha=0.2, lw=0, antialiased=False)  # ... plot a surface for both limits
    
    bar.set_label(r'$\dot{Q}$', rotation=0, fontsize=14)                        # Add a label to the colourbar (Q = Thermo-Hydraulic Performance final evaluation)
    ax.set_xlabel("$\mathrm{A}_1$")                                                      # Set the x-axis label as A1
    ax.set_ylabel("$k_1$")                                                      # Set the y-axis label as k1
    ax.tick_params(axis='both', labelsize=10)                                   # Modify the tick label size
    ax.legend()                                                                 # Add a legend, containing the fixed parameter values label
    
    with open(plotDir / f'Re_{Re}_A2_{A2}_k2_{k2}.plot', "wb") as plotFile:     # Open pickle file for binary writing ...
        pickle.dump(fig, plotFile)                                              # ... store matploltib figure object
    plt.close(fig)                                                              # Close the figure and free up resources
    
    # 2D (quantitative) prediction plot:
    
    fig, ax = plt.subplots(figsize=(2.7, 2))                                    # Create a new 2D figure (default = (6.4, 4.8),  wide = (6.4, 3.0))
    points_num = min(60, len(lumpedRealNorm2D))                                 # Select a maximum amount of points to show in the quantitative plot, no more than 60
    pointsIdx = random.choices(np.arange(len(lumpedRealNorm2D)), k=points_num)  # Take a random selection of the THP array to show in the quantitative plot
    x_lin = np.arange(points_num)                                               # X-axis is a monotonically increasing sequence, one entry per predicted value
    y_mid = ((lumpedPredNorm2D + lumpedRealNorm2D) / 2)[pointsIdx, 0]           # Y-midpoints between predicted and HFM values (for errorbar plot, actual points will not be visible)
    y_lims = None if lumpedLimits2D is None else [((limit + lumpedRealNorm2D) / 2)[pointsIdx, 0] for limit in lumpedLimits2D]  # Limits for the GP to show around the predicted and HFM values
    y_err = np.abs((lumpedPredNorm2D - lumpedRealNorm2D) / 2)[pointsIdx, 0]     # Y-half-errors between predicted and HFM values from each Y-midpoint (for errorbar plot, which is supplied a single symmetric error)
    if xExpanded2D.shape[1] == 2 or xExpanded2D.shape[1] == 3:
        xlabels = [f"$\mathrm{{A}}_1={xExpanded2D[i,0]:.3f}$, $k_1={int(xExpanded2D[i,1])}$" for i in pointsIdx]
    else:
        xlabels = [f"$\mathrm{{A}}_1={xExpanded2D[i,0]:.3f}$, $k_1={int(xExpanded2D[i,2])}$" for i in pointsIdx]
    ax.errorbar(x_lin, y_mid, xerr=0, yerr=y_err, fmt="k", marker="", ls="", alpha=0.2)  # Plot errorbar first (no points, just residuals), semi-transparent, highlighting the difference between predicted and HFM values
    if y_lims is not None:
        ax.fill_between(x_lin, y_lims[0], y_lims[1], color='grey', alpha=0.5)   # Plot the GP confidence region limits when GP is available
    ax.plot(x_lin, lumpedPredNorm2D[pointsIdx], "m", label=f"$\mathrm{{{stateDictDir.parts[pivotIdx + 3].capitalize()}}}$ $\mathrm{{prediction}}$", marker=".", linewidth=0, markersize=2)  # Plot predicted values as a line plot
    ax.plot(x_lin, lumpedRealNorm2D[pointsIdx], "k", label="$\mathrm{HFM}$ $\mathrm{data}$", marker="x", linewidth=0, markersize=2)        # Plot HFM values as black crosses
    ax.set_xticks([], [])                                                       # Disable x-axis ticks, as the x-axis is meaningless
    ax.set_xlabel("$\mathrm{Cases}$", fontsize=10)                              # Set x-label
    ax.set_ylabel(r'$\dot{Q}$', fontsize=10)                                    # Set y-label
    ax.tick_params(axis='both', labelsize=6)                                    # Adjust tick label font size
    ax.legend(fontsize=6)                                                       # Finally, draw legend
    plt.grid(axis="both", alpha=0.5, linewidth=0.1)
    fig.savefig(plotDir / f'Re_{Re}_A2_{A2}_k2_{k2}_2D.pdf', bbox_inches='tight')  # Save the generated figure as a PDF
    plt.close(fig)                                                              # Close the figure and free up resources
    
    # 2D (quantitative) error plot:
    
    fig, ax = plt.subplots(figsize=(2.5, 3))                                    # Create a new 2D figure (default = (6.4, 4.8),  wide = (6.4, 3.0))
    y_err = (np.abs(lumpedPredNorm2D - lumpedRealNorm2D) / np.abs(lumpedRealNorm2D))[:, 0]  # Y relative errors between predicted and HFM values
    ax.plot(y_err, "m.", markersize=2)                                          # Plot Y relative errors
    ax.set_xticks([], [])                                                       # Disable x-axis ticks, as the x-axis is meaningless
    ax.set_yscale('log')                                                        # Set the y-axis scale as log
    ax.set_ylabel('$\mathrm{Relative}$ $\mathrm{Error}$', fontsize=10)                                         # Set y-label
    ax.tick_params(axis='both', labelsize=6)                                    # Adjust tick label font size
    plt.grid(which="both", alpha=0.5, linewidth=0.1)
    fig.savefig(plotDir / f'2D_error.pdf', bbox_inches='tight')                 # Save the generated figure as a PDF
    plt.close(fig)                                                              # Close the figure and free up resources

def varPlot(predictedMax: list[Optional[torch.tensor]],
            predictedBaseline: list[Optional[torch.tensor]],
            stateDictDir: Path) -> None:
    """
    Machine learning prediction surface vs high-fidelity scatter data plot
        of the computed Thermo-Hydraulic Performance over two fixed shape
        parameters and two varying shape parameters
    
    Parameters
    ----------
    xv : array_like                 Array of feature 1 (A1) values for plotting purposes
    yv : array_like                 Array of feature 2 (k1) values for plotting purposes
    plotParams : dict               Dictionary of plotting parameters
    predictedTHPQual : torch.tensor     List of predicted THP tensors [data, upper, lower] for qualitative 3D plot
    predictedTHPQuant : torch.tensor    List of predicted THP tensors [data, upper, lower] for quantitative 2D plot
    stateDictDir : Path             Trained tensor data storage directory
    Re : int                        Reynolds number, only passed when Re is a feature

    Returns
    -------
    None
    """
    def _format_and_save_fig(fig, ax, yLabel, figName, **kwargs):
        ax.set_xlabel("Pipe radius [m]", fontsize=14)                           # Set the x-axis label as neuron count
        ax.set_ylabel(yLabel, fontsize=14)                                      # Set the y-axis label as loss
        ax.tick_params(axis='both', labelsize=10)                               # Modify the tick label size
        fig.savefig(plotDir / figName, bbox_inches='tight', **kwargs)           # Save the generated figure as a PDF
        plt.close(fig)                                                          # Close the figure and free up resources
    
    pivotIdx = stateDictDir.parts.index("mlData")
    plotDir = Path(*stateDictDir.parts[:pivotIdx], "mlPlots", *stateDictDir.parts[pivotIdx + 1:])  # Construct plot directory (same format as state dict path, but mlData is now mlPlots)
    plotDir.mkdir(parents=True, exist_ok=True)                                  # Create the directory where plots will be stored (if it doesn't yet exist)
    tensorDir = Path(*stateDictDir.parts[:pivotIdx + 3])                        # Reconstruct outer tensorDir from state dict path
    PredMax = predictedMax[0]                                                   # Also extract THP data from list of predicted quantitative data (for 2D plot)
    
    PredBaseline = predictedBaseline[0]                                         # Also extract THP data from list of predicted quantitative data (for 2D plot)
    xExpanded = torch.load(tensorDir / "xData.pt")["xExpanded"]                 # Load xExpanded data from xData.pt dictionary of tensors
    lumpedDataExpanded = torch.load(tensorDir / "lumpedDataExpanded.pt")        # Load the unstandardised lumped data dictionary of tensors
    dataExpanded = torch.load(tensorDir / "spatialDataExpanded.pt")             # Load the unstandardised lumped data dictionary of tensors
    
    rc('font', **{'family': 'sans-serif', 'serif': ['Computer Modern Sans Serif']})  # Plot font settings to match default LaTeX style
    rc('text', usetex=True)                                                     # Use TeX for rendering text if available and requested in plotParams.yaml
    
    relevantKeys = ["inletp", "outletU", "outletT"]
    yLabels = ["Pressure [Pa]", "Velocity [m/s]", "Temperature [K]"]
    x_lin = np.arange(0,0.2,0.2/101)
    dataMax = {}
    dataBaseline = {}
    PredMax = {}
    PredBaseline = {}
    limitsBaseline = {}
    limitsMax = {}
    dataMaxIdx = (lumpedDataExpanded["lumpedT"] - lumpedDataExpanded["lumpedp"])  == max(lumpedDataExpanded["lumpedT"] - lumpedDataExpanded["lumpedp"])
    if xExpanded[0].shape[0] == 3 or xExpanded[0].shape[0] == 5:
        baselineIdx = torch.all(xExpanded[:,:-1] == torch.zeros(xExpanded[0].shape[0]-1), axis=1)  # Get the row index where A1=0 and k1=0 to use as a baseline value
    else:
        baselineIdx = torch.all(xExpanded == torch.zeros(xExpanded[0].shape), axis=1)  # Get the row index where A1=0 and k1=0 to use as a baseline value
        
    for key, i in zip(relevantKeys, [2, 3, 4]):                                 # For each unique flow variable in the array, two plots will be created
        dataBaseline[key] = dataExpanded[key][baselineIdx,:]                    # Calculate the Thermo-Hydraulic Performance of the baseline case (A1=0, k1=0)
        dataMax[key] = dataExpanded[key][dataMaxIdx[:,0],:]                     # Calculate the Thermo-Hydraulic Performance of the baseline case (A1=0, k1=0)
        PredMax[key] = predictedMax[0][i]                                       # Also extract THP data from list of predicted quantitative data (for 2D plot)
        PredBaseline[key] = predictedBaseline[0][i]                             # Also extract THP data from list of predicted quantitative data (for 2D plot)
        limitsBaseline[key] = None if predictedBaseline[1] is None else [predicted[i] for predicted in predictedBaseline[1:]]  # Also extract limits from list of predicted quantitative data
        limitsMax[key] = None if predictedMax[1] is None else [predicted[i] for predicted in predictedMax[1:]] # Also extract limits from list of predicted quantitative data
    
    for key, yLabel in zip(relevantKeys, yLabels):                              # For each unique flow variable in the array, two plots will be created
        fig, ax = plt.subplots(figsize=(4, 3))                                  # Create "wide" figure, with neuron count on x-axis and unique valSplit/Layer combinations as lines
        for data, lims, colour, linestyle, plotLabel in zip([dataBaseline, dataMax, PredBaseline, PredMax],
                                                             [None, None, limitsBaseline, limitsMax],
                                                             mpl.colormaps['jet'](np.linspace(0, 1, 4)),
                                                             ["dashed", (0, (5, 5)), "dashdot", (0, (3, 5, 1, 5))],
                                                             ["HFM baseline data", "HFM optimised data", f"{stateDictDir.parts[pivotIdx + 3]} baseline prediction", f"{stateDictDir.parts[pivotIdx + 3]} optimised prediction"]):
            if lims is not None and lims[key] is not None:
                ax.fill_between(x_lin, lims[key][0][0], lims[key][1][0], color=colour, alpha=0.5)
            ax.plot(x_lin, data[key][0,:], label=plotLabel, color=colour, linestyle=linestyle, alpha=0.8)  # Plot per-valSplit/Layer line for final loss vs neuron count
        legend = ax.legend(fontsize=10)                                         # Finally, draw legend
        _format_and_save_fig(fig, ax, yLabel, f"{stateDictDir.parts[pivotIdx + 3]}_{key}_profiles.pdf", bbox_extra_artists=[legend])

def lossPlot(plotParams: dict[str, Union[float, bool]],
             stateDictDir: Path) -> None:
    """
    Per-variable training loss plot 
    
    Parameters
    ----------
    plotParams : dict               Dictionary of plotting parameters
    stateDictDir : Path            Trained tensor data storage directory

    Returns
    -------
    None
    """
    pivotIdx = stateDictDir.parts.index("mlData")
    plotDir = Path(*stateDictDir.parts[:pivotIdx], "mlPlots", *stateDictDir.parts[pivotIdx + 1:])  # Construct plot directory (same format as state dict path, but mlData is now mlPlots)
    plotDir.mkdir(parents=True, exist_ok=True)                                  # Create the directory where plots will be stored (if it doesn't yet exist)
    
    rc('font', **{'family': 'sans-serif', 'serif': ['Computer Modern Sans Serif']})  # Plot font settings to match default LaTeX style
    rc('text', usetex=plotParams["useTex"])                                     # Use TeX for rendering text if available and requested in plotParams.yaml
        
    for checkpointFile in stateDictDir.glob("*.pt"):                            # Iterate over all checkpoint files in the model checkpoint directory
        if checkpointFile == stateDictDir / "optimalFeatureHistory.pt":
            continue
        fig, ax = plt.subplots(figsize=(2.7, 2))                                # Create new figure with single subplot
        if str(checkpointFile).split("/")[-3] == "lnn" or str(checkpointFile).split("/")[-3] == "mnn" or str(checkpointFile).split("/")[-3] == "snn":
            ax.plot(torch.load(checkpointFile)["lossValid"], c="c", label="$\mathrm{Validation}$ $\mathrm{set}$", linewidth=0.5, markersize=2)  # Load and plot validation set loss
            ax.plot(torch.load(checkpointFile)["lossTrain"], c="m", label="$\mathrm{Training}$ $\mathrm{set}$", linewidth=0.5, markersize=2)  # Load and plot training set loss
        else:
            ax.plot(torch.load(checkpointFile)["lossTrain"], c="m", label="Training set", linewidth=0.5, markersize=2)  # Load and plot training set loss
        ax.set_xlabel("$\mathrm{Epoch}$", fontsize=10)                          # Set the x-axis label as epoch
        ax.set_ylabel("$\mathrm{J}$", fontsize=10)                              # Set the y-axis label as loss
        ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=6)                                # Modify the tick label size
        plt.grid(which="both", axis="both", alpha=0.5, linewidth=0.1)
        ax.legend(fontsize=6)                                                   # Add a legend, containing the fixed parameter values label
        fig.savefig(plotDir / f"loss_{checkpointFile.stem}.pdf", bbox_inches='tight')  # Save the generated figure as a PDF
        plt.close(fig)                                                          # Close the figure and free up resources

def historyPlot(plotParams: dict[str, Union[float, bool]],
             stateDictDir: Path) -> None:
    """
    Per-variable training loss plot 
    
    Parameters
    ----------
    plotParams : dict               Dictionary of plotting parameters
    stateDictDir : Path            Trained tensor data storage directory

    Returns
    -------
    None
    """
    pivotIdx = stateDictDir.parts.index("mlData")
    plotDir = Path(*stateDictDir.parts[:pivotIdx], "mlPlots", *stateDictDir.parts[pivotIdx + 1:])  # Construct plot directory (same format as state dict path, but mlData is now mlPlots)
    plotDir.mkdir(parents=True, exist_ok=True)                                  # Create the directory where plots will be stored (if it doesn't yet exist)
    
    rc('font', **{'family': 'sans-serif', 'serif': ['Computer Modern Sans Serif']})  # Plot font settings to match default LaTeX style
    rc('text', usetex=plotParams["useTex"])                                     # Use TeX for rendering text if available and requested in plotParams.yaml
        
    for checkpointFile in stateDictDir.glob("optimalFeatureHistory.pt"):        # Iterate over all checkpoint files in the model checkpoint directory
        fig, ax = plt.subplots(figsize=(5.9, 3))                                # Create new figure with single subplot
        ax.plot(torch.load(checkpointFile)["thpHistory"], c="m", label="$\dot{Q}$ history", linewidth=0.5, markersize=2)  # Load and plot training set loss
        ax.set_xlabel("$\mathrm{Epoch}$", fontsize=10)                          # Set the x-axis label as epoch
        ax.set_ylabel("$\dot{Q}$", fontsize=10)                                 # Set the y-axis label as loss
        ax.tick_params(axis='both', labelsize=6)                                # Modify the tick label size
        plt.grid(which="both", axis="both", alpha=0.5, linewidth=0.1)
        ax.legend()                                                             # Add a legend, containing the fixed parameter values label
        fig.savefig(plotDir / f"thpHistory_{checkpointFile.stem}.pdf", bbox_inches='tight')  # Save the generated figure as a PDF
        plt.close(fig)                                                          # Close the figure and free up resources
        
def mlBenchmarkPlot(plotParams: dict[str, Union[float, bool]],
                    modelDir: Path,
                    lossTable: np.ndarray) -> None:
    """
    Per-architecture ML loss benchmark summary plot
    
    Parameters
    ----------
    plotParams : dict               Dictionary of plotting parameters
    modelDir : Path                 Trained model directory (containing multiple architectures)
    lossTable : array_like          Loss table array with columns [valSplit, layers, neurons, var, dataArr]

    Returns
    -------
    None
    """
    
    pivotIdx = modelDir.parts.index("mlData")
    plotDir = Path(*modelDir.parts[:pivotIdx], "mlPlots", *modelDir.parts[pivotIdx + 1:])  # Construct plot directory (same format as state dict path, but mlData is now mlPlots)
    plotDir.mkdir(parents=True, exist_ok=True)                                  # Create the directory where plots will be stored (if it doesn't yet exist)
    
    rc('font', **{'family': 'sans-serif', 'serif': ['Computer Modern Sans Serif']})  # Plot font settings to match default LaTeX style
    rc('text', usetex=plotParams["useTex"])                                     # Use TeX for rendering text if available and requested in plotParams.yaml
    
    if "Kernel" in lossTable[0, 1]:
        varNames, varIdxs, xVartypes, xVarwidths, lineVar = ["valSplit", "nu"], [0, 2], [float, float], [0.04, 0.04], "kernel"
    else:
        varNames, varIdxs, xVartypes, xVarwidths, lineVar = ["valSplit", "neurons"], [0, 2], [int, float], [0.9, 0.04], "layers"
    
    rows, cols = int(np.ceil(np.unique(lossTable[:, 3]).shape[0]/3)), 3 if np.unique(lossTable[:, 3]).shape[0] == 6 else 2
    figThickness = 3.5 if np.unique(lossTable[:, 3]).shape[0] == 6 else 2
    rowsIdx, colsIdx = [0,0] if np.unique(lossTable[:, 3]).shape[0] == 2 else [0,0,0,1,1,1], [0,1] if np.unique(lossTable[:, 3]).shape[0] == 2 else [0,1,2,0,1,2]
    
    for freeVarName, freeVarIdx, xVarName, xVarIdx, xVarDType, xVarWidth in zip(varNames, varIdxs, varNames[::-1], varIdxs[::-1], xVartypes, xVarwidths):
        uniqueFreeVarPre = sorted(np.unique(lossTable[:, freeVarIdx]))          # Identify and sort all unique free variables (valSplit or neurons)
        for freeVar in uniqueFreeVarPre:                                        # Identify each unique valSplit
            fig, ax = plt.subplots(rows, cols, squeeze=False, figsize=(5.9, figThickness), sharey=True, sharex=True)  # Create "wide" figure, with neuron count on x-axis and unique valSplit/Layer combinations as lines
            for var, row, col in zip(np.unique(lossTable[:, 3]),                # For each unique flow variable in the array, two plots will be created
                                     rowsIdx, colsIdx):                          
                lossTableVar = lossTable[lossTable[:, 3] == var]                # Filter table such that it now only contains entries for the current flow variable
                uniqueLines = sorted(np.unique(lossTableVar[:, 1]))             # Identify and sort all unique layer numbers
                for linesPlot, linestyle, colour in zip(uniqueLines,
                                                        ["solid", "dotted", "dashed", "dashdot"],
                                                        mpl.colormaps['viridis'](np.linspace(0, 1, len(uniqueLines)+1))):  # Identify each unique layer count with a different linestyle and a different colour
                    
                    freeValLineIdx = (lossTableVar[:, freeVarIdx] == freeVar) & (lossTableVar[:, 1] == linesPlot)  # Identify which filtered table indices correspond to this valSplit-Layer combination
                    xData = lossTableVar[freeValLineIdx, xVarIdx].astype(xVarDType)  # Extract ordered list of neurons (x-positions)
                    sortedIdx = np.argsort(xData)                               # Identify order of element indices that would return a sorted array (for plotting data in order)
                    losses = lossTableVar[freeValLineIdx, 4]                    # Extract corresponding array of loss arrays (per-sample)
                    lossMean = np.array([np.mean(loss, axis=-1) for loss in losses], dtype=float)  # Compute corresponding list of mean losses (y-positions)
                    if lineVar == "layers":
                        ax[row,col].plot(range(1,len(xData)+1), lossMean[sortedIdx], label=f"$\mathrm{{{lineVar}}} =\mathrm{{{linesPlot}}}$", marker='x', markersize=2, color=colour, linestyle=linestyle, linewidth=0.5)  # Plot per-valSplit/Layer line for final loss vs neuron count
                        violingParts = ax[row,col].violinplot(losses[sortedIdx], range(1,len(xData)+1), showmeans=True, widths=xVarWidth)  # Also plot violin plot, showing minima, maxima, means and the data distribution)
                    elif lineVar == "kernel":
                        ax[row,col].plot(xData[sortedIdx], lossMean[sortedIdx], label=f"$\mathrm{{{lineVar}}} =\mathrm{{{linesPlot}}}$", marker='x', markersize=2, color=colour, linestyle=linestyle, linewidth=0.5)  # Plot per-valSplit/Layer line for final loss vs neuron count
                        violingParts = ax[row,col].violinplot(losses[sortedIdx], xData[sortedIdx], range(1,len(xData)+1), showmeans=True, widths=xVarWidth)  # Also plot violin plot, showing minima, maxima, means and the data distribution)
                    for key, value in violingParts.items():                     # For each part of the violin plot
                        if key == "bodies":                                     # If the part is "bodies", this is a list of bodies
                            for body in value:                                  # Change the face colour of each body
                                body.set_facecolor(colour)
                        else:                                                   # Otherwise, each item is a single object
                            value.set_color(colour)                             # Change its colour to match the line plot
                            value.set_linewidth(0.5)
                if lineVar == "layers":
                    ax[row,col].set_xticks(range(1,len(xData)+1),xData[sortedIdx])
                ax[row,col].legend(fontsize=6)
                ax[row,col].tick_params(axis='both', labelsize=6)          # Modify the tick label size
                ax[row,col].set_yscale('log')
                ax[row,col].grid(which="both", axis="both", alpha=0.5, linewidth=0.1)
                ax[-1,col].set_xlabel(f"$\mathrm{{{xVarName.capitalize()}}}$", fontsize=8)  # Set the x-axis label as neuron count
                ax[row,0].set_ylabel("$\mathrm{J}$", fontsize=8)                    # Set the y-axis label as loss
            fig.savefig(plotDir / f"mlBenchmark_x{xVarName.capitalize()}.pdf", bbox_inches='tight')  # Save the generated figure as a PDF
            plt.close(fig)                                                          # Close the figure and free up resources

###############################################################################

if __name__ == "__main__":
    raise RuntimeError("This script is not intended for execution. Instead, execute hammerhead.py from the parent directory.")
