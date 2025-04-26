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
from matplotlib.ticker import FormatStrFormatter, MaxNLocator                   # Others -> Matplotlib ticker formatting
from pathlib import Path                                                        # Others -> Path manipulation
from typing import Optional, Union                                              # Others -> Python type hinting

import matplotlib.pyplot as plt                                                 # Others -> Plotting tools
import matplotlib as mpl                                                        # Others -> Low-level matplotlib objects
import numpy as np                                                              # Others -> Array manipulation functions
import pickle                                                                   # Others -> Storing matplotlib figure objects
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
    pivotIdx, plotDir = _getPlotDir(stateDictDir)                               # Create and get path to plot directory
    tensorDir = Path(*stateDictDir.parts[:pivotIdx + 3])                        # Reconstruct outer tensorDir from state dict path
    
    lumpedPred = predictedTHPQuant[0]                                           # Also extract THP data from list of predicted quantitative data (for 2D plot)
    lumpedLimits2D = None if predictedTHPQuant[1] is None else predictedTHPQuant[1:]  # Also extract limits from list of predicted quantitative data
    xExpanded = torch.load(tensorDir / "xData.pt", weights_only=True)["xExpanded"]  # Load xExpanded data from xData.pt dictionary of tensors
    lumpedDataExpanded = torch.load(tensorDir / "lumpedDataExpanded.pt", weights_only=True)  # Load the unstandardised lumped data dictionary of tensors
    harmonics = int(tensorDir.stem.split("_")[-1])                              # Deduce whether harmonics is 1 or 2 from the tensorDir name
    A2, k2 = plotParams["A2"], plotParams["k2"]                                 # Create shorthands for A2 and k2 parameter values from plotParams.yaml
    
    if Re is not None:                                                          # If Re was passed, Re is a feature
        ReIdx = xExpanded[:, -1] == Re                                          # ... find all indices where the last column of xExpanded (Re) matches the current Re
        xExpanded = xExpanded[ReIdx, :-1]                                       # ... and keep only the matchin Re rows
        for key in lumpedDataExpanded.keys():                                   # ... also iterate over all tensors in the lumpedDataExpanded directory
            lumpedDataExpanded[key] = lumpedDataExpanded[key][ReIdx]            # ... and also only keep the matching Re rows
        
    if Re is None:                                                              # If Re was not passed, Re is not a feature
        Re = int(tensorDir.stem.split("_")[1])                                  # ... we still need to deduce it for plot labelling, do that from the tensorDir name
      
    baselineIdx = torch.all(xExpanded == torch.zeros(xExpanded.shape[1]), axis=1)  # Get the row index where all A1, A2, k1, k2 = 0 to use as a baseline value
    lumpedBaseline = lumpedDataExpanded["lumpedT"][baselineIdx] - lumpedDataExpanded["lumpedp"][baselineIdx]  # Calculate the Thermo-Hydraulic Performance of the baseline case (A1=0, k1=0)    
    lumpedReal = (lumpedDataExpanded["lumpedT"] - lumpedDataExpanded["lumpedp"]) / lumpedBaseline  # Calculate the Thermo-Hydraulic Performance of the high-fidelity data for the 3D plot, normalised against the baseline case
    lumpedRealSortIdx = torch.argsort(lumpedReal, dim=0)[:, 0]                  # Get indices of sorted lumpedReal tensor (use these to sort all corresponding tensors)
    lumpedReal = lumpedReal[lumpedRealSortIdx]                                  # Sort lumpedRead tensor (all other tensors are sorted at computation time)
    lumpedPred = (lumpedPred / lumpedBaseline)[lumpedRealSortIdx]               # Normalise predicted qualitative THP against the baseline case for 3D plot
    lumpedLimits2D = None if lumpedLimits2D is None else [limit[lumpedRealSortIdx] / lumpedBaseline for limit in lumpedLimits2D]  # Also normalise both qualitative limits if they exist

    _mplPreamble(plotParams)                                                    # Execute matplotlib formatting preamble code
    
    # 2D (quantitative) prediction plot:
    
    fig, axs = plt.subplots(figsize=(5.4, 2), ncols=2, width_ratios=[1, 0.2])   # Create a new 2D figure (default = (6.4, 4.8)) with 2 cols: prediction (left) and error (right)
    points_num = min(100, len(lumpedReal))                                      # Select a maximum amount of points to show in the quantitative plot, no more than 100
    pointsIdx = np.linspace(0, len(lumpedReal)-1, points_num).astype(int)       # Take an ordered selection of the THP array to show in the quantitative plot
    x_lin = np.arange(points_num)                                               # X-axis is a monotonically increasing sequence, one entry per predicted value
    y_mid = ((lumpedPred + lumpedReal) / 2)[pointsIdx, 0]                       # Y-midpoints between predicted and HFM values (for errorbar plot, actual points will not be visible)
    y_lims = None if lumpedLimits2D is None else [((limit + lumpedReal) / 2)[pointsIdx, 0] for limit in lumpedLimits2D]  # Limits for the GP to show around the predicted and HFM values
    y_err = np.abs((lumpedPred - lumpedReal) / 2)[pointsIdx, 0]                 # Y-half-errors between predicted and HFM values from each Y-midpoint (for errorbar plot, which is supplied a single symmetric error)
    
    axs[0].errorbar(x_lin, y_mid, xerr=0, yerr=y_err, fmt="k", marker="", ls="", alpha=0.2)  # Plot errorbar first (no points, just residuals), semi-transparent, highlighting the difference between predicted and HFM values
    if y_lims is not None:
        axs[0].bar(x_lin, y_lims[1]-y_lims[0], bottom=y_lims[0], width=1, color="m", alpha=0.25, label="$\mathrm{Confidence\ region}$")  # Plot the GP confidence region limits when GP is available
    axs[0].plot(x_lin, lumpedPred[pointsIdx], "m", label=f"$\mathrm{{{stateDictDir.parts[pivotIdx + 3]}}}$ $\mathrm{{prediction}}$", marker=".", linewidth=0, markersize=2)  # Plot predicted values as a line plot
    axs[0].plot(x_lin, lumpedReal[pointsIdx], "k", label="$\mathrm{HFM}$ $\mathrm{data}$", marker="x", linewidth=0, markersize=2)  # Plot HFM values as black crosses
    axs[0].set_xticks([], [])                                                   # Disable x-axis ticks, as the x-axis is meaningless
    axs[0].set_xlabel(("$\mathrm{Subset\ of\ cases}$" if points_num < len(lumpedReal) else "$\mathrm{Cases}$")+"$\mathrm{\ (in\ ascending\ HFM\ }\overline{\dot{Q}})$", fontsize=10)  # Set x-label (depending on number of points)
    axs[0].set_ylabel(r'$\overline{\dot{Q}}$', fontsize=10)                     # Set y-label
    axs[0].tick_params(axis='both', labelsize=6)                                # Adjust tick label font size
    axs[0].grid(axis="y", alpha=0.5, linewidth=0.1)                             # Draw y-axis gridlines
    axs[0].legend(fontsize=6, loc="lower right")                                # Finally, draw legend

    y_err = (np.abs(lumpedPred - lumpedReal) / np.abs(lumpedReal))[:, 0]        # Y relative errors between predicted and HFM values
    axs[1].plot(y_err, "m.", markersize=1, alpha=0.25)                          # Plot Y relative errors
    axs[1].set_xticks([], [])                                                   # Disable x-axis ticks, as the x-axis is meaningless
    axs[1].set_yscale('log')                                                    # Set the y-axis scale as log
    axs[1].set_ylabel('$\mathrm{Relative\ error}$', fontsize=10)                # Set y-label
    axs[1].tick_params(axis='both', labelsize=6)                                # Adjust tick label font size
    axs[1].yaxis.tick_right()                                                   # Move y-axis ticks to the right 
    axs[1].yaxis.set_label_position("right")                                    # Move y-axis label to the right
    axs[1].grid(axis="y", alpha=0.5, linewidth=0.1)                             # Draw y-axis gridlines
    
    plt.tight_layout()
    fig.savefig(plotDir / f'Re_{Re}_A2_{A2}_k2_{k2}_2D.pdf', bbox_inches='tight')  # Save the generated figure as a PDF
    plt.close(fig)                                                              # Close the figure and free up resources
    
    # 3D (qualitative) prediction plot:
    
    if harmonics == 2:                                                          # If harmonics=2, dimensions are too big for plotting, they will need to be shortened to match the form of harmonics=1
        plotIdx = (xExpanded[:, 1] == A2) & (xExpanded[:, 3] == k2)             # ... get a boolean tensor used to select row indices where columns 1 and 3 match the specified A2 and k2 values in plotParams.yaml
        xExpanded = torch.index_select(xExpanded[plotIdx], 1, torch.tensor([0, 2]))  # ... keep only the rows where A2 and k2 match the specified parameters, and then remove their respective (now redundant) columns (keep 0->A1 and 2->k1 only)
        for key in lumpedDataExpanded.keys():                                   # ... also iterate over all tensors in the lumpedDataExpanded directory
            lumpedDataExpanded[key] = lumpedDataExpanded[key][plotIdx]          # ... and select only the relevant rows (but keeping columns unmodified, as they are only a single column)
            
    lumpedLimits3D = None if predictedTHPQual[1] is None else [limit / lumpedBaseline for limit in predictedTHPQual[1:]]  # Extract normalised limits (only for GP, no limits otherwise)
    lumpedPred3D = predictedTHPQual[0] / lumpedBaseline                         # Extract THP data from list of predicted qualitative data for convenience (for 3D plot)            
    lumpedReal3D = (lumpedDataExpanded["lumpedT"] - lumpedDataExpanded["lumpedp"]) / lumpedBaseline  # Calculate the Thermo-Hydraulic Performance of the high-fidelity data for the 3D plot, normalised against the baseline case

    X, Y = np.meshgrid(xv, yv)                                                  # Transform the x, y values into a set of coordinates for a surface plot
    Zpred = lumpedPred3D.reshape(len(xv), len(yv)).T                            # Reshape lumpedPred to match the X, Y grid shape
    Zlims = None if lumpedLimits3D is None else [limit.reshape(len(xv), len(yv)).T for limit in lumpedLimits3D]  # If limits exist, also reshape them to same shape as ZPred
    
    
    
    fig = plt.figure()                                                          # Create a figure (this will contain a single 3D axis for plotting)
    ax = fig.add_subplot(projection="3d")                                       # Add an axis with a 3D projection to the figure
    ax.scatter(*xExpanded.T, lumpedReal3D, c="k", label=f"$\mathrm{{Re}}={Re},A_2={A2},k_2={k2}$")  # Plot the high fidelity data as black dots and set the fixed parameters label
    surf = ax.plot_surface(X, Y, Zpred, cmap=cm.jet, alpha=0.65, lw=0, antialiased=False)  # Plot the predicted lumped data as a surface with a colour gradient
    bar = fig.colorbar(surf, ax=ax, extend="min", shrink=0.75, format=FormatStrFormatter('%.2f'))  # Show a colourbar for the predicted lumped data surface colour gradient values
    if Zlims is not None:                                                       # If surfaces for limits also exist ...
        for Zlim in Zlims:                                                      # ... iterate over both upper and lower limits ...
            ax.plot_surface(X, Y, Zlim, color="k", alpha=0.2, lw=0, antialiased=False)  # ... plot a surface for both limits
    
    bar.set_label(r'$\overline{\dot{Q}}$', rotation=0, labelpad=10)             # Add a label to the colourbar (Q = Thermo-Hydraulic Performance final evaluation)
    ax.set_xlabel("$A_1$")                                                      # Set the x-axis label as A1
    ax.set_ylabel("$k_1$")                                                      # Set the y-axis label as k1
    ax.tick_params(axis='both', labelsize=10)                                   # Modify the tick label size
    ax.legend()                                                                 # Add a legend, containing the fixed parameter values label
    plt.tight_layout()                                                          # Apply tight layout
    with open(plotDir / f'Re_{Re}_A2_{A2}_k2_{k2}.plot', "wb") as plotFile:     # Open pickle file for binary writing ...
        pickle.dump(fig, plotFile)                                              # ... store matplotlib figure object
    plt.close(fig)                                                              # Close the figure and free up resources

def varPlot(plotParams: dict[str, Union[float, bool]],
            profileDictFileHFM: Path,
            profileDictFileSM: Path) -> None:
    """
    Flow variable profile plot 
    
    Parameters
    ----------
    plotParams : dict               Dictionary of plotting parameters
    profileDictFileHFM : Path       Path to outer HDM variable profile dictionary
    profileDictFileSM : Path        Path to inner SM variable profile dictionary

    Returns
    -------
    None
    """
    pivotIdx, plotDir = _getPlotDir(profileDictFileSM.parent)                   # Create and get path to plot directory
    _mplPreamble(plotParams)                                                    # Execute matplotlib formatting preamble code
    
    dataHFM, dataSM = torch.load(profileDictFileHFM, weights_only=True), torch.load(profileDictFileSM, weights_only=True)  # Load data from HFM and SM tensors    
    fig, axs = plt.subplots(figsize=(5.4, 6), ncols=2, nrows=3, sharex="col", sharey="row")  # Create figure and 3x2 subplots
    for row, var in enumerate(["$p\ \mathrm{[kPa]}$", "$v_x\ \mathrm{[ms^{-1}]}$", "$T\ \mathrm{[K]}$"]):  # Iterate over all rows, and their corresponding y-axis labels
        axs[row, 0].set_ylabel(var)                                             # For each row, set y-axis label in the leftmost column
        for col, data in enumerate([dataHFM, dataSM]):                          # Also iterate over all columns, and the respective datasets
            axs[row, col].tick_params(axis='both', labelsize=6)                 # Modify the tick label size for each ax
            axs[row, col].grid(which="both", axis="both", alpha=0.5, linewidth=0.1)  # Draw minor and major gridlines on both axis for each ax
            for key, lbl, c in zip(["profileBaseline",     "profileOptimised"],  # Iterate over data keys, and the corresponding line labels and colours
                                   ["$\mathrm{Baseline}$", "$\mathrm{Maximum\ THP}$"],
                                   ["teal",                "crimson"]):
                axs[row, col].plot(data[key][row], c=c, ls="-", lw=1, label=lbl)  # Plot two lines for each ax

    axs[0, 0].set_title("$\mathrm{HFM}$", fontsize=10)                          # Set top left axis title
    axs[0, 1].set_title("$\mathrm{SM}$", fontsize=10)                           # Set top right axis title
    axs[0, 1].legend(fontsize=6)                                                # Legend only in top right axis
    fig.savefig(plotDir / "varPlot.pdf", bbox_inches='tight')                   # Save figure as PDF
    plt.close(fig)                                                              # Close figure, free up memory


def lossPlot(plotParams: dict[str, Union[float, bool]],
             stateDictDir: Path) -> None:
    """
    Per-variable training loss plot 
    
    Parameters
    ----------
    plotParams : dict               Dictionary of plotting parameters
    stateDictDir : Path             Trained tensor data storage directory

    Returns
    -------
    None
    """
    pivotIdx, plotDir = _getPlotDir(stateDictDir)                               # Create and get path to plot directory
    _mplPreamble(plotParams)                                                    # Execute matplotlib formatting preamble code
    
    model = plotDir.parts[pivotIdx+3]                                           # Get model name from stateDictDir path
    vars = ["lumpedp", "lumpedT"] if model[0] == "L" else ["inletp", "outletp", "inletT", "outletT", "inletU", "outletU"]  # Establish model variables based on model prefix ("L" = lumped)
    plotData = [[torch.load(stateDictDir / f"{var}.pt", weights_only=False)[entry]  # Load loss data (either training or validation) for the current variable
                 for entry in (["lossTrain"] if model[1:] == "GP" else ["lossTrain", "lossValid"])]  # Iterate over all types of loss, based on model suffix
                for var in vars]                                                # Iterate over all model variables, loading all loss data in stateDictDir
    
    fig, axs = plt.subplots(figsize=(5.4, 2 if model[0] == "L" else 6), ncols=2, nrows=int(len(vars)/2), sharex=True, sharey=True, squeeze=False)  # Create figure, subplots and size based on modelPrefix
    for ax, axData in zip(axs.flatten(), plotData):                             # Iterate over each axis (one axis per variable, ordered)
        ax.tick_params(axis='both', labelsize=6)                                # Modify the tick label size
        ax.set_yscale('log')                                                    # Set axis y-scale to logarithmic
        ax.grid(which="both", axis="both", alpha=0.5, linewidth=0.1)            # Draw minor and major gridlines on both axis
        for entry, color, lineData in zip(["$\mathrm{Training\ set}$", "$\mathrm{Validation\ set}$"], ["m", "c"], axData):  # Iterate over all available types of loss
            ax.plot(lineData, c=color, lw=1, label=entry)                       # ... plotting one line per loss type in each axis
    axs[0, 0].set_title("$f_\mathrm{f}$" if model[0] == "L" else "$\mathrm{Inlet}$", fontsize=10)  # Set top left axis title based on model prefix
    axs[0, 1].set_title("$f_\mathrm{h}$" if model[0] == "L" else "$\mathrm{Outlet}$", fontsize=10)  # Set top right axis title based on model prefix
    for i, label in enumerate([""] if model[0] == "L" else ["$(p)$", "$(T)$", "$(v_{x})$"]):  # Iterate over number of axis rows
        axs[i, 0].set_ylabel("$\mathrm{J}\ $" + label, fontsize=10)             # ... for each row, set left axis y-label
    for i in range(2):                                                          # For each column ...
        axs[-1, i].set_xlabel("$\mathrm{Epoch}$", fontsize=10)                  # ... write the x-label on the bottom row
    axs[0, 1].legend(fontsize=6)                                                # Legend only in top right axis
    plt.tight_layout()                                                          # Apply tight layout
    fig.savefig(plotDir / "lossPlot.pdf", bbox_inches='tight')                  # Save figure as PDF
    plt.close(fig)                                                              # Close figure, free up memory

def historyPlot(plotParams: dict[str, Union[float, bool]],
                stateDictDir: Path) -> None:
    """
    THP optimisation search history plot 
    
    Parameters
    ----------
    plotParams : dict               Dictionary of plotting parameters
    stateDictDir : Path             Trained tensor data storage directory

    Returns
    -------
    None
    """
    pivotIdx, plotDir = _getPlotDir(stateDictDir)                               # Create and get path to plot directory
    _mplPreamble(plotParams)                                                    # Execute matplotlib formatting preamble code
        
    fig, ax = plt.subplots(figsize=(5.4, 2))                                    # Create new figure with single subplot
    try:
        thpHistory = torch.load(stateDictDir / "optimalSearchResults.pt", weights_only=False)["thpHistory"]  # Load THP history from stored tensor
        xPredExpanded = torch.load(stateDictDir / "optimalSearchResults.pt", weights_only=False)["xPredExpanded"]  # Load maximised THP input features

        match xPredExpanded.shape[0]:                                           # Formatting string depends on number of maximised features
            case 3:                                                             # Case: A1, k1, Re
                annotation = "$A_1 = {:.3f}$\n$k_1 = {:.0f}$\n$\mathrm{{Re}} = {:.0f}$".format(*xPredExpanded)
            case 4:                                                             # Case: A1, A2, k1, k2
                annotation = r"\begin{{align*}}A_1 &= {:.3f}\\[-6pt] A_2 &= {:.3f}\\[-6pt] k_1 &= {:.0f}\\[-6pt] k_2 &= {:.0f}\end{{align*}}".format(*xPredExpanded)
            case 5:                                                             # Case: A1, A2, k1, k2, Re
                annotation = "$A_1 = {:.3f}$\n$A_2 = {:.3f}$\n$\ k_1 = {:.0f}$\n$\ k_2 = {:.0f}$\n$\mathrm{{Re}} = {:.0f}$".format(*xPredExpanded)

        ax.plot(np.abs(thpHistory), c="m", label="$\dot{Q}$ optimisation history", linewidth=1, markersize=2)  # Plot THP differential evolution history (inverted, as we are maximising)
        ax.annotate(annotation,                                                 # Insert annotation of maximised THP parameters
                    xy=(len(thpHistory)-1, np.abs(thpHistory[-1])), xycoords='data',  # Coordinates of where arrow points to (in data units)
                    xytext=(0.662, 0.6), textcoords='axes fraction',            # Coordinates of text (in axis fractions)
                    verticalalignment='center', horizontalalignment='left',     # Text alignment
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", fc="w"),  # Arrow formatting
                    bbox=dict(boxstyle="round", fc="w"))                        # Bounding box formatting
        ax.set_xlabel("$\mathrm{Iterations}$", fontsize=10)                     # Set the x-axis label as number of iterations
        ax.set_ylabel("$\dot{Q}$", fontsize=10)                                 # Set the y-axis label as loss
        ax.tick_params(axis='both', labelsize=6)                                # Modify the tick label size
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))                   # Ensure x-axis only has ticks at integer locations
        ax.grid(axis="y", alpha=0.5, linewidth=0.1)                             # Draw gridlines on y-axis
        ax.legend(loc="lower right")                                            # Add a legend, containing the fixed parameter values label
        plt.tight_layout()                                                      # Apply tight layout
        fig.savefig(plotDir / "thpHistory_optimalSearchResults.pdf", bbox_inches='tight')  # Save the generated figure as a PDF
    except FileNotFoundError:
        print(f"Cannot plot THP optimisation search history for {'/'.join(stateDictDir.parts[2:])}, ensure optimal search was performed!")
    finally:
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
    lossTable : array_like          Loss table array with columns [valSplit, NNLayers, NNNeurons, var, dataArr]

    Returns
    -------
    None
    """
    pivotIdx, plotDir = _getPlotDir(modelDir)                                   # Create and get path to plot directory
    _mplPreamble(plotParams)                                                    # Execute matplotlib formatting preamble code
    
    def _updateViolinColours(violingParts, colour):
        """
        Private function for updating colour of all violin plot elements
        """
        for key, value in violingParts.items():                                 # For each part of the violin plot
            if key == "bodies":                                                 # If the part is "bodies", this is a list of bodies
                for body in value:                                              # Change the face colour of each body
                    body.set_facecolor(colour)
            else:                                                               # Otherwise, each item is a single object
                value.set_color(colour)                                         # Change its colour to match the line plot
                value.set_linewidth(0.5)
    
    modelPrefix, modelSuffix = modelDir.parts[-1][0], modelDir.parts[-1][1:]    # Infer model prefix and suffix from model path
    vars = ["lumpedp", "lumpedT"] if modelPrefix == "L" else ["inletp", "outletp", "inletT", "outletT", "inletU", "outletU"]  # Establish model variables based on model prefix ("L" = lumped)
    validMarkers = ["X", ".", "^", "s", "p", "P", "*", "o"]                     # Define list of valid markers (and enforce maximum number of lines per axis)
    
    varLabel = ["valSplit", "neurons" if modelSuffix == "NN" else "kernel"]     # List of all variables that will be processed (either on the x-axis, or kept constant)
    varIndices = [0, 2 if modelSuffix == "NN" else 1]                           # Indices of corresponding variables in the loss table
    for freeVarLabel, freeVarIdx, xVarLabel, xVarIdx in zip(varLabel, varIndices, varLabel[::-1], varIndices[::-1]):  # Iterate over both variables, keeping one "free", and the other on the x-axis
        for freeVar in sorted(np.unique(lossTable[:, freeVarIdx])):             # Iterate over all sorted unique values of the "free variable" in the loss table
            fig, axs = plt.subplots(figsize=(5.4, 2 if modelPrefix == "L" else 6), ncols=2, nrows=int(len(vars)/2), sharex="col", sharey="row", squeeze=False)  # Create figure, subplots and size based on modelPrefix
            for ax, var in zip(axs.flatten(), vars):                            # Iterate over all axis and corresponding flow variables (one flow variable per axis)
                lossTableVar = lossTable[lossTable[:, 3 if modelSuffix == "NN" else 2] == var]  # Filter table such that it now only contains entries for the current flow variable
                if modelSuffix == "NN":                                         # If model is NN, there will be one line per layer ...
                    uniqueLines = np.unique(lossTableVar[:, 1])                 # ... find all unique layers, and order them numerically
                    uniqueLines = uniqueLines[np.argsort(uniqueLines.astype(int))]
                else:                                                           # For all other models, there will only be a single line per plot
                    uniqueLines = [""]                                          # ... empty placeholder so line count is correct, its content doesn't matter
                    
                for line, marker, colour in zip(uniqueLines, validMarkers, mpl.colormaps['viridis'](np.linspace(0, 1, len(uniqueLines)+1))):  # For each line to plot in a single axis
                    plotMask = lossTableVar[:, freeVarIdx] == freeVar           # Boolean mask of everything to be plotted in this axis (everything for current "free variable")
                    if modelSuffix == "NN":                                     # As NN plots are per-line, also mark only data for the current line to be plotted
                        plotMask = plotMask & (lossTableVar[:, 1] == line)
                        
                    xData = lossTableVar[plotMask, xVarIdx]                     # Extract ordered list of data for the x-axis
                    sortedIdx = np.argsort(xData.astype(float if modelSuffix == "NN" else str))  # Identify order of element indices that would return a sorted array (for plotting data in order, numerically or alphabetically)
                    losses = lossTableVar[plotMask, -1]                         # Extract corresponding array of loss arrays
                    lossMean = np.array([np.mean(loss, axis=-1) for loss in losses], dtype=float)  # Compute array of mean losses per x-point (y-positions)
                    # Plot mean losses at each x-point, where x-values are just ordered integers (will be labelled separately), ensuring even separation of x-points
                    ax.plot(range(1, len(xData)+1), lossMean[sortedIdx], label=f"$\mathrm{{layers}} = \mathrm{{{line}}}$" if modelSuffix == "NN" else "$\mathrm{Kernel}$", marker=marker, markersize=2, color=colour, linestyle="-", linewidth=1)
                    # For each x-point and line, also plot a violin plot that illustrates the distribution of losses (assuming multiple samples per point)
                    violinParts = ax.violinplot(losses[sortedIdx], range(1,len(xData)+1), showmeans=True, widths=0.9)  # Plot violin plot, showing minima, maxima, means and the data distribution)
                    _updateViolinColours(violinParts, colour)                   # Update colour of violin plot so it matches the current colourmap
                    ax.set_xticks(range(1, len(xData)+1), [tick.lstrip("0") for tick in xData[sortedIdx]])  # Set x-ticks based on previously sorted x data labels (remove leading zeros)
                    ax.tick_params(axis='both', labelsize=6)                    # Modify tick label size
                    ax.set_yscale('log')                                        # As this plot shows loss, y-axis should be logarithmic
                    ax.grid(which="both", axis="both", alpha=0.5, linewidth=0.1)  # Draw gridlines on both axes
            
            # Based on model prefix, set y-labels (row labels) in the first column
            for row, lbl in enumerate([""] if modelPrefix == "L" else ["(p)", "(T)", "(v_{{x}})"]):
                axs[row, 0].set_ylabel(f"$\mathrm{{J}}{{{lbl}}}$", fontsize=8)
            # Based on model prefix, also draw column labels (upper row titles) and x-axis label (bottom row only)
            for col, lbl in enumerate(["$f_\mathrm{f}$", "$f_\mathrm{h}$"] if modelPrefix == "L" else ["$\mathrm{Inlet}$", "$\mathrm{Outlet}$"]):
                axs[0, col].set_title(lbl, fontsize=10) 
                axs[-1, col].set_xlabel("$\mathrm{Validation\ split}$" if xVarLabel == "valSplit" else ("$\mathrm{Neurons}$" if xVarLabel == "neurons" else "$\mathrm{Kernel}$"))
                
            axs[0, -1].legend(fontsize=6)                                       # Legend in top right axis only
            plt.tight_layout()                                                  # Apply tight layout
            fig.savefig(plotDir / f"mlBenchmark_{freeVarLabel}={freeVar}_x={xVarLabel}.pdf", bbox_inches='tight')  # Save the generated figure as a PDF
            plt.close(fig)                                                      # Close the figure and free up resources
            
def _getPlotDir(stateDictDir: Path) -> Path:
    """
    Convenience function for getting+creating plotDir from stateDictDir
    """
    pivotIdx = stateDictDir.parts.index("mlData")                               # Get path parts index of mlData, the "pivot point"
    plotDir = Path(*stateDictDir.parts[:pivotIdx], "mlPlots", *stateDictDir.parts[pivotIdx + 1:])  # Construct plot directory (same format as state dict path, but mlData is now mlPlots)
    plotDir.mkdir(parents=True, exist_ok=True)                                  # Create the directory where plots will be stored (if it doesn't yet exist)
    return pivotIdx, plotDir

def _mplPreamble(plotParams: dict[str, Union[float, bool]]) -> None:
    """
    Convenience function that executes matplotlib formatting preamble code
    """
    rc('font', **{'family': 'sans-serif', 'serif': ['Computer Modern Sans Serif']})  # Plot font settings to match default LaTeX style
    rc('text', usetex=plotParams["useTex"])                                     # Use TeX for rendering text if available and requested in plotParams.yaml
    rc('text.latex', preamble=r'\usepackage{amsmath}')                          # Use amsmath LaTeX package for equation formatting

###############################################################################

if __name__ == "__main__":
    raise RuntimeError("This script is not intended for execution. Instead, execute hammerhead.py from the parent directory.")
