##############################################################################################
#                                                                                            #
#       HAMMERHEAD - Harmonic MOO Model Expedient for the Reduction                          #
#                    Hydraulic-loss/Heat-transfer Enhancement Asset Design                   #
#       Author: Daniela Minerva Segura Galeana                                               #
#       Supervised by: Antonio J. Gil                                                        #
#                                                                                            #
##############################################################################################
#                                                                                            #
#       Script containing miscellaneous data processing routines                             #
#                                                                                            #
##############################################################################################

# IMPORTS: HAMMERHEAD files ###############################################################################################################

# IMPORTS: Others #########################################################################################################################

from multiprocessing import Pool                                                # Others -> Processing multiple tensor groups in parallel
from pathlib import Path                                                        # Others -> Path manipulation
from time import sleep                                                          # Others -> Pause execution between subprocess status polls
from torch.linalg import svd                                                    # Others -> Tensor Singular Value Decomposition
from tqdm import tqdm                                                           # Others -> Progress bar
from typing import Any, Callable, Optional                                      # Others -> Python type hinting
from yaml import dump, safe_load                                                # Others -> YAML file savin and loading

import torch                                                                    # Others -> Tensor manipulation functions
import numpy as np                                                              # Others -> Array manipulation functions

###########################################################################################################################################

def loadyaml(filename: str, override: dict[str, Any] = {}) -> dict[str, Any]:
    """
    Load parameter dictionary from requsted yaml files in "./resources"
    
    Parameters
    ----------
    filename : str                      Name of YAML file containing requested parameters
    override : dict[str, Any]           Dictionary of key-value pairs which supersede the loaded entries

    Returns
    -------
    paramDict : dict                    Dictionary containing the requested parameters
    """
    with open(f"./resources/{filename}.yaml") as yamlFile:
        paramDict = safe_load(yamlFile)
        for key, value in override.items():
            paramDict[key] = value
        return paramDict
    
def normaliseTensor(expandedTensor: torch.tensor) -> list[torch.tensor, torch.tensor, torch.tensor]:
    """
    Normalise a tensor to [0, 1]
    
    Parameters
    ----------
    expandedTensor : torch.tensor       Expanded tensor to be normalised [snapshots, featureSize]

    Returns
    -------
    normalisedTensor: torch.tensor      Normalised tensor [snapshots, featureSize]
    minValue: torch.tensor              Minimum value across dim 0 [1, featureSize]
    maxValue: torch.tensor              Maximum value across dim 0 [1, featureSize]
    """
    minValue, maxValue = torch.aminmax(expandedTensor, dim=0)                   # Compute minimum and maximum of all features (dim 0)
    normalisedTensor = ((expandedTensor - minValue) / (maxValue - minValue))    # Normalise tensor to range [0, 1]
    normalisedTensor[torch.isnan(normalisedTensor)] = 0.0                       # Replace any NaN (occurs if minValue == maxValue) with zeros
    return [normalisedTensor, minValue, maxValue]                               # Return normalised tensor, as well as the mean and standard deviation of the original expanded tensor

def denormaliseTensor(normalisedTensor: torch.tensor, minValue: torch.tensor, maxValue: torch.tensor) -> torch.tensor:
    """
    Revert normalised tensor to original data given original min/max values
    
    Parameters
    ----------
    normalisedTensor: torch.tensor      Normalised tensor [snapshots, featureSize]
    minValue: torch.tensor              Minimum value across dim 0 [1, featureSize]
    maxValue: torch.tensor              Maximum value across dim 0 [1, featureSize]

    Returns
    -------
    expandedTensor : torch.tensor       Original denormalised tensor [snapshots, featureSize]
    """
    return normalisedTensor * (maxValue - minValue) + minValue

def genericPoolManager(taskFuncs: list[Callable],
                       taskArgsList: Optional[list[Any]],
                       taskKwargsList: Optional[list[dict[str, Any]]],
                       nProc: int,
                       progressMessage: str,
                       returnMessage: Optional[str],
                       taskUpdateSecs: int = 1) -> None:
    """
    Generic mulitprocessing pool manager with progress bar tracking and
        dynamic process output formatting and printing
    
    Parameters
    ----------
    taskFuncs : list                    Ordered list of functions that will be called in parallel
    taskArgsList : list                 Optional: Ordered list of lists, where each list represents the arguments passed to the corresponding function
    taskKwargsList : list               Optional: Ordered list of dictionaries, where each dictionary represents the keyword arguments passed to the corresponding function
    nProc : int                         Maximum number of concurrent training processes
    progressMessage : str               Message to be displayed in the progress bar
    returnMessage : str                 Message to be printed for each task completion (empty "{}" are formatted using the tasks return output)
    taskUpdateSecs : int                Number of seconds to wait between checking for completed tasks
    
    Returns
    -------
    None
    """
    if taskArgsList is None: taskArgsList = [[] for _ in taskKwargsList]        # If no args are passed, create a list of empty lists with the same length as the kwargs list
    if taskKwargsList is None: taskKwargsList = [{} for _ in taskArgsList]      # If no kwargs are passed, create an empty list of dictionaries with the same length as the args list
    with Pool(nProc) as taskPool:                                               # Create a pool of tasks with nProc slots
        asyncResults = [taskPool.apply_async(taskFunc, taskArgs, taskKwargs)    # As each task may use a different callable, every task must be added to the pool individually
                        for taskFunc, taskArgs, taskKwargs
                        in zip(taskFuncs, taskArgsList, taskKwargsList)]        # ... submit the list of callables and corresponding args/kwargs in trios
        progressBar = tqdm(desc=progressMessage, total=len(taskFuncs))          # As tasks will finish out of order, we will not wait for get() on each asyncResult, instead we will monitor asyncResults and update everything accordingly
        
        while True:                                                             # Loop until exit condition is reached (all tasks have completed)
            asyncResultsTemp = []                                               # Store all unfinished tasks from each iteration here temporarily (cannot pop list elements while iterating that same list)
            for result in asyncResults:                                         # Iterate over all tasks ...
                if result.ready():                                              # ... if the task reports that it has completed
                    output = result.get()                                       # ... get the task's output (this should be instant)
                    if returnMessage is not None:                               # ... if the user has supplied a returnMessage
                        progressBar.write(returnMessage.format(*output))        # ... format the returnMessage using the task's output and print it via the progress bar's write method (as to not break the progress bar)
                    progressBar.update(1)                                       # ... also increment the progress bar
                else:                                                           # If the task reports that it is not yet done ...
                    asyncResultsTemp.append(result)                             # ... add it to the temporary list of active tasks
            asyncResults = asyncResultsTemp                                     # Finished iterating over active task list, update old list with new list
            
            if len(asyncResults) == 0:                                          # Exit condition: if the new list of tasks is now empty ...
                break                                                           # ... exit the loop, we have finished all tasks
            sleep(taskUpdateSecs)                                               # Otherwise, wait before checking the task list again

def dbTensorUpdate(domain: str, nProc: int, trainingParamsOverride: dict[str, Any]) -> None:
    """
    Manages the creation/updating of tensors from HFM data used by
        ML/RBF algorithms
    
    Parameters
    ----------
    domain : str                        String specifying the domain type (either "2D" or "axisym")
    nProc : int                         Maximum number of concurrent tensor update processes
    trainingParamsOverride : dict[str, Any]  Dictionary of trainingParams that will supersede loaded values

    Returns
    -------
    None
    """
    caseDatabasePaths = [case for case in Path(f"./caseDatabase/{domain}").glob("*") if case.is_dir()]  # Get list of paths for all cases in ./caseDatabase/{domain}
    if len(caseDatabasePaths) == 0:                                             # If there are no cases for the requested domain in ./caseDatabase, exit
        print(f"No available cases found in ./caseDatabase/{domain} for tensor update process!")
        return
    
    trainingParams = loadyaml("trainingParams", override = trainingParamsOverride)  # Load the training parameters from ./resources/trainingParams.yaml; edit this file to use different parameters
    availableRe = sorted(list(set(case.stem.split("_")[1] for case in caseDatabasePaths)))  # Get set of all available Re in ./caseDatabase/{domain} (convert it back into a list to preserve order in the next steps)
    print(f"Found {len(caseDatabasePaths)} available cases for {len(availableRe)} different Reynolds numbers in ./caseDatabase/{domain}")
    
    tensorParentDir = Path(f"./mlData/{domain}")                                # Construct parent tensor directory of the form ./mlData/{domain}
    tensorDirStems = [f"Re_{Re}_modes_{trainingParams['modes']}" for Re in availableRe]  # Construct list of tensor subdirectory stems, one for each unique Re
    caseDirLists = [[case for case in caseDatabasePaths if case.stem.startswith(f"Re_{Re}_")] for Re in availableRe] # Construct list containing lists of case directories (each corresponding to one entry in tensorDirs), one for each unique Re
    featureSizes = [4 for _ in availableRe]                                     # Construct list of maximum number of features per available Re (these will be modified with a harmonic in the worker)
                    
    if len(availableRe) > 1:                                                    # If there is more than one Re in availableRe, we need to generate an Re_All case, add one more entry to each of the above lists
        tensorDirStems += [f"Re_All_modes_{trainingParams['modes']}"]           # tensorDirStem is Re_All_*
        caseDirLists += [caseDatabasePaths]                                     # Contains all cases in ./caseDatabase
        featureSizes += [5]                                                     # Re is the 5th feature
                   
    taskArgs = [[tensorParentDir, tStem, cDirs, features, trainingParams['modes']]  # Construct list of arguments for each tensor creation task computeAllTensors (one for each single-Re and one for all-Re)
                for tStem, cDirs, features                                      # ... each task will receive the tensor parent directory, one tensor sub-directory stem, list of corresponding case directories, and the number of features ...
                in zip(tensorDirStems, caseDirLists, featureSizes)]             # ... where the number of features is 4 for all single-Re tasks (A1, A2, k1, k2) and 5 for the one multi-Re task (A1, A2, k1, k2, Re)
    taskFuncs = [computeAllTensors for _ in taskArgs]                           # As all function calls are made to the same function, create a list of size taskArgs with computeAllTensors callables only

    genericPoolManager(taskFuncs, taskArgs, None, nProc, "Updating tensor groups from HFM data", None)  # Dispatch all tasks to the generic pool manager
    print("Tensor update process completed successfully")
    
def computeAllTensors(tensorParentDir: Path, tensorDirStem: str, caseDirList: list[Path], features: int, modes: int) -> None:
    """
    Worker function for creating/updating tensors from HFM data used by
        ML/RBF algorithms
    
    Parameters
    ----------
    tensorParentDir : pathlib.Path      Path of parent directory where all tensors for this tensor group will be stored (./mlData/{domain})
    tensorDirStem : str                 Tensor sub-directory stem name of the form "Re_{Re}_modes_{modes}", will have "_harmonic_{harmonic}" appended to it in a loop
    caseDirList : list                  List of paths to cases in ./caseDatabase that will be used for tensor computation
    features : int                      Number of features that each tensor will contain
    modes : int                         Number of modes used by PCA

    Returns
    -------
    None
    """
    bcvNames = ["inletU", "inletT", "inletp", "outletU", "outletT", "outletp"]  # Ordered list of names of all boundary condition variables (BCV) that will be computed
    bcvTensorNames = ["spatialDataExpanded.pt", "spatialData.pt", "spatialMin.pt", "spatialMax.pt",  # List of all BCV- and PCA-related tensors that will be stored
                      "modalData.pt", "modalMin.pt", "modalMax.pt", "VTReduced.pt"]
    lumpedTensorNames = ["lumpedDataExpanded.pt", "lumpedData.pt", "lumpedMin.pt", "lumpedMax.pt"]  # List of all lumped tensors that will be stored
    
    def _inferFeatures(caseDir):
        """
        Private function for inferring feature values from caseDir path name
        """
        _, Re, _, A1, A2, _, k1, k2 = caseDir.name.split("_")
        return np.array([float(feature.replace("-", ".")) for feature in [A1, A2, k1, k2, Re]], dtype=np.float32)

    xExpanded = torch.from_numpy(np.array([_inferFeatures(caseDir) for caseDir in caseDirList]))  # Infer feature values from path name for each case directory and convert it to an unstandardised (expanded) feature tensor
    xExpanded = xExpanded[:, :features]                                         # Only use the first n columns, where n is the number of features (if computing single-Re tensors, the last Re variable is not used)
    try:                                                                        # Attempt to load an existing xData tensor (this will fail if it doesn't exist)
        xExpandedExisting = torch.load(tensorParentDir / f"{tensorDirStem}_harmonics_2" / "xData.pt", weights_only=True)["xExpanded"]  # Load the xData[xExpanded] harmonics=2 tensor (if it exists in the current tensorDir)
        if torch.equal(xExpanded, xExpandedExisting):                           # ... compare the new and existing x-data tensors
            return                                                              # ... if they are identical, assume that this tensor group is up-to-date and exit
    except FileNotFoundError:                                                   # If the xData.pt tensor doesn't exist ...
        pass                                                                    # ... ignore the error, it will be created in the following steps
    
    harmOneIdx = (xExpanded[:, 1] == 0) & (xExpanded[:, 3] == 0)                # Get a tensor of bools representing the case indices where A2=0 and k2=0 (only these cases will be selected for harmonics=1)
    harmOnexExpanded = torch.index_select(xExpanded[harmOneIdx], 1, torch.arange(0, xExpanded.shape[1], 2))  # Keep ever second index of xExpanded, thereby removing the redundant A2 and k2 columns, for harmonics=1
    harmOneCaseDirList = [case for case, keep in zip(caseDirList, harmOneIdx) if keep]  # ... also filter out case directories that are not needed for harmonics=1 (keep only cases where A2=0 and k2=0)
    
    for harmonics, xExpanded, caseDirs in [[1, harmOnexExpanded, harmOneCaseDirList],  # Iterate over two scenarios, one where harmonics=1 (caseDirs where A2=0 and k2=0, and xExpanded has 2 or 3 cols) ...
                                           [2, xExpanded, caseDirList]]:          # ... and one where harmonics=2 (all available caseDirs for the current Re, and xExpanded has 4 or 5 cols)
        x, xMin, xMax = normaliseTensor(xExpanded)                                # Normalise the current xExpanded tensor into x, as well as the min and max of the original tensor across dim 0 that can be used to recover it
        bcvAndSvd = computeBCVandSVD(bcvNames, caseDirs, modes)                   # Compute BCV and PCA tensors, output is a dictionary of lists where each key is one of bcvNames and each item in the lists is data for one of the tensorNames
        lumpedTExpanded, lumpedpExpanded = lumpedDataCalculation([bcvAndSvd[key][0] for key in bcvNames])  # Evaluate the Thermo-Hydraulic Performance (THP) to obtain all "lumped" data
        lumpedTensors = zip([lumpedTExpanded] + normaliseTensor(lumpedTExpanded),  # Compute the normalised versions, min and max of both the lumpedTExpanded and lumpedpExpanded tensors
                            [lumpedpExpanded] + normaliseTensor(lumpedpExpanded))  # ... and zip them so they are accessible in pairs
                
        baselineCaseMask = torch.all(torch.eq(x, 0), 1)                         # Mask representing case where A1, A2, k1, k2 = 0
        thp = lumpedTExpanded.squeeze() - lumpedpExpanded.squeeze()             # Compute thermohydraulic performance
        maxTHPCaseMask = torch.eq(thp, torch.max(thp).item())                   # Mask representing max THP case

        tensorDir = tensorParentDir / f"{tensorDirStem}_harmonics_{harmonics}"  # Construct the full tensor directory for this scenario, labelled by Reynolds number, number of modes, and number of harmonics
        tensorDir.mkdir(parents=True, exist_ok=True)                            # Create the directory where all tensors in this group will be stored
        torch.save({name: tensor for name, tensor in zip(["xExpanded", "xMin", "xMax", "x"], [xExpanded, xMin, xMax, x])}, tensorDir / "xData.pt")  # Save all x-related data in an xData dictionary of tensors
        for arrIdx, tensorName in enumerate(bcvTensorNames):                    # Iterate over all BCV- and PCA-related tensor file names (and their corresponding indices, used to index the output of computeBCVandSVD
            torch.save({key: bcvAndSvd[key][arrIdx] for key in bcvNames}, tensorDir / tensorName)  # Store data in dictionary of tensors (where each key is one bcvName); each entry has a 2D shape (len(csaeDirList), mesh/modes)
        for tensorName, tensorGroup in zip(lumpedTensorNames, lumpedTensors):   # Iterate over all lumped-related tensors and their file names, (grouped by expanded, standardised, mean and std tensors)
            torch.save({name: tensor for name, tensor in zip(["lumpedT", "lumpedp"], tensorGroup)}, tensorDir / tensorName)  # Store data from each group in a dictionary, where the keys are lumpedT or lumpedp
            
        torch.save({'profileBaseline': [bcvAndSvd[key][0][baselineCaseMask][0] for key in ["inletp", "inletU", "inletT", "outletp", "outletU", "outletT"]],  # keys represent order of entries as expected by plotting routine
                    'profileOptimised': [bcvAndSvd[key][0][maxTHPCaseMask][0] for key in ["inletp", "inletU", "inletT", "outletp", "outletU", "outletT"]]},
                    tensorDir / "variableProfiles.pt")                          # Store data for plotting variable profiles, comparing baseline vs max cases
        
def computeBCVandSVD(bcvNames: list[str], caseDirList: list[Path], modes: int, postProcessingDir: str = "postProcessing/sample/Helium/6000/") -> dict[str, list[torch.tensor]]:
    """
    Load high fidelity model (HFM) boundary condition variable (BCV) data
        and use it to compute spatial and modal tensors

    Parameters
    ----------
    bcvNames : list                     Ordered list of names of all boundary condition variables (BCV) that will be computed
    caseDirList : list                  List of cases from which BCV data will be loaded
    modes : int                         Number of modes for PCA
    postProcessingDir : str             Location inside each case directory where BCV data is stored

    Returns
    -------
    bcv : dict                          Dictionary containing all BCV/SVD data, where each entry in bcv[bcvName] is a list of standardised and expanded spatial and modal data, including means and standard deviations
    """
    samplingFormats = loadyaml("samplingFormats")                               # Load data from ./resouces/samplingFormats.yaml containing BCV data location information for each supported OpenFOAM version
    bcv = {name: [] for name in bcvNames}                                       # Create dictionary of lists, where each key will initially contain an empty list that will be filled with loaded BCV datasets for stacking
    for caseDir in caseDirList:                                                 # Iterate over all relevant case directories, loading all available BCV data
    
        redirect = list(caseDir.glob("*.redirect"))                             # Find all files with extension ".redirect" in caseDir; if no matches, caseDir is original and remains the same
        if redirect:                                                            # If True (list contains an entry), this caseDir is a duplicate, and we should point to the original caseDir
                caseDir = caseDir.with_name(redirect[0].stem)                   # File name (without extension) of the redirect is the folder name of the original caseDir, replace it and set that as the current caseDir

        openfoamVersion = None                                                  # As each case could have been generated with a different OpenFOAM version, we need to identify the correct one to use (start with None)
        for version in samplingFormats.keys():                                  # Each key in samplingFormats is a supported OpenFOAM version number, iterate over all supported versions ...
            if Path(caseDir / postProcessingDir / samplingFormats[version]["checkFor"]).exists():  # ... for each version, a file exists under key "checkFor", where the presence of this file identifies the OpenFOAM version to use ...
                openfoamVersion = version                                       # ... if the file exists, set this as the OpenFOAM version that will be used to deduce the format/location of the required BCV data
                break
        if openfoamVersion is None:                                             # If no "checkFor" file matched, this case was generated with an unsupported OpenFOAM version, cannot continue until this is resolved by the user
            raise RuntimeError(f"Case {caseDir} was generated using an unsupported version of OpenFOAM! Manually delete this case and regenerate it with a supported OpenFOAM version,"
                               " or update ./resources/samplingFormats.yaml with the relevant sampling format information for the OpenFOAM version that generated this case.")

        for name in bcvNames:                                                   # Iterate over all BCV names that need reading ...
            fileName, fileCols = samplingFormats[version][name]                 # ... use a shorthand for the file name (and corresponding column in the file) where the BCV data identified by bcvName can be found
            bcv[name].append(np.loadtxt(caseDir / postProcessingDir / fileName, usecols=fileCols, dtype=np.float32))  # ... load the data into an array and append it to the list, where it will later be concatenated and used for further computation

    for key in bcv.keys():                                                      # Iterate over all keys (bcvNames) in the BCV dictionary ...
        varExpanded = torch.from_numpy(np.stack(bcv[key]))                      # ... stack data from each case for this bcvName, creating the 2D denormalised (expanded) spatial tensor with the shape (len(caseDirList), mesh)
        var, varMin, varMax = normaliseTensor(varExpanded)                      # ... normalise the spatial tensor and get the min and max of the original expanded tensor
        U, s, VT = svd(varExpanded)                                             # ... perform Singular Value Decomposition (SVD) on the expanded spatial tensor, resulting in (U = left eigenvectors, s = eigenvalues, VT = right eigenvectors)
        sigmaMidStep = torch.zeros_like(varExpanded)                            # ... to truncate, the sigmaMidStep zero-filled tensor is created with the same shape as the spatial tensor (eigenvalue count as length of each dimension)
        sigmaMidStep[:s.shape[0], :s.shape[0]] = torch.diag(s)                  # ... then, the diagonal of the intermediatery tensor is populated with the eigenvalues
        sigmaReduced = sigmaMidStep[:, :modes]                                  # ... finally, the modes chosen by the user in './resources/trainingParams.yaml' are used to truncate the tensor
        VTReduced = VT[:modes, :]                                               # ... the chosen number of modes are also used to truncate the right eigenvectors, which will be stored
        modalExpanded = U.mm(sigmaReduced)                                      # ... from the truncated tensor of left eigenvalues, a dimension-reduced tensor is obtained and used as the modal data tensor
        modal, modalMin, modalMax = normaliseTensor(modalExpanded)              # ... this tensor needs normalisation as it was constructed using expanded data
        bcv[key] = [varExpanded, var, varMin, varMax, modal, modalMin, modalMax, VTReduced]  # Replace existing list of loaded BCV data in BCV dictionary for the current key (bcvName) with a list of tensors

    return bcv

def lumpedDataCalculation(bcvData: list[torch.tensor], Cp: float = 5230.0) -> tuple[torch.tensor, torch.tensor]:
    """
    Perform a Thermo-Hydraulic Performance (THP) Evaluation based on the
        advected heat flux for thermal beahviour, and dissipation rate for
        hydraulic behaviour, generating "lumped" data
    
    Parameters
    ----------
    bcvData : list                      List of unstandardised (expanded) BCV data tensors in order [inU, inT, inp, outU, outT, outp], where each tensor has the shape (len(caseDataList), mesh)
    Cp : float                          Specific heat capacity at a constant pressure

    Returns
    -------
    lumpedT : torch.tensor              Computed advected heat flux tensor from BVC spatial data
    lumpedp : torch.tensor              Computed dissipation rate tensor from BCV spatial data
    """
    def _bSubT(temperature, velocity):
        """Computes the advected heat flux prior to performing integration
            over the given boundary"""
        return (temperature * Cp) * velocity
    
    def _bSubP(pressure, velocity):
        """Computes the dissipation rate prior to performing integration
            over the given boundary"""
        return  (pressure + (1/2) * (velocity**2)) * velocity
    
    inU, inT, inp, outU, outT, outp = bcvData                                   # Expand BCV tensors from list by their shorthand name
    lumpedT = torch.trapz( _bSubT(outT, outU) - _bSubT(inT, inU) )              # Calculate the integral of the substraction between boundaries for advected heat flux
    lumpedp = - torch.trapz( _bSubP(outp, outU) - _bSubP(inp, inU) )            # Calculate the integral of the substraction between boundaries for dissipation rate
    return torch.unsqueeze(lumpedT, 1), torch.unsqueeze(lumpedp, 1)             # Add an extra dimension to each 1D tensor to conform to the format (n, outputDimension)

###############################################################################

if __name__ == "__main__":
    raise RuntimeError("This script is not intended for execution. Instead, execute hammerhead.py from the parent directory.")
