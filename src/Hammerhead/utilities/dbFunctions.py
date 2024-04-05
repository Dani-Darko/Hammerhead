##############################################################################################
#                                                                                            #
#       HAMMERHEAD - Harmonic MOO Model Expedient for the Reduction                          #
#                    Hydraulic-loss/Heat-transfer Enhancement Asset Design                   #
#       Author: Daniela Minerva Segura Galeana                                               #
#       Supervised by: Antonio J. Gil                                                        #
#                                                                                            #
##############################################################################################
#                                                                                            #
#       Script containing functions for populating simulation case database. These functions #
#           are called  as part of HFM directly from hammerhead.py (if running in console    #
#           mode), or as part of a Hammerhead GUI event.                                     #
#                                                                                            #
##############################################################################################

# IMPORTS: HAMMERHEAD files ###############################################################################################################
   
from utilities.dataProcessing import loadyaml                                   # Utilities -> File loading and manipulation from ./resources directory
from utilities.openfoamFunctions import (populateBlockMeshEdges,                # Utilities -> Populate OpenFOAM mesh coordinates using the requested shape parameters
                                         runMesh,                               # Utilities -> Run OpenFOAM blockMesh for the requested shape parameters
                                         runOpenFOAM,                           # Utilities -> Run OpenFOAM HFM for the requested shape parameters
                                         updateVelocity)                        # Utilities -> Update the velocity and turbulence model variables for the requested Reynolds parameter

# IMPORTS: Others #########################################################################################################################

from datetime import timedelta                                                  # Others -> Operations involving relative/elapsed time
from multiprocessing import Process, Queue                                      # Others -> Parallelisation and inter-process communication tools
from pathlib import Path                                                        # Others -> Path manipulation
from queue import Empty                                                         # Others -> Inter-process communcation queue exceptions
from time import sleep                                                          # Others -> Pause execution between subprocess status polls
from tqdm import tqdm                                                           # Others -> Progress bar

import numpy as np                                                              # Others -> Array/matrix/vector manipulation
import os                                                                       # Others -> Get subprocess PID
import shutil                                                                   # Others -> File/directory manipulation
import subprocess                                                               # Others -> External script execution

###########################################################################################################################################

def dbPopulate(domain: str, nProc: int, openfoam: str) -> None:
    """
    Using the specified high fidelty model parameters, compute and dispatch
        unique simulations cases to the multiprocessing database population
        task manager
    
    Parameters
    ----------
    domain : str                        String specifing the domain type (either "2D" or "axisym")
    nProc : int                         Maximum number of concurrent database population processes
    openfoam : str                      Name (or path) of compatible OpenFOAM executable

    Returns
    -------
    None
    """
    hfmParams = loadyaml("hfmParams")                                           # Load HFM parameters from ./resources/hfmParams.yaml; edit this file to change these parameters
    hfmParams["domainType"] = domain                                            # Add specified --domain parameter to HFM parameter dictionary (so all parameters are contained in a single obj)
    
    try:                                                                        # Load ignoreCaseList if it exists (a FileNotFoundError exception is raised otherwise)
        ignoreCaseList = np.loadtxt("./resources/ignoreCaseList.txt")           # This file contains points in the parameter space for which data already exists elsewhere (or data that is not needed)
    except FileNotFoundError:                                                   # If no ignoreCaseList.txt exists in the ./resources directory ...
        ignoreCaseList = np.array([]).reshape(0, 5)                             # ... construct an empty array with a shape matching the expected ignoreCaseList
    ignoreCaseList = set(tuple(x) for x in ignoreCaseList)                      # Convert ignoreCaseList array into a set of tuples (each tuple is a single case to be skipped)
    
    xArray, uniqueCases = computeUniqueCases(hfmParams)                         # Compute x-coordinates, and a list of unique cases, each containing the corresponding y-coordinates and a list of duplicate parameter combinations
    uniqueFilteredCases = []                                                    # Prepare empty list that will contain filtered unique cases ignoring specified parameter combinations
    for uniqueCase in uniqueCases:                                              # Iterate over all unique cases ...
        if not bool(set(tuple(x) for x in uniqueCase[1:]) & ignoreCaseList):    # ... if the set of cases (all duplicates of one unique case) is disjoint with the ignoreCaseList set (their union is empty)
            uniqueFilteredCases.append(uniqueCase)                              # ... add this unique case to the final list of cases to be used for database population
    
    dbTaskManager(xArray, uniqueFilteredCases, hfmParams, nProc, openfoam)      # Start task manager with final list of cases, which will process nProc cases simulateously and provide user with status updates
    print("Database population process has finished")


def computeUniqueCases(hfmParams: dict[str, float]) -> tuple[np.ndarray, list[list[np.ndarray, list[float]]]]:
    """
    Compute HFM surface shape for each parameter set case, and group all cases
        that have identical shapes into unique case lists containing the
        surface y-coordinates and all parameter combinations that correspond
        to the surface  
    
    Parameters
    ----------
    hfmParams : dict                    Dictionary of HFM parameters loaded from ./resources/hfmParams.yaml

    Returns
    -------
    xArray : array-like                 Array of surface x-coordinates
    uniqueCasesMultiRe : list           List of unique cases, where unique case is a list of the form [yArray, [params1], [params2], ..]
    """
    xArray = np.linspace(0, hfmParams["L"], hfmParams["dx"], endpoint=True)     # Compute an array of x-coordinates from mesh parameters, later used to compute the y-coordinates of the surface shape for each case
    A1, A2, k1, k2, Re = [np.round(np.linspace(hfmParams[f"{param}_min"],       # Compute arrays of equally-spaced values for all parameters (A1, A2, k1, k2, Re) where ranges are defined ...
                                           hfmParams[f"{param}_max"],           # ... by the minimum and maximum values (and the total number of samples in that range) defined in hfmParams.yaml
                                           hfmParams[f"{param}_num"],           # ... in order to populate the test parameter space with all available parameter combinations
                                           endpoint=True),                      # Both the minimum and maximum values are included in the array [min, max] 
                               decimals = 3)                                    # Round all values in all arrays to three decimal places
                      for param in ["A1", "A2", "k1", "k2", "Re"]]
    Re = Re.astype(int)                                                         # Restrict all Reynolds numbers to be integers only (all requested floats will be rounded down)
    print("Computed HFM parameter ranges from ./resources/hfmParams.yaml configuration file:")
    for param in ["A1", "A2", "k1", "k2", "Re"]:
        print(f"\t{param}: {list(locals()[param])} ({len(locals()[param])})")
    
    caseList = [[w, x, y, z] for w in A1 for x in A2 for y in k1 for z in k2]   # Compute a list all of possible combinations of parameter values (cases)
    print(f"Generated {len(caseList)*len(Re)} cases representing all parameter combinations")
    
    uniqueCases = [[computeYCoords(xArray, *caseList[0], hfmParams["r"]),       # Create a list of unique cases (initialised using the first case in caseList) that will contain [yArray, [params_1], [params_2], ...]
                    caseList[0]]]                                               # ... where parameter combinations resulting in identical y-coordinate arrays are grouped (each entry will contain at least one set of parameters)
    for case in tqdm(caseList[1:], initial=1, total=len(caseList), unit_scale=len(Re), desc="Grouping unique cases"):  # Iterate over all cases (except for the first), displaying a progress bar
        yArray = computeYCoords(xArray, *case, hfmParams["r"])                  # Compute an array of y-coordinates for current set of parameters
        for uniqueCase in uniqueCases:                                          # Iterate over all existing unique cases (where yArray is the first entry in the list)
            if np.array_equal(uniqueCase[0], yArray):                           # ... if the computed yArray is identical to that case's existing yArray 
                uniqueCase.append(case)                                         # ... append this parameter set to that case's list of parameter sets (all parameter sets in this list generate the same yArray)
                break                                                           # ... no need to search through the rest of the unique case groups once a match was found
        else:                                                                   # If no matching yArray is found, this is a new unique case
                uniqueCases.append([yArray, case])                              # ... add a new sublist to the list of unique cases, containing the new unique yArray and the paramater set that created it
    
    uniqueCasesMultiRe = []                                                     # As each unique case needs to be considered for all requested Re, create an empty list that will contain unique cases including the Re parameter
    for numRe in Re:                                                            # For each requested Re value
        for uniqueCase in uniqueCases:                                          # ... iterate over all existing unique cases
            uniqueCasesMultiRe.append([uniqueCase[0], *[[*case, numRe] for case in uniqueCase[1:]]])  # ... recreating the unique case but now including Re as the last entry in each parameter set
    print(f"Found {len(uniqueCasesMultiRe)} unique cases")
    
    return xArray, uniqueCasesMultiRe

def computeYCoords(xArray: np.ndarray, A1: float, A2: float, k1: float, k2: float, r: float) -> np.ndarray:
    """
    Compute and return y-coordinates using the requested shape parameters at
        the provided x-coordinates 
    
    Parameters
    ----------
    xArray : array-like                 X-coordinates for shape parametrisation
    A1 : float                          Main amplitude value
    A2 : float                          Secondary amplitude value
    k1 : float                          Main wavenumber value
    k2 : float                          Secondary wavenumber value
    r : float                           Pipe radius

    Returns
    -------
    yArray : array-like                 Y-coordinates for shape parametrisation
    """
    return (  (-A1 * np.cos(xArray * k1 * np.pi))
            + (-A2 * np.cos(xArray * k2 * np.pi))
            + A1 + A2 + r)

def dbTaskManager(xArray: np.ndarray, uniqueCases: list[list[np.ndarray, list[float]]], hfmParams: dict[str, float], nProc: int, openfoam: str, taskUpdateSecs: int = 2) -> None:
    """
    Manages the execution of up to nProc parallel subprocesses, each working on
        one of the provided unique cases; inter-process communication is
        collected from each subprocess in order to display a status board
        that is periodically updated
    
    Parameters
    ----------
    xArray : array-like                 Array of surface x-coordinates
    uniqueCases : list                  List of unique cases, where each unique case is a list of the form [yArray, [params1], [parms2], ...]
    hfmParams : dict                    Dictionary of HFM parameters loaded from ./resources/hfmParams.yaml
    nProc : int                         Maximum number of cases to process in parallel
    openfoam : str                      Name (or path) of compatible OpenFOAM executable
    taskUpdateSecs : int                Seconds between task status check (cleanup and resubmission), and message board update
    
    Returns
    -------
    None
    """
    messageQueue = Queue()                                                      # Message queue with a single empty space, subprocesses will put status updates in the queue and the task manager will collect them
    runningProcesses = []                                                       # List that will contain all currently running Process objects
    statusBoard = {}                                                            # Message board where subprocess status is stored, in the form {pid1: [name, status, timestep, clocktime], pid2: [name, status, timestep, clocktime], ...}
    colWidthsMax = np.array([0, 0, 0, 0, 0])                                    # Maximum per-column widths of all table content (not including whitespace)
    lastPrintRows = 0                                                           # Number of rows last printed during status board update (recorded so that those rows can be cleared on next update)
    nextTaskIndex = 0                                                           # Index of unqiue case in uniqueCases that will be assigned to the next available process
    processedCases = 0                                                          # Counter recording the number of processed non-unique cases
    totalCases = sum(len(caseData[1:]) for caseData in uniqueCases)             # Total number of non-unique cases to be processed
    
    while True:                                                                 # Loop forever until exit conditions (all tasks submitted and no tasks running) are met
        for _ in range(nProc - len(runningProcesses)):                          # TASK SUBMISSION STEP: Repeat until nProc processes are running
            try:                                                                # Attempt to submit a unique case to the worker process (this will fail if there are no more cases that need processing)
                runningProcesses.append(Process(                                # Create a Process object and append it to the list of running processes (process will be started on the next step)
                    target=dbComputeHFMProcess,                                 # ... the process will launch the dbComputeHFMProcess function as a subprocess
                    args=[uniqueCases[nextTaskIndex][1:],                       # Pass the following arugments to the dbComputeHFMProcess function: list of all parameter combinations for the current unique case
                          hfmParams,                                            # ... dictionary of HFM parameters previously loaded from ./resources/hfmParams.yaml
                          xArray,                                               # ... array of surface x-coordinates (valid for all cases)
                          uniqueCases[nextTaskIndex][0],                        # ... corresponding array of y-coordinates
                          openfoam,                                             # ... name or path of OpenFOAM executable
                          messageQueue]))                                       # ... message queue that the subprocess will put status updates into
                runningProcesses[-1].start()                                    # Start the execution of the last created Process object
                processedCases += len(uniqueCases[nextTaskIndex][1:])           # Increment the processed cases counter by the amount of parameter combinations this unique case contained
                nextTaskIndex += 1                                              # Increment the task counter so that during the next iteration, the next task unique case will be processed
            except IndexError:                                                  # Exception is raised if nextTaskIndex > len(uniqueCases) - 1 (all unique cases have already been processed)
                pass                                                            # ... do nothing, as we are now waiting for the final running tasks to finish
        
        while True:                                                             # MESSAGE BOARD UPDATE STEP: Repeat until exit condition (message queue is empty)
            try:                                                                # Attempt to get content of message queue, this will fail if the queue is empty
                pid, name, status = messageQueue.get(block=False)               # Each subprocess will place its PID, name (first parameter combination) and current status on the queue
                if pid in statusBoard and status == "Done":                     # If the PID exists in the status board and the process reports "Done" ...
                    statusBoard.pop(pid)                                        # ... remove it's entry from the status board as it completed and a new task will be submitted in its place
                    continue                                                    # ... (do nothing more with the message)
                statusBoard[pid] = [name, status, "0", "0"]                     # Otherwise, update the status of matching process in the status board (leave timestep and clocktime as unknown, will be updated later if available)
            except Empty:                                                       # If the queue is empty (and no more processess submitted messages while we were processing the last status update)
                break                                                           # ... we can stop processing status updates and move onto the next stage
                
        for pid in statusBoard.keys():                                          # Iterate over all PIDs in message board
            try:                                                                # Attempt to extract timestep and clocktime from chtMultiRegionSimpleFoam log file (this can fail for various reasons)
                log = subprocess.run(                                           # Execute bash script to get last few lines of chtMultiRegionSimpleFoam log file for current thread's PID
                    "tail -n 25"                                                # Only last 25 lines are necessary to capture both timestep and clocktime (with some redundancy, if multiple are captured, last is stored)
                    f" $(ps --ppid $(ps --ppid {pid} -o pid=) -o cmd="          # Grand-child of thread's PID represents Allrun script of that case, use ps to find child process, and ps again to find cmd of grand-child process
                    " | cut -d ' ' -f 2"                                        # Use cut to extract the second field, the path to the Allrun script (from which path to the corresponding logfile can be extracted)
                    " | sed 's/Allrun/log.chtMultiRegionSimpleFoam/g')",        # Use sed to replace Allrun with log.chtMultiRegionSimpleFoam, creating full path to the current thread's logfile which will be read by tail
                    shell=True, capture_output=True, check=True, timeout=1      # Output of tail command will be captured, if tail fails or hangs due to insufficient input, check and timout parameters ensure script fails safely
                    ).stdout.decode().splitlines()                              # Extract tail output from stdout (as binary), convert to a string, and split into a list of lines
                statusBoard[pid][2] = [line for line in log if line.startswith("Time =")][-1].split()[-1]  # Find line formatted as "Time = X", extract "X"
                statusBoard[pid][3] = str(timedelta(seconds=int([line for line in log if "ClockTime" in line][-1].split()[-2])))  # Find line formatted as "ExecutionTime = Y s  ClockTime = Z s", extract Z and convert from seconds to time string
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError, ValueError):  # If the above failed, either chtMultiRegionSimpleFoam is not yet running, or logfile contains incomplete data
                continue                                                        # ... ignore this PID, leaving the default values of zero for both fields
        
        print(f"\033[{lastPrintRows}A\033[J", end=None)                         # MESSAGE BOARD DISPLAY STEP: Print \033[nA (move up n rows) and \033[J (clear console from current row until end)
        lastPrintRows = len(statusBoard) + 7                                    # Update the printed row counter with the number of rows that are about to be printed (7 from table formatting and other counters)
        colWidths = [max([len(pid) for pid in statusBoard.keys()] + [3])]       # First column width is the dictionary key (PID), get maximum length of all PIDs in status board ([3] is minimum column width)
        for idx, minWidth in enumerate([22, 6, 4, 7]):                          # Iterate over all following columns (and their min width), corresponding to list entries of matching PID key in status board dictionary
            colWidths.append(max([len(pinfo[idx]) for pinfo in statusBoard.values()] + [minWidth]))  # For each column compute the maximum width of the column's content, taking into account the minimum width
        colWidthsMax = np.maximum(colWidths, colWidthsMax)                      # To prevent repetitive table resizing, take element-wise maximum of all columns and always used the largest recorded column width
        
        hLineStr = f"|{'|'.join(['-'*(width+2) for width in colWidthsMax])}|"   # Construct string representing the horizontal line of the message board table
        statusBoardStr = [f"| {' | '.join([getattr(item, just)(width) for item, width, just in zip([key] + value, colWidthsMax, ['ljust']*3 + ['rjust']*2)])} |"  # Construct list of strings (one for each row), padding and aligning column content
                          for key, value in statusBoard.items()]                # ... one string is generated for each key-value pair in the dictionary
        print('\n'.join([hLineStr,                                              # Join and print all the above and below status board strings
                         f"| {' | '.join([item.center(width) for item, width in zip(['PID', 'Unique Case Parameters', 'Status', 'Step', 'Elapsed'], colWidthsMax)])} |",  # Table header row
                         hLineStr,
                         *statusBoardStr,
                         hLineStr,
                         f"Processed tasks: {nextTaskIndex}/{len(uniqueCases)} ({(nextTaskIndex / len(uniqueCases))*100:.0f}%)",  # Counter (fraction and percentage) of total processed unique cases
                         f"Processed cases: {processedCases}/{totalCases} ({(processedCases / totalCases)*100:.0f}%)"]), end=None)  # Counter (fraction and percentage) of total processed non-unique cases

        runningProcessesTemp = []                                               # TASK CLEANUP STEP: create an empty list where all tasks that are still running will be appended to
        for process in runningProcesses:                                        # Iterate over all "running" processes (some may have finished execution at this stage)
            if process.is_alive():                                              # If the process is still alive (executing)
                runningProcessesTemp.append(process)                            # ... append it to the new list of running processes
            else:                                                               # If the process is no longer alive (finished execution)
                process.close()                                                 # ... free resources used by the process (and do not append it to the new list of running processes)
        runningProcesses = runningProcessesTemp                                 # All processes in the new list are now confirmed to be running, set the new runningProcess list as the current one
        
        if len(runningProcesses) == 0 and nextTaskIndex == len(uniqueCases):    # EXIT CONDITION: if no processes are running and the next task index is equal to the length of the unique case list
            break                                                               # ... all tasks have completed and no more tasks are waiting to be launched, break the loop
        
        sleep(taskUpdateSecs)                                                   # Wait this many seconds before the next loop iteration
        
def dbComputeHFMProcess(uniqueCase: list[list[float]], hfmParams: dict[str, float], xArray: np.ndarray, yArray: np.ndarray, openfoam: str, messageQueue: Queue) -> None:
    """
    Check for existence of completed cases matching one of the case parameters,
        and if one exists, use it to populate all non-unique cases for this
        task. Otherwise, prepare a case directory corresponding to the first
        set of parameters in uniqueCase with files for OpenFOAM execution,
        execute and verify the success of the HFM, and copy the final directory
        contents to all other non-unique cases for this unique case.
            
    Parameters
    ----------
    uniqueCase : list                   List containing all non-unique amplitude/wavenumber parameter sets for the requested unqiue case [[A1, A2, k1, k2, Re], [...]]
    hfmParams : dict                    Dictionary of HFM parameters loaded from ./resources/hfmParams.yaml
    xArray : array-like                 X-coordinates for shape parametrisation
    yArray : array-like                 Y-coordinates for shape parametrisation
    openfoam : str                      Name (or path) of compatible OpenFOAM executable
    messageQueue : Queue                Queue object with a single space, where this process will submit status updates to for the manager to collect
    
    Returns
    -------
    None
    """
    pid = str(os.getpid())                                                      # Get the process ID as reported by the OS (this will differ for each process)
    name = "A1={} A2={} k1={} k2={} Re={} (and {} duplicates)".format(*uniqueCase[0], len(uniqueCase) - 1)  # Construct the "name" of this process based on the case parameters and duplicate count
    baseCasePath = None                                                         # Path of the "base" case, if found, will be used to populate all other duplicates (and simulation will not need to run)
    caseData = []                                                               # List of paths and parameters for cases that are non-existent and need to be populated [[path, *params], ...]
    
    messageQueue.put((pid, name, "Scanning for existing cases"))
    for caseParams in uniqueCase:                                               # Iterate over all parameter sets in this unique case
        A1, A2, k1, k2, Re = caseParams                                         # Expand parameters from current case parameter set
        caseName = "Re_{}_A_{}_{}_k_{}_{}".format(Re, *[str(p).replace('.', '-') for p in [A1, A2, k1, k2]])  # Construct case directory name from parameter values (replacing dots with dashes)
        casePath = Path(f"./caseDatabase/{hfmParams['domainType']}/{caseName}")  # New case directory based on current parameter space; this is the cwd for each OpenFOAM HFM
        if (casePath / "postProcessing").exists():                              # If the directory already exists (and contains the postProcessing subdirectory), no simulations need to run
            baseCasePath = casePath                                             # ... set this directory as the base case, which will be copied to all other cases in caseData
        else:                                                                   # If the directory does not eixst or is invalid (does not contain ./postProcessing)
            caseData.append([casePath, *caseParams])                            # ... append the case path and its parameters to caseData (list of cases that will need to be populated)
            shutil.rmtree(casePath, ignore_errors=True)                         # In the case that the directory is not empty, it will need to be cleared to prepare it for population
            casePath.mkdir(parents=True)                                        # Create an empty directory for this case, which will be filled (either from baseCasePath or via OpenFOAM)

    if baseCasePath is not None:                                                # If a base case has been found, simulations will not need to run
        messageQueue.put((pid, name, "Populating duplicates"))
        for singleCaseData in caseData:                                         # Iterate over all empty cases in caseData ...
            shutil.copytree(baseCasePath, singleCaseData[0], dirs_exist_ok=True)  # ... copying all contents from baseCasePath to the current empty case
            createAkFile(*singleCaseData)                                       # Also create a machine-readable file with the current singleCaseData parameter set (used by ML functions)
        messageQueue.put((pid, name, "Done"))
        return                                                                  # No more needs to be done, all non-unique cases were populated using existing data

    try:                                                                        # No existing valid cases were found, simulations will need to run (first case in caseData is used throughout)
        messageQueue.put((pid, name, "Preparing case files")) 
        populateBlockMeshEdges(hfmParams["domainType"], caseData[0][0], xArray, yArray)  # Update surface coordinates to match the shape specified by current parameter set
        updateVelocity(caseData[0][0], hfmParams, Re)                           # Update velocity and turbulence variables for the current Reynolds number 
        createAkFile(*caseData[0])                                              # Create a machine-readable file with the current case parameter set (used by ML functions)
        messageQueue.put((pid, name, "blockMesh/stitchMesh"))
        runMesh(caseData[0][0], openfoam)                                       # Run OpenFOAM blockMesh and stitchMesh, and rearrange mesh files for further OpenFOAM execution
        messageQueue.put((pid, name, "chtMultiRegionSimpleFoam"))
        success = runOpenFOAM(caseData[0][0], openfoam)                         # Run OpenFOAM chtMultiRegionSimpleFoam in current case directory (record exit success/failure exit code)
        success = success and (caseData[0][0] / "postProcessing").exists()      # If no ./postProcessing subfolder exists, update success flag to False as failure must have occured elsewhere
    except KeyboardInterrupt:                                                   # Catch KeyboardInterrupt (raised when user enters Ctrl+C to stop process execution) in order to clean up
        success = False                                                         # Set success to False as OpenFOAM execution was interrupted
        raise                                                                   # Do not do anything else, the "finally" clause will carry out the clean-up and the exception is reported to the user
    finally:                                                                    # Perform final population operations here regardless of success or failure
        if success:                                                             # In case of success, populate all other cases using data computed by OpenFOAM for the first case
            messageQueue.put((pid, name, "Populating duplicates"))
            for singleCaseData in caseData[1:]:                                 # ... iterate over all cases in caseData (except for the first)
                    shutil.copytree(caseData[0][0], singleCaseData[0], dirs_exist_ok=True)  # .. copy contents of simulation directory to the current empty case
                    createAkFile(*singleCaseData)                               # ... also create a machine-readable file with the current singleCaseData parameter set (used by ML functions)
        else:                                                                   # In case of failure, move all logs to ./caseDatabase/failureLogs and delete all corresponding directories
            messageQueue.put((pid, name, "Failed, cleaning up"))
            failureLogsDir = Path(f'./caseDatabase/failureLogs/{hfmParams["domainType"]}/{caseData[0][0].stem}/')  # Construct path where simulation log files will be copied to
            shutil.rmtree(failureLogsDir, ignore_errors=True)                   # Delete directory if it already exists
            failureLogsDir.mkdir(parents=True)                                  # Create new directory where the failure logs will be copied to
            for logfile in caseData[0][0].glob("log.*"):                        # Iterate over all log files in simulation directory ...
                shutil.move(str(logfile), failureLogsDir)                       # ... moving each file to failureLogsDir
            for singleCaseData in caseData:                                     # Now iterate over all case directories created at the beginning of this function ...
                shutil.rmtree(singleCaseData[0])                                # ... delete each directory
        messageQueue.put((pid, name, "Done"))
        
def createAkFile(casePath: str, A1: float, A2: float, k1: float, k2: float, Re: float) -> None:
    """
    Create file in case directory containing requested parameter space data
    
    Parameters
    ----------
    casePath : str                      Path of simulation case directory
    A1 : float                          Main amplitude value
    A2 : float                          Secondary amplitude value
    k1 : float                          Main wavenumber value
    k2 : float                          Secondary wavenumber value
    Re : float                          Reynolds number

    Returns
    -------
    None
    """
    with open(casePath / "Ak.txt", "w") as fileAk:
        fileAk.write(f"{A1} {A2} {k1} {k2} {Re}\n")

###############################################################################

if __name__ == "__main__":
    raise RuntimeError("This script is not intended for execution. Instead, execute hammehead.py from the parent directory.")
