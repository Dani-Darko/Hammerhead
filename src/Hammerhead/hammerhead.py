##############################################################################################
#                                                                                            #
#       HAMMERHEAD - Harmonic MOO Model Expedient for the Reduction                          #
#                    Hydraulic-loss/Heat-transfer Enhancement Asset Design                   #
#       Author: Daniela Minerva Segura Galeana                                               #
#       Supervised by: Antonio J. Gil                                                        #
#                                                                                            #
##############################################################################################
#                                                                                            #
#       Main Hammerhead script. This should be called to access all Hammerhead features.     #
#           Supports GUI mode (default) or a console-only mode for use in headless           #
#           environments (HPC). Arguments passed to the script are parsed and provided to    #
#           all relevant internal functions.                                                 #
#                                                                                            #
##############################################################################################

# IMPORTS: HAMMERHEAD files ###############################################################################################################

from utilities.dbFunctions import dbPopulate                                    # Utilities -> Populate the database with new shape parameters
from utilities.mlFunctions import (mlTrain,                                     # Utilities -> Train the machine learning algorithms
                                   mlPlot,                                      # Utilities -> Plot 3D surface from the machine learning algorithms
                                   mlOptimalSearch)                             # Utilities -> Optimal shape prediction from machine learning algorithms
from utilities.dataProcessing import dbTensorUpdate                             # Utilities -> Process the data from the simulations/ml to train/plot/predict

# IMPORTS: Others #########################################################################################################################

from argparse import ArgumentParser, Namespace                                  # Other -> Parse and validate command line arguments
from multiprocessing import cpu_count                                           # Other -> Get number logical CPU cores
from subprocess import CalledProcessError, check_output                         # Other -> Check openFOAM executable exists in current environment

import os                                                                       # Other -> Path and working director manipulation tools

###########################################################################################################################################

def setup_argparse() -> ArgumentParser:
    """
    Sets up argument parser
    
    Parameters
    ----------
    None

    Returns
    -------
    parser : argparse.ArgumentParser namespace object of parsed args
    """
    parser: argparse.ArgumentParser = ArgumentParser(description="HAMMERHEAD - Harmonic MOO Model Expedient for the Reduction Hydraulic-loss/Heat-transfer Enhancement Asset Design")
    parser.add_argument('-c', '--console',  action="store_true",      required=False, default=False,                                                                          help="Launch Hammerhead in console-only mode")
    parser.add_argument('-d', '--domain',   action="store", type=str, required=True,                         choices=['axisym', '2D'],                                        help="Type of HFM data domain")
    parser.add_argument(      '--noHFM',    action="store_true",      required=False, default=False,                                                                          help="Disable HFM database population process")
    parser.add_argument(      '--noTensor', action="store_true",      required=False, default=False,                                                                          help="Disable the tensor update process from HFM database")
    parser.add_argument('-n', '--nProc',    action="store", type=int, required=False, default=cpu_count(),                                                                    help="Number of concurrent processes")
    parser.add_argument('-o', '--openfoam', action="store", type=str, required=False, default="openfoam2212",                                                                 help="OpenFOAM executable name or path")
    parser.add_argument('-p', '--plot',     action="store", type=str, required=False, default=[], nargs="*", choices=['kriging', 'lumped', 'modal', 'rbf', 'spatial', 'all'], help="Plot ML surface vs HF sparse data")
    parser.add_argument('-s', '--search',   action="store", type=str, required=False, default=[], nargs="*", choices=['kriging', 'lumped', 'modal', 'rbf', 'spatial', 'all'], help="Line search for optimal shape parameters")
    parser.add_argument('-t', '--train',    action="store", type=str, required=False, default=[], nargs="*", choices=['kriging', 'lumped', 'modal', 'rbf', 'spatial', 'all'], help="Train ML algorithms")
    return parser

def parse_args(parser: ArgumentParser) -> Namespace:
    """
    Parses arguments and verifies that they are valid
    
    Parameters
    ----------
    parser : argparse.ArgumentParser argument parser object

    Returns
    -------
    args : argparse.Namespace       namespace object of parsed args
    """
    args: Namespace = parser.parse_args()                                       # Construct namespace object full of arguments passed by user (or provided by default values)
        
    if args.nProc < 1:                                                          # Check that the number of requested concurrent processes is a positive integer
        raise RuntimeError(f"--nProc must be greater than 1, got {args.nProc}")
        
    if not args.noHFM:                                                          # If HFM database population process is enabled, check that a compatible OpenFOAM version exists in current environment
        try:                                                                    # Get available OpenFOAM version as a string (this will fail if OpenFOAM doesn't exist on system)
            openfoam_version: str = check_output(f"{args.openfoam} --version", shell=True).decode().strip()
            if openfoam_version not in ["2212", "2106"]:                        # Compare reported OpenFOAM version against lsit of supported versions
                print(f"An unsupported version (v{openfoam_version}) of openFOAM has been detected. This may cause errors if sampling file format differs! (Supported versions: v2212, v2106)") 
        except CalledProcessError:                                              # If no OpenFOAM version is detected, an exception is raised
            raise RuntimeError(f"{args.openfoam} not found in current environment."
                                "  1) if you do not wish to run any simulations, do not specify --domain"
                                "  2) if you do wish run simulations, openFOAM versions 2212 or 2106 must be available on this system"
                                "  3) if a supported version of openFOAM exists on this system but not in this environment, specify its path via --openfoam")
    
    for arg in ['plot', 'search', 'train']:                                     # For each of the --plot, --search and --train arguments ...
            if "all" in getattr(args, arg):                                     # ... if "all" has been specified ...
                setattr(args, arg, ['kriging', 'lumped', 'modal', 'rbf', 'spatial'])  # ... replace "all" with list of all possible inputs (so these arguments are ALWAYS a list)
     
    if args.noTensor:                                                           # If tensor update process is disabled ...
        for arg in ["plot", "train"]:                                           # ... iterate over --plot and --train arguments (both of which require tensor data if enabled)
            if len(getattr(args, arg)) > 0:                                     # ... if the length of --plot or --train is greater than zero, they will run, therefore warn the user
                print(f"Specified --noTensor, but also --{arg} which requires tensor data, ensure this data is locally available")

    for arg in ['plot', 'search']:                                              # For each of the --plot and --search arguments
        for algorithm in getattr(args, arg):                                    # ... and for each of the possible algorithms that the argument takes
            if algorithm not in args.train:                                     # ... if that algorithm is not specified in --train but is specified in --plot or --search, warn user 
                print(f"Specified --{arg} for {algorithm} but not --train, ensure training data is locally available")
    
    return args 
    
###############################################################################
    
if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    args: Namespace = parse_args(setup_argparse())                              # Parse and validate arguments
    if not args.console:                                                        # If console-only mode not requested, GUI mode is used ...
        from hammerheadQt import launch_gui                                     # ... load necessary GUI resources into memory
        launch_gui(args)                                                        # ... launch HammerHead GUI and autofill provided arguments
    else:
        if not args.noHFM:                                                      # If HFM database population process is enabled ...
            dbPopulate(args.domain, args.nProc, args.openfoam)                  # Utilities -> Populate the database with HFM data generated by OpenFOAM
        else:
            print("--noHFM specified, HFM database population process will be skipped")
            
        if not args.noTensor:                                                   # If tensor update process is enabled ...
            dbTensorUpdate(args.domain, args.nProc)                             # Utilities -> Pre-process data from HFM database generated by OpenFOAM, used by --train and --plot
        else:
            print("--noTensor specified, tensor update process from HFM database will be skipped")
        
        mlTrain(args.domain, args.train, args.nProc)                            # Utilities -> Train the specified ML/RBF models
        mlPlot(args.domain, args.plot, args.nProc)                              # Utilities -> Plot a 3D surface using the predictions of the chosen ML/RBF models
        mlOptimalSearch(args.domain, args.search, args.nProc)                   # Utilities -> Predict the optimal shape with the chosen ML/RBF models using a torch optimiser
