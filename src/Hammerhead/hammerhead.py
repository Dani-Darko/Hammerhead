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
from gpytorch import kernels                                                    # Other -> All available gpytorch kernels
from inspect import signature                                                   # Other -> Get signature (params, returns) of functions
from subprocess import CalledProcessError, check_output                         # Other -> Check openFOAM executable exists in current environment

import multiprocessing                                                          # Other -> Get number logical CPU cores, set context/start methods
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
    parser.add_argument('-c', '--console',     action="store_true",        required=False, default=False,                                            help="Launch Hammerhead in console-only mode")
    parser.add_argument('-d', '--domain',      action="store", type=str,   required=False, default=None,          choices=['axisym', '2D'],          help="Type of HFM data domain")
    parser.add_argument(      '--noHFM',       action="store_true",        required=False, default=False,                                            help="Disable HFM database population process")
    parser.add_argument(      '--noTensor',    action="store_true",        required=False, default=False,                                            help="Disable the tensor update process from HFM database")
    parser.add_argument('-n', '--nProc',       action="store", type=int,   required=False, default=multiprocessing.cpu_count(),                      help="Number of concurrent processes")
    parser.add_argument('-o', '--openfoam',    action="store", type=str,   required=False, default="openfoam2212",                                   help="OpenFOAM executable name or path")
    parser.add_argument('-p', '--plot',        action="store", type=str,   required=False, default=[], nargs="*", choices=availableModels + ["all"], help="Plot ML surface vs HF sparse data")
    parser.add_argument('-s', '--search',      action="store", type=str,   required=False, default=[], nargs="*", choices=availableModels + ["all"], help="Line search for optimal shape parameters")
    parser.add_argument('-t', '--train',       action="store", type=str,   required=False, default=[], nargs="*", choices=availableModels + ["all"], help="Train ML algorithms")
    
    # Optional arguments to override hfmParams.yaml
    parser.add_argument(       '--Re_min',     action="store", type=int,   required=False, default=None,                                             help="Minimum Reynolds number; takes precedence if specified")
    parser.add_argument(       '--Re_max',     action="store", type=int,   required=False, default=None,                                             help="Maximum Reynolds number; takes precedence if specified")
    parser.add_argument(       '--Re_num',     action="store", type=int,   required=False, default=None,                                             help="Number of Reynolds numbers between Re_min and Re_max (inclusive); takes precedence if specified")
    parser.add_argument(       '--A1_min',     action="store", type=float, required=False, default=None,                                             help="Minimum main amplitude [m] of wall shape function; takes precedence if specified")
    parser.add_argument(       '--A1_max',     action="store", type=float, required=False, default=None,                                             help="Maximum main amplitude [m] of wall shape function; takes precedence if specified")
    parser.add_argument(       '--A1_num',     action="store", type=int,   required=False, default=None,                                             help="Number of amplitudes between A1_min and A1_max (inclusive); takes precedence if specified")
    parser.add_argument(       '--k1_min',     action="store", type=float, required=False, default=None,                                             help="Minimum main wavenumber of wall shape function; takes precedence if specified")
    parser.add_argument(       '--k1_max',     action="store", type=float, required=False, default=None,                                             help="Maximum main wavenumber of wall shape function; takes precedence if specified")
    parser.add_argument(       '--k1_num',     action="store", type=int,   required=False, default=None,                                             help="Number of wavenumbers between k1_min and k1_max (inclusive); takes precedence if specified")
    parser.add_argument(       '--A2_min',     action="store", type=float, required=False, default=None,                                             help="Minimum secondary amplitude [m] of wall shape function; takes precedence if specified")
    parser.add_argument(       '--A2_max',     action="store", type=float, required=False, default=None,                                             help="Maximum secondary amplitude [m] of wall shape function; takes precedence if specified")
    parser.add_argument(       '--A2_num',     action="store", type=int,   required=False, default=None,                                             help="Number of amplitudes between A2_min and A2_max (inclusive); takes precedence if specified")
    parser.add_argument(       '--k2_min',     action="store", type=float, required=False, default=None,                                             help="Minimum secondary wavenumber of wall shape function; takes precedence if specified")
    parser.add_argument(       '--k2_max',     action="store", type=float, required=False, default=None,                                             help="Maximum secondary wavenumber of wall shape function; takes precedence if specified")
    parser.add_argument(       '--k2_num',     action="store", type=int,   required=False, default=None,                                             help="Number of wavenumbers between k2_min and k2_max (inclusive); takes precedence if specified")
    
    # Optional arguments to override trainingParams.yaml
    parser.add_argument(       '--layers',     action="store", type=int,   required=False, default=[], nargs="*",                                    help="Number of hidden layers for NN architecture; takes precedence if specified")
    parser.add_argument(       '--modes',      action="store", type=int,   required=False, default=None,                                             help="Number of PCA modes used for training; takes precedence if specified")
    parser.add_argument(       '--neurons',    action="store", type=int,   required=False, default=[], nargs="*",                                    help="Number of neurons per layer for NN architecture; takes precedence if specified")
    parser.add_argument(       '--kernelsGP',  action="store", type=str,   required=False, default=[], nargs="*", choices=availableKernelsGP,        help="Types of kernel to use for GP, see gpytorch.kernels documentation; takes precedence if specified")
    parser.add_argument(       '--kernelsRBF', action="store", type=str,   required=False, default=[], nargs="*", choices=availableKernelsRBF,       help="Types of kernel to use for RBF, see scipy.interpolate.RBFInterpolator documentation; takes precedence if specified")
    parser.add_argument(       '--samples',    action="store", type=int,   required=False, default=None,                                             help="Number of times training is performed; takes precedence if specified")
    parser.add_argument(       '--valSplit',   action="store", type=float, required=False, default=[], nargs="*",                                    help="Ratio of data used for validation; takes precedence if specified")

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
    
    if args.console and args.domain is None:                                    # Ensure that if running in console mode, the --domain is specified (in GUI mode, enforced by GUI layout)
        raise RuntimeError(f"--domain must be specified (one of: [2D, axisym])")
        
    if args.nProc < 1:                                                          # Check that the number of requested concurrent processes is a positive integer
        raise RuntimeError(f"--nProc must be greater than 1, got {args.nProc}")
        
    def _OFVersionWarning(openfoam_version):
        if openfoam_version not in ["2212", "2106"]:                            # Compare reported OpenFOAM version against list of supported versions
            print(f"An unsupported version (v{openfoam_version}) of openFOAM has been detected. This may cause errors if sampling file format differs! (Supported versions: v2212, v2106)") 
            
    if not args.noHFM:                                                          # If HFM database population process is enabled, check that a compatible OpenFOAM version exists in current environment
        try:                                                                    # Get available OpenFOAM version as a string (this will fail if OpenFOAM doesn't exist on system)
            _OFVersionWarning(check_output(f"{args.openfoam} --version", shell=True).decode().strip())
        except CalledProcessError:                                              # If no OpenFOAM version is detected, maybe we are in already in an OpenFOAM environment
            try:                                                                # If we are in an OpenFOAM environment, we can detect OpenFOAM version from an environment variable
                _OFVersionWarning(os.environ["WM_PROJECT_VERSION"])             # If this variable does not exist, we get a KeyError, and therefore the necessary OpenFOAM is inaccessible
                args.openfoam: str = ""                                         # If we got this far, we are in an existing OpenFOAM environment, and therefore the executable/prefix is unnecessary
            except KeyError:                                                    # Rise a runtime exception, cannot continue without OpenFOAM if database population is requested
                raise RuntimeError(f"{args.openfoam} not found in current environment."
                                    "  1) if you do not wish to run any simulations, specify --noHFM"
                                    "  2) if you do wish run simulations, openFOAM versions 2212 or 2106 must be available on this system"
                                    "  3) if a supported version of openFOAM exists on this system but not in this environment, specify its path via --openfoam")
    
    for arg in ['plot', 'search', 'train']:                                     # For each of the --plot, --search and --train arguments ...
            if "all" in getattr(args, arg):                                     # ... if "all" has been specified ...
                setattr(args, arg, available_models)                            # ... replace "all" with list of all possible inputs (so these arguments are ALWAYS a list)
     
    if args.noTensor:                                                           # If tensor update process is disabled ...
        for arg in ["plot", "train"]:                                           # ... iterate over --plot and --train arguments (both of which require tensor data if enabled)
            if len(getattr(args, arg)) > 0:                                     # ... if the length of --plot or --train is greater than zero, they will run, therefore warn the user
                print(f"Specified --noTensor, but also --{arg} which requires tensor data, ensure this data is locally available")

    for arg in ['plot', 'search']:                                              # For each of the --plot and --search arguments
        for algorithm in getattr(args, arg):                                    # ... and for each of the possible algorithms that the argument takes
            if algorithm not in args.train:                                     # ... if that algorithm is not specified in --train but is specified in --plot or --search, warn user 
                print(f"Specified --{arg} for {algorithm} but not --train, ensure training data is locally available")
                
    # ** Checks for optional hfmParams.yaml arguments **
    
    # Enforce minimum values
    for arg in ["Re_min", "Re_max", "A1_min", "A1_max", "k1_min", "k1_max", "A2_min", "A2_max", "k2_min", "k2_max"]:
        if getattr(args, arg) is not None and getattr(args, arg) < 0:
            raise RuntimeError(f"--{arg} must be positive")
    for arg in ["Re_num", "A1_num", "k1_num", "A2_num", "k2_num"]:
        if getattr(args, arg) is not None and getattr(args, arg) < 1:
            raise RuntimeError(f"--{arg} must not be smaller than 1")
            
    # Enforce maximum values
    if sum([getattr(args, arg) for arg in ["A1_max", "A2_max"] if getattr(args, arg) is not None] + [0]) >= 0.06:
        raise RuntimeError("Sum of --A1_max and --A2_max must be lower than 0.06")
        
    # Maximum warnings
    if args.Re_max is not None and args.Re_max > 9000:
        print("--Re_max should not exceed 9000 (the limit of mesh verification)")
    for arg in ["A1_min", "A1_max", "A2_min", "A2_max"]:
         if getattr(args, arg) is not None and getattr(args, arg) > 0.01:
            print(f"Warning: --{arg} should not exceed 0.01 (the limit of mesh verification)")
    for arg in ["k1_min", "k1_max", "k2_min", "k2_max"]:
        if getattr(args, arg) is not None and getattr(args, arg) > 64:
            print(f"Warning: --{arg} should not exceed 64 (the limit of mesh verification)")
    
    # Other warnings
    if any([getattr(args, arg) is not None is not None for arg in ["k1_min", "k1_max", "k1_num", "A2_min", "A2_max", "A2_num", "k2_min", "k2_max", "k2_num"]]):
        print(f"Warning: ensure that kn_min, kn_max and kn_num are specified such that wavenumbers for all cases are multiples of 4 (otherwise mesh generation behaviour is unspecified)")
    
    # Compile all hfmParams arguments into a single dictionary
    args.hfmParams = {arg: getattr(args, arg) for arg in ["Re_min", "Re_max", "Re_num", "A1_min", "A1_max", "A1_num", "k1_min", "k1_max", "k1_num", "A2_min", "A2_max", "A2_num", "k2_min", "k2_max", "k2_num"] if getattr(args, arg) is not None}
    
    # ** Checks for optional trainingParams.yaml arguments **
    
    # Enforce minimum values (single number)
    for arg in ["modes", "samples"]:
        if getattr(args, arg) is not None and getattr(args, arg) < 1:
            raise RuntimeError(f"--{arg} must not be less than 1")
            
    # Enforce minimum values (list of values)
    for arg in ["layers", "neurons"]:
        if len(getattr(args, arg)) != 0 and any([val < 1 for val in getattr(args, arg)]):
            raise RuntimeError(f"Every entry in --{arg} must not be less than 1")
            
    # Enforce bounds
    if len(args.valSplit) != 0 and any([(val <= 0 or val >= 1) for val in args.valSplit]):
        raise RuntimeError(f"Every entry in --valSplit must be greater than 0 but less than 1")
        
    # Compile all trainingParams arguments into a single dictionary
    args.trainingParams = {arg: getattr(args, arg) for arg in ["modes", "samples"] if getattr(args, arg) is not None}
    args.trainingParams |= {arg: getattr(args, arg) for arg in ["layers", "neurons", "kernelsGP", "kernelsRBF", "valSplit"] if len(getattr(args, arg)) != 0}

    return args 
    
###############################################################################
    
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')                                   # All sub-processes will "spawn" rather than "fork" (needed by PyTorch)
    os.chdir(os.path.abspath(os.path.dirname(__file__)))                        # Ensure script's CWD is always where the main hammerhead.py script is located
    
    # Available surrogate model options = prefix+suffix (example: MNN; prefix M = modal; suffix N = neural network)
    modelPrefixes = ["L", "M", "S"]                                            # Prefixes represent model output types: "L" = lumped; "M" = modal; "S" = spatial
    modelSuffixes = ["GP", "NN", "RBF"]                                        # Suffixes represent surrogate model algorithms: "GP" = Gaussian process; "NN" = neural network; "RBF" = radial basis function interpolation
    availableModels = [prefix+suffix for prefix in modelPrefixes for suffix in modelSuffixes]  # Construct list of surrogate models from prefix+suffix combinations
    
    # Available GP gpytorch kernels
    availableKernelsGP: list[str] = [                                           # Compile list of all available kernels (which have no required arguments)
        kernel for kernel in dir(kernels)                                       # Iterate over all kernels in gpytorch.kernels
        if kernel[0].isupper()                                                  # As kernels are classes, ignore all items that do not start with an uppercase letter
        and not any([parameter.default is parameter.empty                       # Ensure that all parameter defaults are not empty (if any are, then those are required arguments, and use that kernel)
                     for parameter_name, parameter                              # Unpacking named dictionary of parameters for this kernel
                     in signature(getattr(kernels, kernel)).parameters.items()  # Get a ordered dictionary of kernel parameters, where key is kernel name (str) and parameter is an inspect.Parameter object
                     if parameter_name != "kwargs"])]                           # Ignore parameter "kwargs", this represents the keyword arguments
    # Available RBF scipy.interpolate.RBFInterpolator kernels
    availableKernelsRBF: list[str] = ["linear", "thin_plate_spline", "cubic", "quintic", "multiquadric", "inverse_multiquadric", "inverse_quadratic", "gaussian"]
    
    args: Namespace = parse_args(setup_argparse())                              # Parse and validate arguments
    if not args.console:                                                        # If console-only mode not requested, GUI mode is used ...
        from hammerheadQt import launch_gui                                     # ... load necessary GUI resources into memory
        launch_gui(args)                                                        # ... launch HammerHead GUI and autofill provided arguments
    else:
        if not args.noHFM:                                                      # If HFM database population process is enabled ...
            dbPopulate(args.domain, args.nProc, args.openfoam, args.hfmParams)  # Utilities -> Populate the database with HFM data generated by OpenFOAM
        else:
            print("--noHFM specified, HFM database population process will be skipped")
            
        if not args.noTensor:                                                   # If tensor update process is enabled ...
            dbTensorUpdate(args.domain, args.nProc, args.trainingParams)        # Utilities -> Pre-process data from HFM database generated by OpenFOAM, used by --train and --plot
        else:
            print("--noTensor specified, tensor update process from HFM database will be skipped")
        
        mlTrain(args.domain, args.train, args.nProc, args.trainingParams)       # Utilities -> Train the specified ML/RBF models
        mlOptimalSearch(args.domain, args.search, args.nProc, args.trainingParams)  # Utilities -> Predict the optimal shape with the chosen ML/RBF models using a torch optimiser
        mlPlot(args.domain, args.plot, args.nProc, args.trainingParams)         # Utilities -> Plot a 3D surface using the predictions of the chosen ML/RBF models
