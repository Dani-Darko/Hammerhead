##############################################################################################
#                                                                                            #
#       HAMMERHEAD - Harmonic MOO Model Expedient for the Reduction                          #
#                    Hydraulic-loss/Heat-transfer Enhancement Asset Design                   #
#       Author: Daniela Minerva Segura Galeana                                               #
#       Supervised by: Antonio J. Gil                                                        #
#                                                                                            #
##############################################################################################
#                                                                                            #
#       Script containing functions for preparing OpenFOAM directories and executing         #
#           OpenFOAM scripts.                                                                #
#                                                                                            #
##############################################################################################

# IMPORTS: HAMMERHEAD files ############################################################################################################### 

# IMPORTS: Others #########################################################################################################################

import numpy as np                                                              # Others -> Mathematical operations
import shutil                                                                   # Others -> File/directory manipulation
import subprocess                                                               # Others -> External script execution

from pathlib import Path                                                        # Other -> Path manipulation

#############################################################################################################################################

def runOpenFOAM(casePath: Path, openfoam: str) -> bool:
    """
    Run OpenFOAM chtMultiRegionSimpleFoam in the requested case directory
    
    Parameters
    ----------
    casePath : pathlib.Path             Path of case directory where OpenFOAM will be executed
    openfoam : str                      Name (or path) of compatible OpenFOAM executable

    Returns
    -------
    success : bool                      True if OpenFOAM execution concluded without errors, otherwise False
    """
    failure = bool(subprocess.run(                                              # Run the following command, and convert its return code into a boolean (0 = False = success)
        f"chmod +x {casePath / 'Allrun'} && {openfoam} {casePath / 'Allrun'}",  # Make "Allrun" script executable by OpenFOAM, then pass it to the selected OpenFOAM binary for execution
        shell=True,                                                             # Execute inside the system shell
        stdout=subprocess.DEVNULL,                                              # Redirect standard output to devnull (silence all output)
        stderr=subprocess.DEVNULL).returncode)                                  # Also silence all output in standard error, and capture the exit code
    return not failure                                                          # Reutrn True if success (boolean inverse of failure)

def runMesh(casePath: Path, openfoam: str) -> None:
    """
    Run OpenFOAM blockMesh and stitchMesh in the requested case directory
    
    Parameters
    ----------
    casePath : pathlib.Path             Path of case directory where OpenFOAM will be executed
    openfoam : str                      Name (or path) of compatible OpenFOAM executable

    Returns
    -------
    None
    """
    subprocess.run(                                                             # Run the following command in an external shell
        f"chmod +x {casePath / 'runMesh'} && {openfoam} {casePath / 'runMesh'}",  # Make "runMesh" script executable by OpenFOAM, then pass it to the selected OpenFOAM binary for execution
        shell=True,                                                             # Execute inside the system shell
        stdout=subprocess.DEVNULL,                                              # Redirect standard output to devnull (silence all output)
        stderr=subprocess.DEVNULL)                                              # Also silence all output in standard error
    
    shutil.move(casePath / "cellZones", casePath / "1/polyMesh/cellZones")      # Move the "./cellZones" file containing topology set information to the case subdirectory "./1/polyMesh/"
    shutil.rmtree(casePath / "constant/polyMesh/")                              # Delete the case subdirectory "./constant/polyMesh/" (mesh information is updated by stitching process in "./1/polyMesh/", rendering this directory useless)
    shutil.move(casePath / "1/polyMesh/", casePath / "constant/")               # Move updated mesh from "./1/polyMesh/" to "./constant/polyMesh/" (chtMultiRegionSimpleFoam reads this format)
    shutil.rmtree(casePath / "1/")                                              # Delete everything remaining in "./1/" (start OpenFOAM iterations from "./0/", not from "./1/")
    
def populateBlockMeshEdges2D(casePath: Path, xArray: np.ndarray, yArray: np.ndarray) -> None:
    """
    Write the current shape parameter coordinates in the mesh generation
        file for the 2D domain type
    
    Parameters
    ----------
    casePath : pathlib.Path             Path of case directory where OpenFOAM will be executed
    xArray : array_like                 Array of x-coordinates of current surface shape
    yArray : array_like                 Array of y-coordinates of current surface shape

    Returns
    -------
    None
    """
    shutil.copytree("./resources/base2DFiles", casePath, dirs_exist_ok=True)    # Copy OpenFOAM simulation files from 2D base directory to the current case directory
    edgeCoors0 = "".join([f"({x} {y} 0.0)\n" for x, y in zip(xArray, yArray)])  # Create string of coordinates at z=0.0 for blockMesh edge morphing
    edgeCoors1 = "".join([f"({x} {y} 0.1)\n" for x, y in zip(xArray, yArray)])  # Create string of coordinates at z=0.1 for blockMesh edge morphing, OpenFOAM needs 3D coordinates even for 2D domains
    edgesList = ['polyLine 3 2\n(\n', edgeCoors0,                               # These are the edge numbers formatted as expected by OpenFOAM, you can find these in the blockMeshDict file
                 ')\npolyLine 7 6\n(\n', edgeCoors1,                            # ... this format is needed to deform the mesh into the new shape by OpenFOAM blockMesh
                 ')\npolyLine 8 9\n(\n', edgeCoors0,
                 ')\npolyLine 12 13\n(\n', edgeCoors1,
                 ')\n']                                
    with open(casePath / "system/blockMeshDict", 'r') as meshFile:              # Open blockMeshDict file for reading ...
        meshFileList = [line for line in meshFile]                              # ... read the contents of the file into a list where each item is a single line
    with open(casePath / "system/blockMeshDict", 'w') as meshFile:              # Open blockMeshDict file for writing (overwriting existing content) ...
        meshFile.write("".join(meshFileList[:66] + edgesList + meshFileList[66:]))  # Insert previously constructed edge strings into the expected position of meshFileList, and update blockMeshDict with the new contents 

def populateBlockMeshEdgesAxisym(casePath: Path, xArray: np.ndarray, yArray: np.ndarray) -> None:
    """
    Write the current shape parameter coordinates in the mesh generation
        file for the axisymmetric domain type
    
    Parameters
    ----------
    casePath : pathlib.Path             Path of case directory where OpenFOAM will be executed
    xArray : array_like                 Array of x-coordinates of current surface shape
    yArray : array_like                 Array of y-coordinates of current surface shape

    Returns
    -------
    None
    """
    shutil.copytree("./resources/baseAxisymFiles", casePath, dirs_exist_ok=True)  # Copy OpenFOAM simulation files from Axisym base directory to the current case directory
    # In the context of our OpenFOAM model, the length of the cylindrical pipe corresponds to the x-axis (xArray coordinates), and its circular cross-section is defined by three points in (y, z); one point coincides with the x-axis ...
    # ... and the two other points lie at a fixed positive z-value, and are mirrored in the z-axis (therefore have positive and negative y-values); while the length can still be defined by xArray, the yArray and zArray have to be computed
    yArrayOpposite = yArray * np.sin(10 * np.pi / 180)                          # Opposite side of a right-angled triangle, the 10 representing the angle between the hypotenuse and the z-axis
    zArrayAdjacent = yArray * np.cos(10 * np.pi / 180)                          # Adjacent side of a right-angled triangle
    edgeCoors0 = "".join([f"({x} {y} {z})\n" for x, y, z in zip(xArray, -yArrayOpposite, zArrayAdjacent)])  # Create string of coordinates for left-side of the circular cross-section for blockMesh edge morphing
    edgeCoors1 = "".join([f"({x} {y} {z})\n" for x, y, z in zip(xArray,  yArrayOpposite, zArrayAdjacent)])  # Create string of coordinates for right-side of the circular cross-section for blockMesh edge morphing
    edgesList = ['polyLine 4 5\n(\n', edgeCoors0,                              # These are the edge numbers formatted as expected by OpenFOAM, you can find these in the blockMeshDict file
                 ')\npolyLine 7 6\n(\n', edgeCoors1,                           # ... this format is needed to deform the mesh into the new shape by OpenFOAM blockMesh
                 ')\npolyLine 8 9\n(\n', edgeCoors0,
                 ')\npolyLine 11 10\n(\n', edgeCoors1,
                 ')\n']
    with open(casePath / "system/blockMeshDict", 'r') as meshFile:              # Open blockMeshDict file for reading ...
        meshFileList = [line for line in meshFile]                              # ... read the contents of the file into a list where each item is a single line
    with open(casePath / "system/blockMeshDict", 'w') as meshFile:              # Open blockMeshDict file for writing (overwriting existing content) ...
        meshFile.write("".join(meshFileList[:102] + edgesList + meshFileList[102:]))  # Insert previously constructed edge strings into the expected position of meshFileList, and update blockMeshDict with the new contents 

def populateBlockMeshEdges(domainType: str, casePath: Path, xArray: np.ndarray, yArray: np.ndarray) -> None:
    """
    Dispatcher for populateBlockMeshEdges* functions based on specified
        domain type
    
    Parameters
    ----------
    domainType : str                    String specifing the domain type (either "2D" or "axisym")
    casePath : pathlib.Path             Path of case directory where OpenFOAM will be executed
    xArray : array_like                 Array of x-coordinates of current surface shape
    yArray : array_like                 Array of y-coordinates of current surface shape

    Returns
    -------
    None
    """
    (populateBlockMeshEdges2D if domainType == "2D" else populateBlockMeshEdgesAxisym)(casePath, xArray, yArray)
    
        
def updateVelocity(casePath: Path, hfmParams: dict[str, float], Re: int, C_mu: float = 0.09) -> None:
    """
    Update velocity data in the current case directory based on the specified
        Reynolds number
    
    Parameters
    ----------
    casePath : pathlib.Path             Path of case directory where OpenFOAM will be executed
    hfmParams : dict                    Dictionary of HFM parameters loaded from ./resources/hfmParams.yaml
    Re : int                            Reynolds number
    C_mu : float                        Dynamic viscosity of fluid

    Returns
    -------
    None
    """
    u = np.round(Re * hfmParams['mu'] / hfmParams['L'])                         # Compute the inlet velocity from the Reynolds number and domain dimensions
    k = (3/2) * ((u * hfmParams['I']) ** 2)                                     # Compute turbulent kinetic energy (takes effect when Re > 2000)
    omega = (k ** 0.5) / ((C_mu ** 0.25) * hfmParams['r'])                      # Compute specific turbulent dissipation rate (takes effect when Re > 2000)
    epsilon = ((k ** 1.5) * (C_mu ** 0.75)) / hfmParams['r']                    # Compute turbulent dissipation rate (takes effect when Re > 2000)
    turbulenceInfo = [[f"    internalField   uniform {epsilon};\n"],            # Transform the turbulence variables into strings that will be inserted into the OpenFOAM changeDictionaryDict file
                      [f"    internalField   uniform {k};\n"],
                      [f"    internalField   uniform {omega};\n"]]
    uInfo = [[f"     internalField   uniform ({u} 0 0);\n"],                    # Prepare strings for parabolic velocity profile, to be inserted into the OpenFOAM changeDictionaryDict file
             [f"    			scalar Umax = {u}, r = {hfmParams['r']};\n"]]
    
    with open(casePath / "system/Helium/changeDictionaryDict", "r") as uFile:   # Open changeDictionaryDict file for reading ...
        uFileList = [line for line in uFile]                                    # ... read the contents of the file into a list where each item is a single line
    with open(casePath / "system/Helium/changeDictionaryDict", "w") as uFile:   # Open changeDictionaryDict file (where fluid variables are stored) for writing, overwriting it with new data
        if hfmParams["domainType"] == "2D":                                     # ... insert prepared strings where appropriate based on specified domain type
            uFile.write("".join(uFileList[:23]
                                + uInfo[0] + uFileList[23:35]
                                + uInfo[1] + uFileList[35:111]
                                + turbulenceInfo[0] + uFileList[111:141]
                                + turbulenceInfo[1] + uFileList[141:172]
                                + turbulenceInfo[2] + uFileList[172:]))
        else:
            uFile.write("".join(uFileList[:23]
                                + uInfo[0] + uFileList[23:35]
                                + uInfo[1] + uFileList[35:102]
                                + turbulenceInfo[0] + uFileList[102:134]
                                + turbulenceInfo[1] + uFileList[134:168]
                                + turbulenceInfo[2] + uFileList[168:]))
    if Re >= 2000:                                                              # If Reynolds number is greater than 2000, additional files need to be copied that enable turbulence model 
        shutil.copy(casePath / "constant/RASProperties",
                    casePath / "constant/Helium/RASProperties")
        shutil.copy(casePath / "constant/turbulenceProperties",
                    casePath / "constant/Helium/turbulenceProperties")
                    
        
###############################################################################

if __name__ == "__main__":
    raise RuntimeError("This script is not intended for execution. Instead, execute hammehead.py from the parent directory.")