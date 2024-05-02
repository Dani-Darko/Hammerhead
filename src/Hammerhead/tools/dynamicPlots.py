from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pickle

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
    parser: argparse.ArgumentParser = ArgumentParser(description="Dynamic plot visualiser")
    parser.add_argument('-p', '--path', type=Path, required=True, help="Path to plot")
    return parser
    
###############################################################################

if __name__ == "__main__":
    plotPath = setup_argparse().parse_args().path
    with open(plotPath, "rb") as plotFile:
        fig = pickle.load(plotFile)
    plt.show(block=True)
