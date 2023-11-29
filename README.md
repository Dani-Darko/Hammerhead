<h1 align="center">
    <br>
    <img src="src/Hammerhead/assets/vectors/hammerheadMascot.svg" alt="Hammerhead Mascot" width="256">
    <br>
    Hammerhead
    <br>
</h1>

<h4 align="center"> Harmonic MOO Model Expedient for the Reduction Hydraulic-loss/Heat-transfer Enhancement Asset Design</h4>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-31011">
    <img src="https://img.shields.io/badge/python-3.10-brigtgreen.svg" alt="Python 3.10">
  </a>
  
  <a href="https://github.com/Dani-Darko/Hammerhead/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/dani-darko/hammerhead" alt="GPL-3.0">
  </a>
  
  <a href="https://github.com/Dani-Darko/Hammerhead/releases/latest">
    <img src="https://img.shields.io/github/v/release/dani-darko/hammerhead?include_prereleases&sort=semver"
         alt="Latest release">
  </a>
  
  <a href="https://github.com/ajulik1997/Dani-Darko/Hammerhead/latest">
    <img src="https://img.shields.io/github/release-date-pre/dani-darko/hammerhead" alt="Latest release date">
  </a>
  
  <a href="https://github.com/Dani-Darko/Hammerhead/commits">
    <img src="https://img.shields.io/github/commits-since/dani-darko/hammerhead/latest" alt="Commits since latest release">
  </a>
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#usage">Usage</a>
</p>

## Key Features

Hammerhead is developed as a tool to study the influence of micro-structures (like the structures found on the skin of
fast sharks) on fluid flow inside a pipe using Computational Fluid Dynamics (CFD) high-fidelity models and Machine
Learning (ML) models. The tool has a workflow as follows:

-    **High fidelity model**: Hammerhead provides the means to obtain multiple verified high-fidelity models by
varying the shape of a pipe's inner surface (in contact with the fluid) to build a database of shape parametrisation
cases, used for thermal and hydraulic behaviour comparison studies. The shape parametrisation computes a surface with
a double harmonic equation using 4 parameters: 2 amplitudes (`A1` and `A2`) and 2 wavenumbers (`k1` and `k2`). The CFD
software of choice is OpenFOAM v2212, but Hammerhead can be compatible with other versions of OpenFOAM by specifying
the file structure and sampling file format.

-    **Data reduction**: Tensors are constructed from the high-fidelity model database: `spatial` (dataset size, mesh
size), `modal` (dataset size, number of modes) and `lumped` (dataset size) tensors, which are used as the output of
the ML models, as well as the shape/Reynolds number parameter tensor, which represents the features to feed the ML
models. For the modal tensors, modes are computed using Singular Value Decomposition (SVD) to reduce the
mesh-dependent data dimension keeping only the most prominent eigenvalues (the number of which can be specified by the
user). The Thermo-Hydraulic Performance (THP) for the lumped tensors is computed using the equations of advected heat
flux and dissipation rate.

-    **ML training**: Hammerhead uses the stored tensors to train several models: three Neural Networks (`NN`) based
on the output size (using spatial, modal and lumped), a Gaussian Process (`GP`) and a Radial Basis Function (`RBF`)
Interpolator (both using modal tensors only). The user can specify which models to train, and the state of each
trained model is stored.

-    **Plotting and optimal shape search**: The trained ML models are used to predict THP surfaces and compare them
with high-fidelity data. An inverse problem applied to the ML models predicts a set of parameters describing the
optimal surface shape of the pipe and Reynolds number regime (if the Reynolds number is part of the training features).
The user's trust in this prediction shall depend on the similarity of the predicted surface plots to the high-fidelity
data.

## Getting Started

### Hardware requirements

-   **Operating system**: Ubuntu 22.04 (with or without a desktop environment), but most modern Linux distributions
(including those running under WSL) that support Python 3.10 and OpenFOAM v2212 should work

-   **Storage**: 10GB recommended (+ about 100GB for datasets, stored tensors and model checkpoints)

-   **Memory**: 8GB minimum, 16GB recommended (not including base memory usage)

-   **Processor**: 64-bit with 4+ logical processors (the more the better, as Hammerhead can distribute tasks to any
number of available processors)

### Software requirements

- **Python 3.10**: Tested on Python 3.10, although newer versions should work (may work on 3.9, but will NOT
work on 3.8); the provided requirements are for Python 3.10.12

- **OpenFOAMv2212**: Optional if using an externally-built HFM database, otherwise necessary for HFM database
population (v2212 is recommended but v2106 is also supported, newer versions may work if their sampling file format
is provided in an auxiliary configuration file)

- **LaTeX**: Optional, can used for rendering text in figures and plots

### Running Hammerhead

COMING SOON! No installation is required, simply download, extract and run the portable executable. To run from
source:

Clone the repository: `git clone https://github.com/Dani-Darko/Hammerhead.git`

Move into the repository: `cd Hammerhead`

Create a new virtual environment (recommended): `python3 -m venv .venv`

Activate virtual environment: `source .venv/bin/activate`

Install all required packages: `pip install -r requirements.txt`

Execute Hammerhead: `python ./src/Hammerhead/hammerhead.py --domain 2D` (can be executed from any directory as
long as the path to `hammerhead.py` is provided)

Temporary note: while the GUI is under construction, you can use all hammerhead features by running it in console
mode using the `--console` flag.

### Compiling from source

To compile Hammerhead into a Python-independent executable, first ensure that you have a 64-bit version of
`python3.10` or higher (the current development version is 3.10.12) as well as the relevant python headers
(`python-dev`) installed. The packaging process uses `PyInstaller`, which must also be present on your system
and can be installed using `pip`. Note that this process is still experimental for multi-threaded Hammerhead.

Make the compilation script executable: `chmod +x ./compile.sh`

Run the compilation script: `./compile.sh`

## Usage

All arguments/flags, their use-case scenarios and default values will be described here.

### Hammerhead GUI

(no flags except for `--domain` are required, all optional flags are passed as defaults for the GUI)

(optionally, run Hammerhead in console-only mode using `--console`)

### Command-line arguments

## Output Files

### `./caseDatabase`

### `./mlData`

### `./plots`

## Frequently Asked Questions

## Development Notes

### Core file structure

### Development goals

TODO: Remove redundant OpenFOAM base files

TODO: GUI development

TODO: Plotting of confidence regions, residuals and optimisation process

TODO: OpenFOAM cleanup and space optimisations

TODO: GPU Training Support

TODO: Provide compiled releases, update compilation instructions

TOOD: Multithreading support for compiled code

## Authors
This software has been developed by
Daniela M Segura Galeana

Supervised by
Antonio J. Gil and Michael Edwards from Swansea University

Aleksander Dubas and Andrew Davies from UKAEA

See also the list of [contributors](...) who participated in this project.

### Acknowledgments

## How to Cite

## Versioning

This project uses [SemVer](http://semver.org/) for versioning. For the versions available, see the
[tags in this repository](). 

## License

This project is licensed under the GNU GPL-3.0 license - see the 
[LICENSE.md](https://github.com/Dani-Darko/Hammerhead/blob/master/LICENSE) file for details
