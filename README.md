<h1 align="center">
    <br>
    <img src="src/Hammerhead/assets/images/hammerheadMascot.png" alt="Hammerhead Mascot" width="240">
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
    <img src="https://img.shields.io/github/license/Dani-Darko/Hammerhead" alt="GPL-3.0">
  </a>
  
  <a href="https://github.com/Dani-Darko/Hammerhead/releases/latest">
    <img src="https://img.shields.io/github/v/release/Dani-Darko/Hammerhead?include_prereleases&sort=semver"
         alt="Latest release">
  </a>
  
  <a href="https://github.com/ajulik1997/Dani-Darko/Hammerhead/latest">
    <img src="https://img.shields.io/github/release-date-pre/Dani-Darko/Hammerhead" alt="Latest release date">
  </a>
  
  <a href="https://github.com/Dani-Darko/Hammerhead/commits">
    <img src="https://img.shields.io/github/commits-since/Dani-Darko/Hammerhead/latest" alt="Commits since latest release">
  </a>
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#usage">Usage</a>
</p>

## Key Features

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

Author + Supervisor

See also the list of [contributors](...) who participated in this project.

### Acknowledgments

## How to Cite

## Versioning

This project uses [SemVer](http://semver.org/) for versioning. For the versions available, see the
[tags in this repository](). 

## License

This project is licensed under the GNU GPL-3.0 license - see the 
[LICENSE.md](https://github.com/Dani-Darko/Hammerhead/blob/master/LICENSE) file for details
