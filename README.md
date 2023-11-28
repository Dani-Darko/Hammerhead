# HammerHead

Work in progress!

Prerequisites:
- Python 3.10+
- Working OpenFOAM environment (for case database population)
- LaTeX (optional, for rendering figure text in LaTeX style)

Development notes:

1. Clone hammerhead
2. CD to hammerhead directory
3. Create venv "python3 -m venv .venv"
4. Activate venv "source .venv/bin/activate"
5. pip install -r requirements.txt

To compile (may not work yet when multithreaded):

6. chmod +x ./compile.sh
7. ./compile.sh

TODO: Remove redundant OpenFOAM base files
TODO: GUI development
TODO: Plotting of confience regions, residuals and optimisation process
