#!/bin/bash --login
###
# job name
#SBATCH --job-name=PipeFlow_%a
##SBATCH --output=/dev/null
# job stderr file
#SBATCH --error=PipeFlow.err.%a
##SBATCH --error=/dev/null
# maximum job time in D-HH:MM
#SBATCH --time=0-00:10
# maximum memory megabytes
##SBATCH --mem-per-cpu=8G
# run a two tasks
#SBATCH --ntasks=40
# run the tasks across two nodes; i.e. one per node
#SBATCH --nodes=1
# specify our current project
#SBATCH --account=scw1706
#SBATCH --mail-user=2115589@swansea.ac.uk
#SBATCH --mail-type=END
###


module load compiler/gnu/12 mpi/openmpi/4.1.5 boost/1.82.0 fftw/3.3.10 cmake/3.14.3
source /apps/local/materials/OpenFOAM/v2212/el7/AVX512/gnu-12.1/openmpi-4.1/OpenFOAM-v2212/etc/bashrc
./Preproc
mpirun -np 40 chtMultiRegionSimpleFoam -parallel
./Postproc
#srun --nodes 1 --ntasks 4 ./run
