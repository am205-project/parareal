To run this code, submit the scripts runscriptNxN.sh using sbatch to the Odyssey Resonance cluster.

The format of each of these bash scripts is as follows:

#!/bin/bash
#SBATCH --ntasks 1                #Number of processes
#SBATCH --nodes 1                  #Number of nodes
#SBATCH -t 60                 #Runtime in minutes
#SBATCH -p general            #Partition to submit to

#SBATCH --mem-per-cpu=200     #Memory per cpu in MB (see also --mem)
#SBATCH -o algo_1x1procs.out    #File to which standard out will be written
#SBATCH -e algo_1x1procs.err    #File to which standard err will be written

# load modules
module load centos6/mpi4py-1.3.1_python-2.7.3_openmpi-1.6.4_gcc-4.8.0
module load centos6/numpy-1.7.1_python-2.7.3

mpirun -n 1 python parallel_wave.py 1 1


where ntasks should be changed, as well as the output and error files.

The main command is

mpirun -n N python parallel_wave.py px py

where N is the number of processes, and (px py) is the dimension of the grid of processors that the work will be allocated over.