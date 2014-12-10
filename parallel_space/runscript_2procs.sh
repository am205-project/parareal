#!/bin/bash
#SBATCH --ntasks 2                #Number of processes
#SBATCH --nodes 1                  #Number of nodes
#SBATCH -t 60                 #Runtime in minutes
#SBATCH -p general            #Partition to submit to

#SBATCH --mem-per-cpu=200     #Memory per cpu in MB (see also --mem)
#SBATCH -o algo_2procs.out    #File to which standard out will be written
#SBATCH -e algo_2procs.err    #File to which standard err will be written

# load modules
module load centos6/mpi4py-1.3.1_python-2.7.3_openmpi-1.6.4_gcc-4.8.0
module load centos6/numpy-1.7.1_python-2.7.3
module load centos6/Anaconda-1.8.0

mpirun -n 4 python P2A.py 2 2
