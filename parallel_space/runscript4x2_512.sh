#!/bin/bash
#SBATCH --ntasks 8                #Number of processes
#SBATCH --nodes 1                  #Number of nodes
#SBATCH -t 60                 #Runtime in minutes
#SBATCH -p general            #Partition to submit to

#SBATCH --mem-per-cpu=200     #Memory per cpu in MB (see also --mem)
#SBATCH -o algo512_4x2procs.out    #File to which standard out will be written
#SBATCH -e algo512_4x2procs.err    #File to which standard err will be written

# load modules
module load centos6/mpi4py-1.3.1_python-2.7.3_openmpi-1.6.4_gcc-4.8.0
module load centos6/numpy-1.7.1_python-2.7.3

mpirun -n 8 python parallel_wave512.py 4 2
