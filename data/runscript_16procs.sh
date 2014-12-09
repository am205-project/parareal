#!/bin/bash
#SBATCH --ntasks 16                #Number of processes
#SBATCH --nodes 1                  #Number of nodes
#SBATCH -t 180                 #Runtime in minutes
#SBATCH -p general            #Partition to submit to

#SBATCH --mem-per-cpu=200     #Memory per cpu in MB (see also --mem)
#SBATCH -o algo_16procs.out    #File to which standard out will be written
#SBATCH -e algo_16procs.err    #File to which standard err will be written

# load modules
module load centos6/mpi4py-1.3.1_python-2.7.3_openmpi-1.6.4_gcc-4.8.0
module load centos6/numpy-1.7.1_python-2.7.3

cd ~/parareal/data

# make sure you change the -n argument for mpirun and the output files

# change the quality factor
k=2
for q in 10 50 100
do
    mpirun -n 16 python ../pararealAlgo.py --nsteps 100 --quality-factor $q --correction-level $k
done

# change the number of corrections
q=100
for k in 1 2 5 10 50 100
do
    mpirun -n 64 python ../pararealAlgo.py --nsteps 100 --quality-factor $q --correction-level $k
done
