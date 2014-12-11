parareal
========

AM 205 Final Project, Fall 2014. 

Wesley Chen, Brandon Sim, Andy Shi

How to run: 
This requires an installation of MPI to run. To run the example problems we
provided, simply run

`mpiexec -n $N python pararealAlgo.py`

For further help on the options for `pararealAlgo.py`, run

`python pararealAlgo.py -h`

The directory `parallel_space` contains the files for the parallel by space
paradigm implementation. 

The directory `data` contains data files, SLURM scripts for collecting the
data on Odyssey, and an IPython Notebook for analyzing and plotting the data. 

Our paper is in the file `paper.tex` and supporting figures are the PNG files. 

`serial_euler.py` contains the code for a serial implementation of the Euler
method, and `serial_integrate.py` contains serial implementations of other
integration schemes. 
