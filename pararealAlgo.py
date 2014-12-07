# Generic Parareal Implementation
# Updated 12/6/2014

from mpi4py import MPI
import numpy as np
import time

from serial_euler import forward_euler

# Parareal begin
def startParareal(deriv, init, k, tmin, tmax, numSteps, comm):
  '''Starts running k iterations of the parareal algorithm.
  In each iteration, the program runs the coarse method in serial, then fine corrections in parallel'''

  # get a coarse answer
  u = serial_coarse(deriv, init, tmin, tmax, numSteps)[:,0]

  # fine time steps
  qualityFactor = 100
  fine = numSteps * qualityFactor

  # rethink logic here!
  if k > 1:
    for iteration in xrange(k-1):

      # get a coarse answer
      u = serial_coarse(deriv, init, tmin, tmax, numSteps)[:,0]

      # pass in a coarse answer to refine and correct
      corrections = parallel_corrections(deriv, u, time, comm, p_root=0)

      # update u with the corrections
      u += corrections

  return u

def serial_coarse(deriv, init, tmin, tmax, numSteps):
  '''Runs a coarse numerical method in serial, as part of the parareal algorithm'''
  # call forward euler
  return forward_euler(deriv, init, tmin, tmax, numSteps)

def parallel_corrections(u, tmin, tmax, numSteps, comm, p_root=0):
  '''Parallel Portion of the Parareal Algorithm
  Assumes the arrays exist on process p_root and returns the result to
  process p_root for a given iteration.
  By default, p_root = process 0.'''
  rank = comm.Get_rank()
  size = comm.Get_size()

  # Save the number of tasks to a variable
  # numtasks = len(u)

  # Start and end indices of the local dot product computed by function to allow for undivisble sizes
  start = getStart(numtasks, size, rank)
  end = getStart(numtasks, size, rank + 1)

  # Compute the partial dot product
  #local_dot = serial_dot(a[start:end], b[start:end])

  # Gather the partial results to the root process
  result = comm.gather(local_dot, root=p_root)

  return result

def getStart(numtasks, size, rank):
  # Offset by rank to account for n+1 spaces
  if rank < numtasks % size:
    return (numtasks / size + 1) * rank
    
  # Offset only by max numtasks % size
  else:
    return numtasks / size * rank + numtasks % size


if __name__ == '__main__':
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  # Example specific parameters and derivatives with IC
  def deriv(t, u):
    return lam * u

  init = np.array([1.])

  tmin = 0.
  tmax = 1.
  numSteps = 100

  init = np.array([1.])
  ###

  # Communciation barrier and timing function of MPI
  comm.barrier()
  p_start = MPI.Wtime()

  # Use parareal algorithm
  p_result = startParareal(deriv, init, tmin, tmax, step, comm)

  # Communciation barrier and ending time with MPI timing function
  comm.barrier()
  p_stop = MPI.Wtime() 

  # Check and output results on process 0
  if rank == 0:

    # compute serial result
    s_start = time.time()
    s_result = forward_euler(deriv, init, 0, 1, 0.01)[:,0]
    s_stop = time.time()

    # compute exact result
    e_result = np.exp(np.linspace(tmin, tmax, len(res)))

    # compute errors
    s_error = abs(s_result - e_result) / abs(e_result)
    p_error = abs(p_result - e_result) / abs(e_result)

    # print timers
    print "Serial Time:   %f secs" % (s_stop - s_start)
    print "Parallel Time: %f secs" % (p_stop - p_start)

    # print results
    print "Serial Result   = %f" % s_result
    print "Parallel Result = %f" % p_result
    print "Exact           = %f" % e_result

    # print errors
    print "Serial Relative Error    = %e" % s_error
    print "Parallel Relative Error  = %e" % p_error
