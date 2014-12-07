# Generic Parareal Implementation
# Updated 12/6/2014

from mpi4py import MPI
import numpy as np
import time
import copy

from serial_euler import forward_euler
from serial_euler import forward_euler_step

# Parareal begin
def startParareal(deriv, init, k, tmin, tmax, numSteps, comm, 
        qualityFactor = 100):
  '''Starts running k iterations of the parareal algorithm.
  In each iteration, the program runs the coarse method in serial, then fine corrections in parallel
  The result is only gathered to the root process at every iteration'''

  rank = comm.Get_rank()
  size = comm.Get_size()

  dt = (tmax-tmin) / (numSteps - 1)

  # all coarse
  upast = forward_euler(deriv, init, tmin, tmax, numSteps)
  unext = None

  # if further iterations are required
  if k > 1:
    for iteration in xrange(k-1):

      # pass in a coarse answer to refine and correct, bectorized
      corrections = parallel_corrections(deriv, upast, tmin, tmax, numSteps, qualityFactor, comm, p_root=0)

      if rank == 0:
        # get a coarse answer for the next iteration, vectorized
        unext = serial_coarse_with_correction(deriv, init, corrections, dt, tmin, tmax, numSteps)

        # update unext for next iteration
        upast = copy.copy(unext)

  return unext

def serial_coarse_with_correction(deriv, init, corrections, dt, tmin, tmax, numSteps):
  '''Runs a coarse numerical method in serial, as part of the parareal algorithm.
  Will call the stepwise g_coarse function per step'''

  fixed_corrections = np.empty((0, init.shape[0]))
  # corrections is a list
  for arr in corrections:
    fixed_corrections = np.concatenate((fixed_corrections, arr))
  unext = np.zeros((numSteps+1, init.shape[0]))

  # initialize times
  times = np.linspace(tmin, tmax, numSteps+1)

  # initial condition
  unext[0] = init

  # call forward euler per step and store in vector
  for ind in xrange(numSteps):
    # get a coarse answer
    # TODO is this upast[ind] off by one?
    unext[ind] = forward_euler_step(deriv, times[ind], unext[ind], dt)

    # update unext with corrections computed with upast
    unext[ind+1] = unext[ind] + fixed_corrections[ind]

  return unext

def parallel_corrections(deriv, upast, tmin, tmax, numSteps, qualityFactor, comm, p_root=0):
  '''Parallel Portion of the Parareal Algorithm
  Divides the work based on the unext vector, into equal parameters
  Gathers the result on process p_root for the given iteration.
  By default, p_root = process 0.'''

  rank = comm.Get_rank()
  size = comm.Get_size()

  # synonym for parallel
  numtasks = numSteps

  # values for the fine computations
  dt = float(tmax - tmin) / (numSteps - 1)
  fineNumSteps = numSteps*qualityFactor
  fineDt = dt / qualityFactor

  # broadcast upast
  upast = comm.bcast(upast, root=p_root)

  # Start and end indices of the times to divide into with a +1 for python range offset
  start = getStart(numtasks, size, rank)
  end = getStart(numtasks, size, rank + 1)

  # initialize times
  times = np.linspace(tmin, tmax, numSteps+1)

  local_differences = np.empty((0, upast.shape[1]))
  # in a loop, compute fine and then corrections one piece at a time that processor is responsible for, serially
  while (start < end):
    print("Rank: %d, start: %d" %(rank, start))
    fineResult = forward_euler(deriv, upast[start], times[start], times[start+1], fineNumSteps)

    # get difference for the last time step of the fine result and compare to coarse result answer for same time
    difference = (fineResult[-1] - upast[start+1]).reshape(1, upast.shape[1])
    # concatenate the local results
    local_differences = np.concatenate((local_differences, difference), axis=0)

    # move to solve next step
    start += 1

  # Gather the partial results to the root process
  differences = comm.gather(local_differences, root=p_root)

  return differences

def getStart(numtasks, size, rank):
  '''Returns optimally equal partioned index for the begin of the assigned array to divide'''
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
  lam = 1.
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
  # we only want column 1 so numpy doesn't complain
  p_result = startParareal(deriv, init, 2, tmin, tmax, numSteps, comm)

  # Communciation barrier and ending time with MPI timing function
  comm.barrier()
  p_stop = MPI.Wtime() 

  # Check and output results on process 0
  if rank == 0:

    p_result = p_result[:,0]
    # compute serial result
    s_start = time.time()

    # we only want column 1 so numpy doesn't complain
    s_result = forward_euler(deriv, init, tmin, tmax, numSteps)[:,0]
    s_stop = time.time()

    # compute exact result
    e_result = np.exp(np.linspace(tmin, tmax, numSteps+1))

    # compute errors
    s_error = np.abs(s_result - e_result)# / abs(e_result)
    p_error = np.abs(p_result - e_result)# / abs(e_result)

    # print timers
    print "Serial Time:   %f secs" % (s_stop - s_start)
    print "Parallel Time: %f secs" % (p_stop - p_start)

    # print results
    #print "Serial Result   = " + str(s_result)
    print "Parallel Result = " + str(p_result)
    #print "Exact           = " + str(e_result)

    # print errors
    print "Serial Relative Error    = %e" % np.max(s_error)
    print "Parallel Relative Error  = %e" % np.max(p_error)

    print "Parallel Error = " + str(p_error)
    print "Serial Error = " + str(s_error)
