# Generic Parareal Implementation
# Updated 12/6/2014

# TODO 
# Confirm that Euler_Step function does not repeat calculations - else should memoize and look up

from mpi4py import MPI
import numpy as np
import time
import copy

from serial_euler import forward_euler
from serial_euler import forward_euler_step

# Parareal begin
def startParareal(deriv, init, k, tmin, tmax, numSteps, qualityFactor = 100, comm):
  '''Starts running k iterations of the parareal algorithm.
  In each iteration, the program runs the coarse method in serial, then fine corrections in parallel
  The result is only gathered to the root process at every iteration'''

  rank = comm.Get_rank()
  size = comm.Get_size()

  dt = (tmax-tmin) / (numSteps - 1)

  # upast and unext vectors
  upast = np.zeros(numSteps)
  unext = np.zeros(numSteps)

  # if further iterations are required
  if k > 1:
    for iteration in xrange(k-1):

      # pass in a coarse answer to refine and correct, bectorized
      corrections = parallel_corrections(deriv, upast, tmin, tmax, numSteps, qualityFactor, comm, p_root=0)

      if rank == 0:
        # get a coarse answer for the next iteration, vectorized
        unext = serial_coarse(deriv, upast, dt, tmin, tmax, numSteps)[:,0]

        # update unext for next iteration
        upast = copy.copy(unext)

        # update unext with corrections computed with upast
        # TODO check the edge case - this is correct?
        for ind in xrange(numSteps-1):
          unext[ind+1] += unext[ind] + corrections[ind]

  return unext

def serial_coarse(deriv, upast, dt, tmin, tmax, numSteps):
  '''Runs a coarse numerical method in serial, as part of the parareal algorithm.
  Will call the stepwise g_coarse function per step'''

  # initialize times
  times = np.linspace(time_min, time_max, numSteps)[1:]

  # call forward euler per step and store in vector
  for ind in xrange(numSteps):
    # get a coarse answer
    upast[ind] = forward_euler_step(deriv, times[ind], u[ind], dt)
  return upast

def parallel_corrections(upast, tmin, tmax, numSteps, qualityFactor, comm, p_root=0):
  '''Parallel Portion of the Parareal Algorithm
  Divides the work based on the unext vector, into equal parameters
  Gathers the result on process p_root for the given iteration.
  By default, p_root = process 0.'''

  rank = comm.Get_rank()
  size = comm.Get_size()

  # Save the number of tasks
  numtasks = len(numSteps)

  # values for the fine computations
  fineNumSteps = numSteps*factor
  fineDt = dt / factor

  # broadcast upast
  upast = comm.broadcast(upast, root=p_rot)

  # Start and end indices of the times to divide into with a +1 for python range offset
  start = getStart(numtasks, size, rank)
  end = getStart(numtasks, size, rank + 1)

  # initialize times
  times = np.linspace(time_min, time_max, numSteps)[1:]

  # in a loop, compute fine and then corrections one piece at a time that processor is responsible for, serially
  while (start < end):
    fineResult = forward_euler(deriv, upast[start], times[start], times[start+1], fineNumSteps)[:,0]

    # get difference for the last time step of the fine result and compare to coarse result answer for same time
    difference = fineResult[-1] - upast[start]
    # concatenate the local results
    local_unext = np.concatenate(local_unext, difference)

    # move to solve next step
    start += 1

  # Gather the partial results to the root process
  result = comm.gather(local_unext, root=p_root)

  return result

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
    s_result = forward_euler(deriv, init, tmin, tmax, numSteps)[:,0]
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
