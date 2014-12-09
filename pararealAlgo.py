# Generic Parareal Implementation
# Updated 12/6/2014

from mpi4py import MPI
import numpy as np
import time
import argparse
import itertools

from serial_euler import forward_euler
from serial_euler import forward_euler_step

# Parareal begin
def startParareal(deriv, init, k, tmin, tmax, numSteps, comm, 
        qualityFactor = 100):
    '''Starts running k iterations of the parareal algorithm.  In each
    iteration, the program runs the coarse method in serial, then fine
    corrections in parallel The result is only gathered to the root process at
    every iteration'''

    rank = comm.Get_rank()
    size = comm.Get_size()

    dt = (tmax-tmin) / (numSteps - 1)

    # all coarse
    upast = forward_euler(deriv, init, tmin, tmax, numSteps)
    unext = None

    # if further iterations are required
    if k > 1:
        for iteration in xrange(k-1):

            # pass in a coarse answer to refine and correct, vectorized
            corrections = parallel_corrections(deriv, upast, tmin, tmax,
                    numSteps, qualityFactor, comm, p_root=0)

            if rank == 0:
                # get a coarse answer for the next iteration, vectorized
                unext = serial_coarse_with_correction(deriv, init, corrections,
                        dt, tmin, tmax, numSteps)

                # update unext for next iteration
                upast = np.copy(unext)
    else:
        return upast

    return unext

def serial_coarse_with_correction(deriv, init, corrections, dt, tmin, tmax,
        numSteps):
    '''Runs a coarse numerical method in serial, as part of the parareal
    algorithm. 
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
    for ind in xrange(1,numSteps):
        # get a coarse answer
        # TODO is this upast[ind] off by one?
        unext[ind] = forward_euler_step(deriv, times[ind-1], unext[ind-1], dt)

        # update unext with corrections computed with upast
        unext[ind+1] = unext[ind] + fixed_corrections[ind]

    return unext

def parallel_corrections(deriv, upast, tmin, tmax, numSteps, qualityFactor,
        comm, p_root=0):
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
    # in a loop, compute fine and then corrections one piece at a time that
    # processor is responsible for, serially
    while (start < end):
        fineResult = forward_euler(deriv, upast[start], times[start],
                times[start+1], fineNumSteps)

        # get difference for the last time step of the fine result and compare
        # to coarse result answer for same time
        difference = (fineResult[-1] - upast[start+1]).reshape(1, upast.shape[1])
        # concatenate the local results
        local_differences = np.concatenate((local_differences, difference),
                axis=0)

        # move to solve next step
        start += 1

    # Gather the partial results to the root process
    differences = comm.gather(local_differences, root=p_root)

    return differences

def getStart(numtasks, size, rank):
    '''Returns optimally equal partioned index for the begin of the assigned
    array to divide'''
    # Offset by rank to account for n+1 spaces
    if rank < numtasks % size:
        return (numtasks / size + 1) * rank
        
    # Offset only by max numtasks % size
    else:
        return numtasks / size * rank + numtasks % size


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nsteps", default=100, type=int,
            help="Number of time steps over which to solve the ODE")
    parser.add_argument("-q", "--quality-factor", default=100., type=float,
            help="Factor by which to increase the number of time steps")
    parser.add_argument("-k", "--correction-level", default=2, type=int,
            help="Number of parallel correction steps to do")
    parser.add_argument("-d", "--debug", default=False, type=bool,
            help="If switched on, prints all the outputs. Otherwise just prints errors")
    args = parser.parse_args()

    numSteps = args.nsteps
    k = args.correction_level
    qualityFactor = args.quality_factor
    fineNumSteps = qualityFactor * numSteps

    # Example specific parameters and derivatives with IC
    lam = 0.5
    def deriv1(t, u):
        return lam * u
    
    def exact1(t):
        return np.exp(lam * t)

    init1 = np.array([1.])

    def deriv2(t, u):
        y = u[0]
        v = u[1]
        return(np.array([v,
                        -2*v - 5*y]))

    def exact2(t):
        exps = np.exp(-t)
        return(exps * np.cos(2*t) + 
           0.5*exps * np.sin(2*t))

    init2 = np.array([1., 0.])

    derivs = [deriv1, deriv2]
    inits = [init1, init2]
    times_lst = [(0., 1.), (0., 3.)]
    exacts = [exact1, exact2]

    iters = 1
    # wrap everything in a for loop
    for (deriv, init, time_tup, exact) in itertools.izip(derivs, inits,
            times_lst, exacts):
        tmin, tmax = time_tup

        # Communciation barrier and timing function of MPI
        comm.barrier()
        p_start = MPI.Wtime()

        # Use parareal algorithm
        # we only want column 1 so numpy doesn't complain
        p_result = startParareal(deriv, init, k, tmin, tmax, numSteps, comm,
                qualityFactor = 100)

        # Communciation barrier and ending time with MPI timing function
        comm.barrier()
        p_stop = MPI.Wtime() 

        # Check and output results on process 0
        if rank == 0:
            p_result = p_result[:,0]
            # compute serial result
            s_start = time.time()

            # we only want column 1 so numpy doesn't complain
            s_result = forward_euler(deriv, init, tmin, tmax, fineNumSteps)[:,0]
            s_stop = time.time()

            # compute exact result
            coarse_times = np.linspace(tmin, tmax, numSteps+1)
            fine_times = np.linspace(tmin, tmax, fineNumSteps+1)
            e_result_p = exact(coarse_times)
            e_result_s = exact(fine_times)

            print("\n\nExample %d\n" %iters)
            iters += 1

            # compute errors
            p_error = np.abs(p_result - e_result_p)# / abs(e_result)
            s_error = np.abs(s_result - e_result_s)# / abs(e_result)

            # print parameters:
            print("Parameters:")
            print("numSteps:        %d" %numSteps)
            print("Quality Factor:  %d" %qualityFactor)
            print("Corrections:     %d" %k)
            print("")
            # print results if debug mode on
            if args.debug:
                print("Serial Result   = \n" + str(s_result))
                print("Parallel Result = \n" + str(p_result))
                print("Exact           = \n" + str(e_result))
                print("")

            # print errors
            #print("Parallel Error = \n" + str(p_error))
            #print("Serial Error = \n" + str(s_error))
            print("Serial Max Abs Error    = %e" % np.max(s_error))
            print("Parallel Max Abs Error  = %e" % np.max(p_error))
            print("Serial Mean Abs Error    = %e" % np.mean(s_error))
            print("Parallel Mean Abs Error  = %e" % np.mean(p_error))

            # print timers
            print("Serial Time:   %f secs" % (s_stop - s_start))
            print("Parallel Time: %f secs" % (p_stop - p_start))

            print("numSteps,Quality Factor,Corrections,Serial Max Abs Error,Parallel Max Abs Error,Serial Mean Abs Error,Parallel Mean Abs Error, Serial Time, Parallel Time")

            print("%d, %d, %d, %e, %e, %e, %e, %f, %f" %(numSteps,
                qualityFactor, k, np.max(s_error), np.max(p_error),
                np.mean(s_error), np.mean(p_error), (s_stop - s_start), (p_stop
                    - p_start)))
