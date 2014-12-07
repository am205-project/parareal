import numpy as np

def forward_euler(deriv, init, time_min, time_max, numSteps):
    '''
    Solves a differential equation using Euler's method from time_min to
    time_max, using a step of time_step

    deriv: callable; the derivative
    init: numpy 1D array; initial condition
    time_min: float; Minimum time
    time_max: float; Maximum time
    numSteps: int; number of stes
    '''
    dim = init.shape[0]
    # checks for 1D array
    assert(init.shape == (dim,))

    # initialize times
    times = np.linspace(time_min, time_max, numSteps)[1:]

    ys = np.zeros((nsteps, dim))
    ys[0] = init

    for (ind, t) in enumerate(times):
        time_ind = ind + 1
        curr = ys[time_ind-1]
        ys[time_ind] = curr + time_step * deriv(t, curr)

    return ys

if __name__ == "__main__":

    lam = 1.

    def deriv(t, u):
        return lam * u

    init = np.array([1.])
    
    res = forward_euler(deriv, init, 0, 1, 0.01)[:,0]
    
    exact = np.exp(np.linspace(0,1,len(res)))
    print res
    print exact - res
