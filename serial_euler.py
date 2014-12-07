import numpy as np

def forward_euler_step(deriv, t, u, dt):
    return u + dt * deriv(t, u)

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
    times = np.linspace(time_min, time_max, numSteps+1)[1:]
    time_step = float(time_max - time_min) / (numSteps - 1)

    ys = np.zeros((numSteps+1, dim))
    ys[0] = init

    for (ind, t) in enumerate(times):
        time_ind = ind + 1
        curr = ys[time_ind-1]
        ys[time_ind] = forward_euler_step(deriv, t, curr, time_step)

    return ys

if __name__ == "__main__":

    lam = 1.

    def deriv(t, u):
        return lam * u

    nsteps = 200
    init = np.array([1.])
    
    res = forward_euler(deriv, init, 0, 1, nsteps)[:,0]
    print res
    print len(res)
    
    exact = np.exp(np.linspace(0,1,nsteps+1))
    print exact - res
