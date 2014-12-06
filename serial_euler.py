import numpy as np

def forward_euler(deriv, init, time_min, time_max, time_step):
    '''
    Solves a differential equation using Euler's method from time_min to
    time_max, using a step of time_step

    deriv: callable; the derivative
    init: numpy 1D array; initial condition
    time_min: float; Minimum time
    time_max: float; Maximum time
    time_step: float; step size
    '''
    dim = init.shape[0]
    # checks for 1D array
    assert(init.shape == (dim,))

    # initialize times
    nsteps = (float(time_max - time_min) / time_step) + 1
    times = np.linspace(time_min, time_max, nsteps)[1:]

    ys = np.zeros((nsteps, dim))
    ys[0] = init

    for (ind, t) in enumerate(times):
        time_ind = ind + 1
        curr = ys[time_ind-1]
        ys[time_ind] = curr + time_step * deriv(t, curr)

    return ys

def test():
    lam = 1.

    def deriv(t, u):
        return lam * u

    init = np.array([1.])

    res = forward_euler(deriv, init, 0, 1, 0.01)[:,0]
    
    exact = np.exp(np.linspace(0,1,len(res)))
    print exact - res


def main():
    test()

if __name__ == "__main__":
    main()
