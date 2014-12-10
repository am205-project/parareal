import numpy as np

def integrate(deriv, init, time_min, time_max, numSteps, step):
    '''
    Solves a differential equation using a callable method from time_min to
    time_max, using a step of time_step

    deriv: callable; the derivative
    init: numpy 1D array; initial condition
    time_min: float; Minimum time
    time_max: float; Maximum time
    numSteps: int; number of stes
    step: callable; takes deriv, current time, current y, and time step and
    returns the next current y
    '''
    dim = init.shape[0]
    # checks for 1D array
    assert(init.shape == (dim,))

    # initialize times
    times = np.linspace(time_min, time_max, numSteps+1)
    time_step = float(time_max - time_min) / (numSteps - 1)

    ys = np.zeros((numSteps+1, dim))
    ys[0] = init

    for ind in xrange(numSteps):
        t = times[ind]
        curr = ys[ind]
        ys[ind + 1] = step(deriv, t, curr, time_step)

    return ys


def forward_euler_step(deriv, t, u, dt):
    return(u + dt * deriv(t, u))

def forward_euler(deriv, init, time_min, time_max, numSteps):
    return(integrate(deriv, init, time_min, time_max, numSteps,
            forward_euler_step))

def rk4_step(deriv, t, u, dt):
    k1 = deriv(t, u)
    k2 = deriv(t + (dt/2.), u + 0.5*k1*dt)
    k3 = deriv(t + (dt/2.), u + 0.5*k2*dt)
    k4 = deriv(t + dt, u + k3*dt)

    return(u + ((dt/6.) * (k1 + 2*k2 + 2*k3 + k4)))

def rk4(deriv, init, time_min, time_max, numSteps):
    return(integrate(deriv, init, time_min, time_max, numSteps, rk4_step))

def midpoint_step(deriv, t, u, dt):
    k1 = dt * deriv(t, u)
    return(u + dt*deriv(t + 0.5*dt, u + 0.5*k1))

def midpoint_method(deriv, init, time_min, time_max, numSteps):
    return(integrate(deriv, init, time_min, time_max, numSteps, midpoint_step))

if __name__ == "__main__":

    def deriv(t, u):
        y = u[0]
        v = u[1]
        return(np.array([v,
                        -2*v - 5*y]))

    def exact(t):
        exps = np.exp(-t)
        return(exps * np.cos(2*t) + 0.5*exps * np.sin(2*t))

    nsteps_lst = [200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
    init = np.array([1., 0.])
    tmin = 0.0
    tmax = 30.
    #print("type,nsteps,err")
    with open('out', 'wb') as f:
        f.write('type,nsteps,err\n')
        for nsteps in nsteps_lst:
            fe_res = forward_euler(deriv, init, tmin, tmax, nsteps)[:,0]
            rk4_res = rk4(deriv, init, tmin, tmax, nsteps)[:,0]
            midpoint_res = midpoint_method(deriv, init, tmin, tmax, nsteps)[:,0]
            
            times = np.linspace(tmin, tmax, nsteps+1)
            exact_res = exact(times)
            fe_err = np.abs(fe_res - exact_res)
            rk4_err = np.abs(rk4_res - exact_res)
            midpoint_err = np.abs(midpoint_res - exact_res)

            f.write("euler,%d,%e" %(nsteps, np.mean(fe_err)) + '\n')
            f.write("rk4,%d,%e" %(nsteps, np.mean(rk4_err)) + '\n')
            f.write("midpt,%d,%e" %(nsteps, np.mean(midpoint_err)) + '\n')
