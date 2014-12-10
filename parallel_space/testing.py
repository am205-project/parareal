# Parallelism by Space 
import numpy as np
import custom_plot
import copy
from mpi4py import MPI
import time
import argparse
import itertools

# (a)
# pressure in region S
def pressure_s(t, p=10., w=100.*np.pi):
    return p*np.sin(w*t)

def pressure(up, right, down, left, center, prev, c=34300., h=36.6):
    dt = h/2./c
    return c*c/h/h*dt*dt*(up+right+down+left-4*center)+2.*center-prev

p_B = [0,0] # person at (10, 167)
p_C = [0,0] # person at (35, 73)
p_M = [0,0] # person at (67, 115)

h = 36.6
c = 34300.

pierce = np.loadtxt("pierce.txt", dtype=np.int8)

# simulates from t = 0 to given t
def simulate(t, fn):
    region = np.zeros([100, 200])
    
    # set initial condition in region S
    for j in xrange(57, 61):
        for k in xrange(15, 19):
            region[j][k] = 1
    
    p0 = np.zeros([100, 200])
    p1 = np.zeros([100, 200])
    for (x, y),v in np.ndenumerate(p1):
        if region[x][y] == 1:
            p1[x][y] = pressure_s(h/2./c)
    
    p_pprev = copy.copy(p0)
    p_prev = copy.copy(p1)
    p_curr = copy.copy(p_prev)
    
    t0 = h/c
    dt = h/2./c

    for t in np.arange(t0, t+dt/2., dt):
        for (x, y), v in np.ndenumerate(p_prev):
            # nothing on map for these values
            if x == 0 or x == 99 or y == 0 or y == 199:
                continue
            
            # region S is driven to satisfy this condition
            if region[x][y] == 1:
                p_curr[x][y] = pressure_s(t)
                continue
                
            # if x, y is not a wall
            if pierce[x][y] == 0:
                up = p_prev[x][y-1]
                right = p_prev[x+1][y]
                down = p_prev[x][y+1]
                left = p_prev[x-1][y]
            
                # if neighbors are walls, ghost nodes
                if pierce[x][y-1] == 1:
                    up = p_prev[x][y]
                if pierce[x+1][y] == 1:
                    right = p_prev[x][y]
                if pierce[x][y+1] == 1:
                    down = p_prev[x][y]
                if pierce[x-1][y] == 1:
                    left = p_prev[x][y]
        
                p_curr[x][y] = pressure(up, right, down, left, p_prev[x][y], p_pprev[x][y])
            if x == 10 and y == 167:
                p_B.append(p_curr[x][y])
            elif x == 35 and y == 73:
                p_C.append(p_curr[x][y])
            elif x == 67 and y == 115:
                p_M.append(p_curr[x][y])
        
        p_pprev = copy.copy(p_prev)
        p_prev = copy.copy(p_curr)
    
    custom_plot.plot1(fn, p_curr, pierce, -1.1, 1.1, 3)

if __name__ == "__main__":
    simulate(0.015, 'out_0015.png')
