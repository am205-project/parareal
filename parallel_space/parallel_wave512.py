# domain Decomposition to solve 2-d wave equation, using isend/irecv

from mpi4py import MPI
import numpy as np 
import sys

def initial_conditions(DTDX, X, Y, comm_col, comm_row):
  '''Construct the grid points and set the initial conditions.
  X[i,j] and Y[i,j] are the 2D coordinates of u[i,j]'''
  assert X.shape == Y.shape

  um = np.zeros(X.shape)     # u^{n-1}  "u minus"
  u  = np.zeros(X.shape)     # u^{n}    "u"
  up = np.zeros(X.shape)     # u^{n+1}  "u plus"
  # Define Ix and Iy so that 1:Ix and 1:Iy define the interior points
  Ix = u.shape[0] - 1
  Iy = u.shape[1] - 1
  # Set the interior points: Initial condition is Gaussian
  u[1:Ix,1:Iy] = np.exp(-50 * (X[1:Ix,1:Iy]**2 + Y[1:Ix,1:Iy]**2))
  # Set the ghost points to the boundary conditions
  set_ghost_points(u, comm_col, comm_row)
  # Set the initial time derivative to zero by running backwards
  apply_stencil(DTDX, um, u, up)
  um *= 0.5
  # Done initializing up, u, and um
  return up, u, um

def apply_stencil(DTDX, up, u, um):
  '''Apply the computational stencil to compute u^{n+1} -- "up".
  Assumes the ghost points exist and are set to the correct values.'''

  # Define Ix and Iy so that 1:Ix and 1:Iy define the interior points
  Ix = u.shape[0] - 1
  Iy = u.shape[1] - 1
  # Update interior grid points with vectorized stencil
  up[1:Ix,1:Iy] = ((2-4*DTDX)*u[1:Ix,1:Iy] - um[1:Ix,1:Iy]
                   + DTDX*(u[0:Ix-1,1:Iy  ] +
                           u[2:Ix+1,1:Iy  ] +
                           u[1:Ix  ,0:Iy-1] +
                           u[1:Ix  ,2:Iy+1]))

def set_ghost_points(u, comm_col, comm_row):
  '''Set the ghost points.
  In serial, the only ghost points are the boundaries.
  In parallel, each process will have ghost points:
      some will need data from neighboring processes,
      others will use these boundary conditions.'''

  row_rank = comm_col.Get_rank()
  col_rank = comm_row.Get_rank()
  num_rows = comm_col.Get_size()
  num_cols = comm_row.Get_size()

  # Define Nx and Ny so that Nx+1 and Ny+1 are the boundary points
  Ix = u.shape[0] - 1
  Iy = u.shape[1] - 1

  # Domain Decomposition: in each column, send bottom of each row to ghost points of top of next row
  # in each column, send top of each row to ghost points of bottom of previous row
  top_buffer = np.ascontiguousarray(u[0,1:Iy])
  bot_buffer = np.ascontiguousarray(u[Ix,1:Iy])

  recv_top = comm_col.Irecv(top_buffer, source=(row_rank-1)%num_rows)
  comm_col.Isend(u[Ix-1, 1:Iy], dest=(row_rank+1)%num_rows)
  recv_top.Wait(MPI.Status())

  recv_bot = comm_col.Irecv(bot_buffer, source=(row_rank+1)%num_rows)
  comm_col.Isend(u[1,1:Iy], dest=(row_rank-1)%num_rows)
  recv_bot.Wait(MPI.Status())

  u[0,1:Iy] = top_buffer
  u[Ix,1:Iy] = bot_buffer

  # equivalent to:
  #comm_col.Sendrecv(u[Ix-1,1:Iy], dest=(row_rank+1)%num_rows, recvbuf=u[0,1:Iy], source=(row_rank-1)%num_rows)
  #comm_col.Sendrecv(u[1,1:Iy], dest=(row_rank-1)%num_rows, recvbuf=u[Ix,1:Iy], source=(row_rank+1)%num_rows)

  # Domain Decomposition: in each row, send left part of each column to right ghost points of previous
  # in each column, send right part of each column to left ghost points of ghost points of next
  # Can't send columns, make it contiguous so we can send as buffer
  left_buffer = np.ascontiguousarray(u[1:Ix,0])
  right_buffer = np.ascontiguousarray(u[1:Ix, Iy])

  recv_left = comm_row.Irecv(left_buffer, source=(col_rank-1)%num_cols)
  comm_row.Isend(np.ascontiguousarray(u[1:Ix,Iy-1]), dest=(col_rank+1)%num_cols)
  recv_left.Wait(MPI.Status())

  recv_right = comm_row.Irecv(right_buffer, source=(col_rank+1)%num_cols)
  comm_row.Isend(np.ascontiguousarray(u[1:Ix,1]), dest=(col_rank-1)%num_cols)
  recv_right.Wait(MPI.Status())

  u[1:Ix,0] = left_buffer
  u[1:Ix,Iy] = right_buffer

  # equivalent to:
  #comm_row.Sendrecv(np.ascontiguousarray(u[1:Ix,Iy-1]), dest=(col_rank+1)%num_cols, recvbuf=temp1, source=(col_rank-1)%num_cols)
  #comm_row.Sendrecv(np.ascontiguousarray(u[1:Ix,1]), dest=(col_rank-1)%num_cols, recvbuf=temp2, source=(col_rank+1)%num_cols)


  # Update ghost points with boundary condition, overwrites the values that were looped back earlier with mod
  if row_rank == 0:
    u[0,:] = u[2,:] # u_{0,j} = u_{2,j} x = 0
  if row_rank == num_rows - 1:
    u[Ix,:] = u[Ix-2,:] # u_{Nx+1,j} = u_{Nx-1,j} x = 1
  if col_rank == 0:
    u[:,0] = u[:,2] # u_{i,0} = u_{i,2} y = 0
  if col_rank == num_cols - 1:
    u[:,Iy] = u[:,Iy-2] # u_{i,Ny+1} = u_{i,Ny-1} y = 1

if __name__ == '__main__':
# Global constants
  xMin, xMax = 0.0, 1.0     # Domain boundaries
  yMin, yMax = 0.0, 1.0     # Domain boundaries
  Nx = 512                   # Number of total grid points in x
  Ny = Nx                   # Number of total grid points in y
  dx = (xMax-xMin)/(Nx-1)   # Grid spacing, Delta x
  dy = (yMax-yMin)/(Ny-1)   # Grid spacing, Delta y
  dt = 0.4 * dx             # Time step (Magic factor of 0.4)
  T = 5                     # Time end
  DTDX = (dt*dt) / (dx*dx)  # Precomputed CFL scalar

  # Get MPI data
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  # Get Px and Py from command line
  try:
    Px = int(sys.argv[1])
    Py = int(sys.argv[2])
  except:
    print 'Usage: mpiexec -n (Px*Py) python parallel_wave.py Px Py'
    sys.exit()

  # Sanity check
  assert Px*Py == MPI.COMM_WORLD.Get_size()

  # Create row and column communicators
  comm_col  = comm.Split(rank%Px)
  comm_row  = comm.Split(rank/Px)

  # Get the row and column indicies for this process
  p_row     = comm_col.Get_rank()
  p_col     = comm_row.Get_rank()

  # Local constants
  Nx_local = Nx/Px          # Number of local grid points in x
  Ny_local = Ny/Py          # Number of local grid points in y

  # The global indices: I[i,j] and J[i,j] are indices of u[i,j]
  [I,J] = np.mgrid[Ny_local*p_row-1:Ny_local*(p_row+1)+1,
                    Nx_local*p_col-1:Nx_local*(p_col+1)+1]

  # Convenience so u[1:Ix,1:Iy] are all interior points
  Ix, Iy = Ny_local + 1, Nx_local + 1

  # Set the initial conditions
  up, u, um = initial_conditions(DTDX, I*dx-0.5 , J*dy, comm_col, comm_row)

  comm.barrier()
  p_start = MPI.Wtime()
  for k,t in enumerate(np.arange(0,T,dt)):
    # Compute u^{n+1} with the computational stencil
    apply_stencil(DTDX, up, u, um)

    # Set the new boudnary points/ghost points on u^{n+1}
    set_ghost_points(up, comm_col, comm_row)

    # Swap references for the next step
    # u^{n-1} <- u^{n}
    # u^{n}   <- u^{n+1}
    # u^{n+1} <- u^{n-1} to be overwritten in next step
    um, u, up = u, up, um

  comm.barrier()
  time = (MPI.Wtime() - p_start)
  average = time / (T / dt)
  if rank == 0:
    print "Time per iteration: %f" % average
    print "Total time: %f" % time