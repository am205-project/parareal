import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpi4py import MPI
plt.ion()

"""
class MeshPlotter3D:
        '''A class to help with 3D mesh plotting'''
        def __init__(self):
		'''Perform the required precomputation and make a dummy plot'''
		self.fig = plt.figure()
		self.axes = Axes3D(self.fig)
		self.mesh = self.axes.plot_wireframe(0, 0, 0)  # Dummy mesh
		self.axes.set_xlabel('I')
		self.axes.set_ylabel('J')

	def __plot_figure(self, X, Y, Z, zMin=-0.25, zMax=0.5):
		'''Private helper function to plot
		the (x,y,z) triples in the 2D arrays X, Y, Z'''
		self.mesh.remove()
		self.mesh = self.axes.plot_wireframe(X, Y, Z)
		self.axes.set_zlim3d(zMin, zMax);

	def draw_now(self, I, J, u):
		'''Update the plot with the data from u'''
		self.__plot_figure(I, J, u)
		plt.draw();

	def save_now(self, I, J, u, filename):
		'''Update the plot with the data from u'''
		self.__plot_figure(I, J, u)
		self.fig.savefig(filename)

"""

class MeshPlotter3DParallel:
  '''A class to help with 3D interactive plotting from distributed data'''
  def __init__(self, comm=MPI.COMM_WORLD):
    '''Perform the required precomputation and make an initial plot'''
    self.comm = comm
    if self.comm.Get_rank() == 0:
      self.plotter = MeshPlotter3D()

  def __gather_data(self, I, J, u):
    # Sanity check
    assert I.size == J.size == u.size
    # TODO: Generalize beyond integers?
    assert I.dtype == J.dtype == np.int64 or I.dtype == J.dtype == np.int32

    # Get the size of each distributed portion
    counts = self.comm.gather(u.size, root=0)
    totalsize = np.sum(counts)

    # Allocate a buffer
    if self.comm.Get_rank() == 0:
      I0 = np.zeros(totalsize, dtype=I.dtype)
      J0 = np.zeros(totalsize, dtype=I.dtype)
      u0 = np.zeros(totalsize, dtype=u.dtype)
    else:
      I0, J0, u0 = None, None, None

    # Gather the data with vector-gathers
    self.comm.Gatherv(sendbuf=I.reshape(I.size),
                      recvbuf=(I0, (counts, None)), root=0)
    self.comm.Gatherv(sendbuf=J.reshape(J.size),
                      recvbuf=(J0, (counts, None)), root=0)
    self.comm.Gatherv(sendbuf=u.reshape(u.size),
                      recvbuf=(u0, (counts, None)), root=0)

    # Reorganize
    if self.comm.Get_rank() == 0:
			i0min, j0min = I0.min(), J0.min()
			I0 -= i0min
			J0 -= j0min
			u = np.zeros((I0.max()+1, J0.max()+1))
			I, J = np.zeros(u.shape), np.zeros(u.shape)
			u[I0,J0], I[I0,J0], J[I0,J0] = u0, I0+i0min, J0+j0min
			return I, J, u
    return None, None, None

  def draw_now(self, I, J, u):
    I, J, u = self.__gather_data(I, J, u)
    if self.comm.Get_rank() == 0:
      self.plotter.draw_now(I, J, u)

  def save_now(self, I, J, u, filename):
    I, J, u = self.__gather_data(I, J, u)
    if self.comm.Get_rank() == 0:
      self.plotter.save_now(I, J, u, filename)


if __name__ == '__main__':
	# Small parallel example: oscillating membrane (no communication)
	import sys

	# Global constants
	xMin, xMax = 0.0, 1.0     # Domain boundaries
	yMin, yMax = 0.0, 1.0     # Domain boundaries
	Nx = 64                   # Number of total grid points in x
	Ny = Nx                   # Number of total grid points in y
	dx = (xMax-xMin)/(Nx-1)   # Grid spacing, Delta x
	dy = (yMax-yMin)/(Ny-1)   # Grid spacing, Delta y
	dt = 0.4 * dx             # Time step (Magic factor of 0.4)
	T = 7                     # Time end
	omega = 2.0 * np.pi       # Oscillator frequency

	# Get MPI data
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()

	# Get Px and Py from command line
	try:
		Px = int(sys.argv[1])
		Py = int(sys.argv[2])
	except:
		print 'Usage: mpiexec -n (Px*Py) python Plotter3DCS205.py Px Py'
		sys.exit()

	# Sanity check
	assert Px*Py == MPI.COMM_WORLD.Get_size()

	# Create row and column communicators
	comm_col  = comm.Split(rank%Px)
	comm_row  = comm.Split(rank/Px)
	# Get the row and column indices for this process
	p_row     = comm_col.Get_rank()
	p_col     = comm_row.Get_rank()

	# Local constants
	Nx_local = Nx/Px          # Number of local grid points in x
	Ny_local = Ny/Py          # Number of local grid points in y

        # The global indices: I[i,j] and J[i,j] are indices of u[i,j]
	[I,J] = np.mgrid[(Nx_local*p_col):(Nx_local*(p_col+1)),
                         (Ny_local*p_row):(Ny_local*(p_row+1))]

	# Plot data using parallel plotter -- Gather the data and create one plot
	plotter = MeshPlotter3DParallel()

	# Plot data using a serial plotter -- Create one plot for each process
	#plotter = MeshPlotter3D()

	for k,t in enumerate(np.arange(0,T,dt)):
		# Compute u
		u = 0.5 * np.sin(I*dx*np.pi) * np.sin(J*dy*np.pi) * np.cos(omega*t)

		# Print out the step and simulation time
		if rank == 0:
			print "Step: %d  Time: %f" % (k,t)
		# All processes draw the image. Comment when non-interactive.
		if k % 5 == 0:
			plotter.draw_now(I, J, u)

	# Save an image of the final data
	plotter.save_now(I, J, u, "Oscillator.png")
