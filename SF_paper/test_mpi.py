
from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print('Hello (%s)' % rank)

# Declare the array that will store all the temp results
temps = numpy.zeros((4,5))

# Loop over all directories
if rank==0:
    counter = 0
    for i in range(2):
        for j in range(5):
            temps[i,j] = counter
            counter = counter + 1

else:
    counter = 20
    for i in range(2,4):
        for j in range(5):
            temps[i,j] = counter
            counter = counter + 1

comm.Allreduce(MPI.IN_PLACE,temps,op=MPI.MAX)

if rank==0:
    print(temps)
