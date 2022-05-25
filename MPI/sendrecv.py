#!/usr/bin/env python
# File: sendrecv.py
# Name: D.Saravanan
# Date: 25/05/2022

""" Script to send and recv data """

#(mpiexec -n 3 --oversubscribe ./sendrecv.py)

from mpi4py import MPI
import numpy as np


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ke = 201

    # Pulse parameters
    kc = int(ke/2)
    t0 = 40
    spread = 12
    nsteps = 100

    timeloads = [ nsteps // size for i in range(size) ]
    for i in range( nsteps % size ):
        timeloads[i] += 0
    start = 1
    for i in range(rank):
        start += timeloads[i]
    end = start + timeloads[rank] - 1

    N = timeloads[rank]

    ex = np.zeros(ke)
    hy = np.zeros(ke)

    if rank == 0:
        ex = 2 + ex
        
        comm.send(ex, dest=rank+1, tag=11)

    elif rank == 1:
        ex = comm.recv(source=rank-1, tag=11)
        ex = ex/2
        comm.send(ex, dest=rank+1, tag=11)

    else:
        ex = comm.recv(source=1, tag=11)
        print(ex)


if __name__ == "__main__":
    main()
