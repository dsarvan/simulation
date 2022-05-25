#!/usr/bin/env python
# File: gather.py
# Name: D.Saravanan
# Date: 25/05/2022

""" Script to gather data from all processes in a communicator """

from mpi4py import MPI
import numpy as np


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ke = 100

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
        for time_step in range(start, end + 1):
            ex = 2 + ex
        
    exn = np.zeros(N)
    comm.Scatter(ex, exn, root=0)

    if rank == 0:
        print(exn)

    if rank == 3:
        print(exn)

    exnn = np.zeros(ke)
    comm.Gather(exn, exnn, root=0)

    if rank == 0:
        print(exnn)


if __name__ == "__main__":
    main()
