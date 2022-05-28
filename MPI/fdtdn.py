#!/usr/bin/env python
# File: fdtdn.py
# Name: D.Saravanan
# Date: 21/05/2022

""" Script for 1-dimensional fdtd with gaussian pulse as source """

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt


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
        for time_step in range(start, end + 1):

            ex[1:ke] = ex[1:ke] + 0.5 * (hy[0:ke-1] - hy[1:ke])
            ex[kc] = np.exp(-0.5 * ((t0 - time_step) / spread) ** 2)
            hy[0:ke-1] = hy[0:ke-1] + 0.5 * (ex[0:ke-1] - ex[1:ke])

        comm.send(ex, dest=rank+1, tag=11)
        comm.send(hy, dest=rank+1, tag=12)


    for n in range(size):
        n = n + rank

        print(n)

        ex = comm.recv(source=n-1, tag=11)
        hy = comm.recv(source=n-1, tag=12)

        for time_step in range(start, end + 1):

            ex[1:ke] = ex[1:ke] + 0.5 * (hy[0:ke-1] - hy[1:ke])
            ex[kc] = np.exp(-0.5 * ((t0 - time_step) / spread) ** 2)
            hy[0:ke-1] = hy[0:ke-1] + 0.5 * (ex[0:ke-1] - ex[1:ke])

        comm.send(ex, dest=n+1, tag=11)
        comm.send(hy, dest=n+1, tag=11)

    #print(ex)

    #plt.plot(ex)
    #plt.savefig('fdtdn.png')


if __name__ == "__main__":
    main()
