#!/usr/bin/env python
# File: parallel_pi.py
# Name: D.Saravanan
# Date: 21/05/2022

""" Script for computing pi """

from mpi4py import MPI
import math

def compute_pi(n, start=0, step=1):
    h = 1./n
    s = 0.
    for i in range(start, n, step):
        x = h * (i + 0.5)
        s += 4./(1. + x**2)
    return s*h

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    n = 10
else:
    n = None

print(n)

n = comm.bcast(n, root=0)

pi = compute_pi(n, rank, size)

if rank == 0:
    error = abs(pi - math.pi)
    print("pi is approximately %.16f, "
          "error is %.16f" % (pi, error)) 
