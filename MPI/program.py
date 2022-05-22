#!/usr/bin/env python
# File: program.py
# Name: D.Saravanan
# Date: 22/05/2022

""" Script to query information about communicators """

from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
name = MPI.Get_processor_name()

print("Hello, World! "
      "I am process %d of %d on %s" %
      (rank, size, name))
