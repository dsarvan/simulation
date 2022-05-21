#!/usr/bin/env python
# File: fdtd.py
# Name: D.Saravanan
# Date: 21/05/2022

""" Script for 1-dimensional fdtd with gaussian pulse as source """

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
