#!/usr/bin/env python
# File: compute.py
# Name: D.Saravanan
# Date: 13/05/2022

""" Script to call cos on every element of a float array """

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import numpy as np
import sys

for size in range(1, 32):

    size_of_x = size
    size_of_y = size

    if((size_of_x * size_of_y) > 1024):
        print("Error will this is bad")
        sys.exit()


    for n in range(0, 100):

        a_gpu = gpuarray.to_gpu(np.random.randn(size_of_x, size_of_y).astype(np.float32))
        a_gpucos = (cumath.cos(a_gpu)).get()
        print(a_gpucos)
