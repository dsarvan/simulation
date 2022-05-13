#!/usr/bin/env python
# File: compute.py
# Name: D.Saravanan
# Date: 13/05/2022

""" Script to call cos on every element of a float array """

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import sys

mod = SourceModule("""
    __global__ void gpucos(float *a) {
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] = cos(a[idx]);
    }
""")

for size in range(1, 32):

    size_of_x = size
    size_of_y = size

    if((size_of_x * size_of_y) > 1024):
        print("Error will this is bad")
        sys.exit()

    time_cpu = []
    time_gpu = []

    for n in range(0, 100):
        a = np.random.randn(size_of_x, size_of_y).astype(np.float32)
        
        # allocate memory on the device
        a_gpu = cuda.mem_alloc(a.nbytes)

        # transfer the data to the GPU
        cuda.memcpy_htod(a_gpu, a)

        func = mod.get_function("gpucos")
        func(a_gpu, block=(size_of_x, size_of_y, 1))

        a_gpucos = np.empty_like(a)
        cuda.memcpy_dtoh(a_gpucos, a_gpu)
        print(a_gpucos)
