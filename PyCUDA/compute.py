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
