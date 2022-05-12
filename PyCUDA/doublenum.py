#!/usr/bin/env python
# File: doublenum.py
# Name: D.Saravanan
# Date: 12/05/2022

""" Script to double each entry in gpu array """

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# 4x4 array of random numbers
a = np.random.randn(4,4)

# double precision to single precision
a = a.astype(np.float32)

# allocate memory on the device
a_gpu = cuda.mem_alloc(a.nbytes)

# transfer the data to the GPU
cuda.memcpy_htod(a_gpu, a)

# executing a kernel
mod = SourceModule("""
    __global__ void doublify(float *a) {
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] *= 2;
    }
""")

func = mod.get_function("doublify")
func(a_gpu, block=(4,4,1))

a_doubled = np.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print(a_doubled)
print(a)
