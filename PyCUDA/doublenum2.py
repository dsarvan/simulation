#!/usr/bin/env python
# File: doublenum.py
# Name: D.Saravanan
# Date: 12/05/2022

""" Script to double each entry in gpu array """

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

a_gpu = gpuarray.to_gpu(np.random.randn(4,4).astype(np.float32))
a_doubled = (2 * a_gpu).get()
print(a_doubled)
