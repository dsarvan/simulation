#!/usr/bin/env python
# File: script.py
# Name: D.Saravanan
# Date: 04/05/2022

""" Script to compare CPU vs. GPU """

import numpy as np
import cupy as cp
from time import time

def benchmark_processor(arr, func, argument):
    """ function that is going to be used for benchmarking """
    start_time = time()
    func(arr, argument)
    final_time = time()
    elapsed_time = final_time - start_time
    return elapsed_time

# load a matrix to global memory
array_cpu = np.random.randint(0, 255, size=(9999, 9999))

# load the same matrix to GPU memory
array_gpu = np.asarray(array_cpu)

# benchmark matrix addition on CPU by using NumPy addition function
cpu_time = benchmark_processor(array_cpu, np.add, 999)

# you need to run a pilot iteration on GPU first to compile and cache the function kernel on GPU
benchmark_processor(array_gpu, cp.add, 1)

# benchmark matrix addition on GPU by using CuPy addition function
#gpu_time = benchmark_processor(array_gpu, cp.add, 999)

# determine how much is GPU faster
#processor = (gpu_time - cpu_time)/gpu_time * 100

#print(f"CPU time: {cpu_time} seconds\nGPU time: {gpu_time} seconds.\nGPU was {processor} percent faster") 
