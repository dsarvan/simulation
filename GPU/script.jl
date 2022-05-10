#!/usr/bin/env julia
# File: script.jl
# Name: D.Saravanan
# Date: 09/05/2022

""" Script to compare CPU vs. GPU """

using CUDA
using Printf

function benchmark_processor(arr, func, argument)
    """ function that is going to be used for benchmarking """
    start_time = time()
    func(arr .* argument)
    final_time = time()
    elapsed_time = final_time - start_time
    return elapsed_time
end

# load a matrix to global memory
array_cpu = rand(0:255, 9999, 9999)

# load the same matrix to GPU memory
array_gpu = CuArray(array_cpu)

# benchmark matrix addition on CPU by using 
cpu_time = benchmark_processor(array_cpu, sum, 999)

# benchmark matrix addition on GPU by using CUDA addition function
gpu_time = benchmark_processor(array_gpu, sum, 999)

# determine how much is GPU faster
processor = abs((gpu_time - cpu_time)/gpu_time) * 100

@printf("CPU time: %f seconds\nGPU time: %f seconds\nGPU was %f percent faster\n", cpu_time, gpu_time, processor)
