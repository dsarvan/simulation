#!/usr/bin/env python
# File: vectoradd.py
# Name: D.Saravanan
# Date: 16/05/2022

""" Script for vector addition using numba """

import numpy as np
from timeit import default_timer as timer
from numba import vectorize

@vectorize(["float32(float32, float32)"], target='cuda')
def VectorAdd(a, b):
    return a + b

def main():
    N = 32000000

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    C = np.zeros(N, dtype=np.float32)

    start = timer()
    C = VectorAdd(A, B)
    vectoradd_timer = timer() - start

    start = timer()
    C_np = A + B
    np_vectoradd_timer = timer() - start

    error = np.abs(C - C_np).max()
    print("Error: ", error)

    print("VectorAdd took %f seconds" % vectoradd_timer)
    print("VectorAdd(NP) took %f seconds" % np_vectoradd_timer)


if __name__ == '__main__':
    main()


