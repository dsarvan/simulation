#!/usr/bin/env python
# File: sequential_pi.py
# Name: D.Saravanan
# Date: 21/05/2022

""" Script for computing pi """

import math

def compute_pi(n):
    h = 1./n
    s = 0.
    for i in range(n):
        x = h * (i + 0.5)
        s += 4./(1. + x**2)
    return s*h

n = 10
pi = compute_pi(n)
error = abs(pi - math.pi)

print("pi is approximately %.16f, "
      "error is %.16f" % (pi, error))

