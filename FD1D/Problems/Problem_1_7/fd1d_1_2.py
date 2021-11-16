#!/usr/bin/env python
# File: fd1d_1_2.py
# Name: D.Saravanan
# Date: 16/11/2021

""" Calculation of amplitude of the sine wave in the lossy dielctric """

import numpy as np

epsz = 8.854e-12  # vacuum permittivity (F/m)
epsilon = 4  # relative permittivity 

# impedance 
eta = lambda mu, epsz, epsilon: np.sqrt(mu/(epsz * epsilon))

# reflection coefficient
gamma = eta(mu, epsz, epsilon) - eta(mu, epsz, epsilon)/eta(mu, epsz, epsilon) + eta(mu, epsz, epsilon)

# transmission coefficient
tau = 2 * eta(mu, epsz, epsilon)/eta(mu, epsz, epsilon) + eta(mu, epsz, epsilon)

# wave number
k = (omega/c0) * np.sqrt(epsilon)
