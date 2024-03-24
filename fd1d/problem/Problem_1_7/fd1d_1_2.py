#!/usr/bin/env python
# File: fd1d_1_2.py
# Name: D.Saravanan
# Date: 16/11/2021

""" Calculation of amplitude of the sine wave in the lossy dielctric """

import numpy as np

epsz = 8.854e-12  # vacuum permittivity (F/m)
epsilon = 4  # relative permittivity 

musz = 1.2566e-6  # vaccum permeability (H/m)
mu = 1

# impedance 
eta = lambda mu, epsz, epsilon: np.sqrt(mu/(epsz * epsilon))

# reflection coefficient (the fraction that is reflected into medium 1)
gamma = (eta(mu, epsz, epsilon) - eta(musz, epsz, epsilon)) / (eta(mu, epsz, epsilon) + eta(musz, epsz, epsilon))

# transmission coefficient (the fraction that is transmitted into medium 2)
tau = (2 * eta(mu, epsz, epsilon)) / (eta(mu, epsz, epsilon) + eta(musz, epsz, epsilon))

freq = 700e6    # frequency 700 MHz
omega = 2*np.pi*freq # angular frequency
c0 = 3e8        # speed of light in vacuum (m/s)

# wave number
k = (omega/c0) * np.sqrt(epsilon)

# amplitude of an electric field propagating in the positive z direction
# in lossy dielectric medium is given by
E[z] = E[0] * np.exp(-k.real * z) * np.exp(-k.imag * z)
