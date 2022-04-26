#!/usr/bin/env python
# File: fd1d_2_2.py
# Name: D.Saravanan
# Date: 26/01/2022

""" Simulation of a pulse striking a dielectric medium and implements
    the discrete Fourier transform with a Gaussian pulse as its source """

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["pgf.texsystem"] = "pdflatex"
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "figure.titlesize": 10,
    }
)

ke = 201
ex = np.zeros(ke)
ix = np.zeros(ke)
Dx = np.zeros(ke)
hy = np.zeros(ke)

dx = 0.01   # cell size
dt = dx / 6e8   # time step size
freq = np.array((500e6, 200e6, 100e6))  # frequency 500 MHs, 200 MHz, 100 MHz
nfreq = len(freq)   # number of frequencies

boundary_low = [0, 0]
boundary_high = [0, 0]

# dielectric profile
epsz = 8.854e-12    # vaccum permittivity (F/m)
epsr = 4    # relative permittivity
sigma = 0   # conductivity (S/m)

gax = np.ones(ke)
gbx = np.zeros(ke)
gax[100:] = 1/(epsr + (sigma * dt/epsz))
gbx[100:] = sigma * dt/epsz

# used in the fourier transform
real_in = np.zeros(nfreq)
imag_in = np.zeros(nfreq)

amp_in = np.zeros(nfreq)
phase_in = np.zeros(nfreq)

real_pt = np.zeros((nfreq, ke))
imag_pt = np.zeros((nfreq, ke))

amp = np.zeros((nfreq, ke))
phase = np.zeros((nfreq, ke))

# gaussian pulse
t0 = 50
spread = 10
nsteps = 400

# FDTD loop
for time_step in range(1, nsteps + 1):

    # calculate the Dx flux
    for k in range(1, ke):
        Dx[k] = Dx[k] + 0.5 * (hy[k - 1] - hy[k])

    # put a gaussian pulse at the low end
    Dx[5] = Dx[5] + np.exp(-0.5 * ((t0 - time_step)/spread)**2)

    # calculate the Ex field from Dx
    for k in range(1, ke):
        ex[k] = gax[k] * (Dx[k] - ix[k])
        ix[k] = ix[k] + gbx[k] * ex[k]

    # calculate the fourier transform of Ex
    for k in range(ke):
        for m in range(nfreq):
            real_pt[m, k] = real_pt[m, k] + np.cos(2*np.pi * m * dt * time_step) * ex[k]
            imag_pt[m, k] = imag_pt[m, k] - np.sin(2*np.pi * m * dt * time_step) * ex[k]

    # fourier trasform of the input pulse
    if time_step < 100:
        for m in range(nfreq):
            real_in[m] = real_in[m] + np.cos(2*np.pi * m * dt * time_step) * ex[10]
            imag_in[m] = imag_in[m] - np.sin(2*np.pi * m * dt * time_step) * ex[10]

    # absorbing boundary conditions
    ex[0] = boundary_low.pop(0)
    boundary_low.append(ex[1])
    ex[ke - 1] = boundary_high.pop(0)
    boundary_high.append(ex[ke - 2])

    # calculate the Hy field
    for k in range(ke - 1):
        hy[k] = hy[k] + 0.5 * (ex[k] - ex[k + 1])

    # calculate the amplitude and phase at each frequency
    for m in range(nfreq):
        amp_in[m] = np.sqrt(real_in[m]**2 + imag_in[m]**2)
        phase_in[m] = np.arctan2(imag_pt[m], real_pt[m])

        for k in range(ke):
            amp[m,k] = (1/amp_in[m]) * np.sqrt(real_pt[m,k]**2 + imag_pt[m,k]**2)
            phase[m,k] = np.arctan2(imag_pt[m,k], real_pt[m,k]) - phase_in[m]






















