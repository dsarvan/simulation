#!/usr/bin/env python
# File: fd1d_2_2.py
# Name: D.Saravanan
# Date: 07/12/2021

""" The Fourier Transform has been added """

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
freq = np.array((500e6, 200e6, 100e6))  # frequency 500 MHz, 200 MHz, 100 MHz
nfreq = len(freq)   # number of frequencies

boundary_low = [0, 0]
boundary_high = [0, 0]

# dielectric profile
epsz = 8.854e-12    # vacuum permittivity (F/m)
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

# desired points for plotting
points = [
    {"num_steps": 200, "ex": None, "amp": None, "phase": None, "label": " ", "label_ab": "(a)"},
    {"num_steps": 400, "ex": None, "amp": None, "phase": None, "label": "FDTD cells", "label_ab": "(b)"}
]

t0 = 50
spread = 10
nsteps = 400

# FDTD loop
for time_step in range(1, nsteps + 1):
    
    # calculate the Dx flux
    for k in range(1, ke):
        Dx[k] = Dx[k] + 0.5 * (hy[k - 1] - hy[k])

    # put a sinusoidal at the low end
    Dx[5] = Dx[5] + np.exp(-0.5 * ((t0 - time_step)/spread)**2)

    # calculate the Ex field from Dx
    for k in range(1, ke):
        ex[k] = gax[k] * (Dx[k] - ix[k])
        ix[k] = ix[k] + gbx[k] * ex[k]

    # calculate the Fourier transform of Ex
    for k in range(ke):
        for m in range(nfreq):
            real_pt[m, k] = real_pt[m, k] + np.cos(2*np.pi * m* dt * time_step) * ex[k]
            imag_pt[m, k] = imag_pt[m, k] - np.sin(2*np.pi*m*dt*time_step) * ex[k]

    # fourier transform of the input pulse
    if time_step < 100:
        for m in range(nfreq):
            real_in[m] = real_in[m] + np.cos(2*np.pi*m*dt*time_step) * ex[10]
            imag_in[m] = imag_in[m] - np.sin(2*np.pi*m*dt*time_step) * ex[10]

    # absorbing boundary conditions
    ex[0] = boundary_low.pop(0)
    boundary_low.append(ex[1])
    ex[ke - 1] = boundary_high.pop(0)
    boundary_high.append(ex[ke - 2])

    # calculate the Hy field
    for k in range(ke - 1):
        hy[k] = hy[k] + 0.5 * (ex[k] - ex[k + 1])

    # save data at certain points for plotting
    for plot_data in points:
        if time_step == plot_data["num_steps"]:
            
            # calculate the amplitude and phase at each frequency
            for m in range(nfreq):
                amp_in[m] = np.sqrt(real_in[m]**2 + imag_in[m]**2)
                phase_in[m] = np.arctan2(real_in[m], imag_in[m])

                for k in range(ke):
                    amp[m,k] = (1/amp_in[m]) * np.sqrt(real_pt[m,k]**2 + imag_pt[m,k]**2)
                    phase[m,k] = np.arctan2(real_pt[m,k] , imag_pt[m,k]) - phase_in[m]

            plot_data["ex"] = np.copy(ex)
            plot_data["amp"] = np.copy(amp)
            plot_data["phase"] = np.copy(phase)

fig = plt.figure(figsize=(8,7))
fig.suptitle(r"The Fourier Transform has been added")

def plotting_ex(data, ga, time_step, label_ab):
    """plot of E field at a single time step"""
    ax.plot(data, color="k", linewidth=1)
    ax.plot(-(ga - 1)/0.75, 'k--', linewidth=0.75)
    ax.set(xlim=(0, 200), ylim=(-1.2, 1.2), ylabel=r"E$_x$")
    ax.set(xticks=np.arange(0,220,20), yticks=np.arange(-1.2, 1.4, 0.5))
    ax.text(35, 0.3, "Time Domain, T = {}".format(time_step), horizontalalignment="center")
    ax.text(-25, -2.1, label_ab, horizontalalignment="center")
    return

def plotting_amp(data, ga, freq, label):
    """plot of the Fourier transform amplitude at a single time step"""
    ax.plot(data[0], color='k', linewidth=1)
    ax.plot(-(ga - 1)/0.75, 'k--', linewidth=0.75)
    ax.set(xlim=(0,200), ylim=(0,2), xlabel=r"{}".format(label), ylabel=r"Amp")
    ax.set(xticks=np.arange(0,220,20), yticks=np.arange(0,3,1))
    ax.text(150, 1.2, "Freq. Domain at {} MHz".format(int(round(freq[0]/1e6))), horizontalalignment="center")
    return


for subplot_num, plot_data in enumerate(points, start=0):
    ax = fig.add_subplot(4, 1, subplot_num * 2 + 1)
    plotting_ex(plot_data["ex"], gax, plot_data["num_steps"], plot_data["label_ab"])

    ax = fig.add_subplot(4, 1, subplot_num * 2 + 2)
    plotting_amp(plot_data["amp"], gax, freq, plot_data["label"])


plt.subplots_adjust(bottom=0.1, hspace=0.45)
plt.tight_layout()
plt.savefig("fd1d_2_2.png")
