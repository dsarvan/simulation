#!/usr/bin/env python
# File: fdtd.py
# Name: D.Saravanan
# Date: 21/05/2022

""" Script for 1-dimensional fdtd with gaussian pulse as source """

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

plt.style.use("classic")
plt.rcParams["text.usetex"] = True
plt.rcParams["pgf.texsystem"] = "pdflatex"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "figure.titlesize": 10,
    }
)

def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ke = 200

    # Pulse parameters
    kc = int(ke/2)
    t0 = 40
    spread = 12
    nsteps = 100

    # determine the workload of each rank
    workloads = [ ke // size for i in range(size) ]
    for i in range( ke % size ):
        workloads[i] += 1
    start = 0
    for i in range(rank):
        start += workloads[i]
    end = start + workloads[rank]


    #timeloads = [ nsteps // size for i in range(size) ]
    #for i in range( nsteps % size ):
    #    timeloads[i] += 1
    #start = 0
    #for i in range(rank):
    #    start += timeloads[i]
    #end = start + timeloads[rank]


    N = workloads[rank]
    
    ex = np.zeros(N)
    hy = np.zeros(N)
    
    for time_step in range(1, nsteps + 1):

        ex[1:N] = ex[1:N] + 0.5 * (hy[0:N-1] - hy[1:N])
        ex[10] = np.exp(-0.5 * ((t0 - time_step) / spread) ** 2)
        hy[0:N-1] = hy[0:N-1] + 0.5 * (ex[0:N-1] - ex[1:N])


    comm.Bcast(ex, root=0)
    comm.Bcast(hy, root=0)


    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(r"FDTD simulation of a pulse in free space after 100 time steps")
    ax1.plot(ex, "k", lw=1)
    ax1.text(100, 0.5, "T = {}".format(time_step), horizontalalignment="center")
    ax1.set(xlim=(0, 200), ylim=(-1.2, 1.2), ylabel=r"E$_x$")
    ax1.set(xticks=range(0, 220, 20), yticks=np.arange(-1, 1.2, 1))
    ax2.plot(hy, "k", lw=1)
    ax2.set(xlim=(0, 200), ylim=(-1.2, 1.2), xlabel=r"FDTD cells", ylabel=r"H$_y$")
    ax2.set(xticks=range(0, 220, 20), yticks=np.arange(-1, 1.2, 1))
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("fdtd.png")



if __name__ == "__main__":
    main()
