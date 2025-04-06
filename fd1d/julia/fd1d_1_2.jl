#!/usr/bin/env julia
# File: fd1d_1_2.jl
# Name: D.Saravanan
# Date: 24/02/2025

""" Simulation of a pulse with absorbing boundary conditions """

import PyPlot as plt

plt.matplotlib.style.use("classic")
plt.matplotlib.style.use("../pyplot.mplstyle")


function visualize(ns::Int, nx::Int, ex::Array{Float64})::Nothing
    fig, ax = plt.subplots(figsize=(8,3), gridspec_kw=Dict("hspace" => 0.2))
    fig.suptitle(raw"FDTD simulation of a pulse with absorbing boundary conditions")
    ax.plot(ex, color="black", linewidth=1)
    ax.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    ax.set(xticks=0:round(Int, div(nx,10)/10)*10:nx)
    ax.set(xlabel=raw"$z\;(cm)$", ylabel=raw"$E_x\;(V/m)$")
    ax.text(0.02, 0.90, raw"$T$ = "*"$ns", transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("fd1d_1_2.png", dpi=100)
end


function gaussian(t::Int32, t0::Int, sigma::Float64)::Float64
    return exp(-0.5 * ((t - t0)/sigma)^2)
end


function main()

    nx::Int = 512  # number of grid points
    ns::Int = 570  # number of time steps

    ex = zeros(Float64, nx)
    hy = zeros(Float64, nx)

    bc = zeros(Float64, 4)

    for t in Int32.(1:ns)
        # calculate the Ex field
        @views ex[2:nx] .+= 0.5 .* (hy[1:nx-1] .- hy[2:nx])
        # put a Gaussian pulse in the middle
        ex[div(nx,2)+1] = gaussian(t, 40, 12.0)
        # absorbing boundary conditions
        ex[1], bc[1], bc[2] = bc[1], bc[2], ex[2]
        ex[nx], bc[4], bc[3] = bc[4], bc[3], ex[nx-1]
        # calculate the Hy field
        @views hy[1:nx-1] .+= 0.5 .* (ex[1:nx-1] .- ex[2:nx])
    end

    visualize(ns, nx, ex)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
