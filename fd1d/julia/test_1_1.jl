#!/usr/bin/env julia
# File: test_1_1.jl
# Name: D.Saravanan
# Date: 23/02/2025

""" Simulation of a pulse in free space """

import PyPlot as plt
using Printf

plt.matplotlib.style.use("classic")
plt.matplotlib.style.use("../pyplot.mplstyle")


function visualize(ns::Int, nx::Int, ex::Array{Float32})::Nothing
    fig, ax = plt.subplots(figsize=(8,3), gridspec_kw=Dict("hspace"=>0.2))
    fig.suptitle(raw"FDTD simulation of a pulse in free space")
    ax.plot(ex, color="black", linewidth=1)
    ax.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    ax.set(xticks=0:round(Int, div(nx,10)/10)*10:nx)
    ax.set(xlabel=raw"$z\;(cm)$", ylabel=raw"$E_x\;(V/m)$")
    ax.text(0.02, 0.90, raw"$T$ = "*"$ns", transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.show()
end


function gaussian(t::Int32, t0::Int, sigma::Float32)::Float32
    return exp(-0.5*((t-t0)/sigma)^2)
end


function main()

    nx::Int = 38000  # number of grid points
    ns::Int = 40000  # number of time steps

    ex = zeros(Float32, nx)
    hy = zeros(Float32, nx)

    stime = time_ns()

    for t in Int32.(1:ns)
        # calculate the Ex field
        @views ex[2:nx] .+= 0.5 .* (hy[1:nx-1] .- hy[2:nx])
        # put a Gaussian pulse in the middle
        ex[div(nx,2)+1] = gaussian(t, 40, 12.0f0)
        # calculate the Hy field
        @views hy[1:nx-1] .+= 0.5 .* (ex[1:nx-1] .- ex[2:nx])
    end

    ntime = time_ns()
    @printf("Total compute time on CPU: %.3f s\n", (ntime - stime)/1e9)

    println(ex[1:50])
    visualize(ns, nx, ex)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
