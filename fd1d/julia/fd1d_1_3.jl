#!/usr/bin/env julia
# File: fd1d_1_3.jl
# Name: D.Saravanan
# Date: 25/02/2025

""" Simulation of a pulse hitting a dielectric medium """

import PyPlot as plt

plt.matplotlib.style.use("classic")
plt.matplotlib.style.use("../pyplot.mplstyle")


function visualize(ns::Int, nx::Int, epsr::Float64, cb::Vector{Float64}, ex::Vector{Float64})::Nothing
    fig, ax = plt.subplots(figsize=(8,3), gridspec_kw=Dict("hspace" => 0.2))
    fig.suptitle(raw"FDTD simulation of a pulse striking dielectric material")
    medium = epsr > 1 ? (div.(0.5,cb) .- 1)/(epsr - 1)*1e3 : div.(0.5,cb) .- 1
    medium[medium.==0] .= -1e3
    ax.plot(ex, color="black", linewidth=1)
    ax.fill_between(0:nx-1, medium, medium[1], color="y", alpha=0.3)
    ax.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    ax.set(xticks=0:round(Int, div(nx,10)/10)*10:nx)
    ax.set(xlabel=raw"$z\;(cm)$", ylabel=raw"$E_x\;(V/m)$")
    ax.text(0.02, 0.90, raw"$T$ = "*"$ns", transform=ax.transAxes)
    ax.text(0.90, 0.90, raw"$\epsilon_r$ = "*"$epsr", transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("fd1d_1_3.png", dpi=100)
end


function gaussian(t::Int32, t0::Int, sigma::Float64)::Float64
    return exp(-0.5 * ((t - t0)/sigma)^2)
end


function dielectric(nx::Int, epsr::Float64)::Vector{Float64}
    cb = 0.5 * ones(Float64, nx)
    cb[div(nx,2)+1:nx] .= 0.5/epsr
    return cb
end


function main()

    nx::Int = 512  # number of grid points
    ns::Int = 740  # number of time steps

    ex = zeros(Float64, nx)
    hy = zeros(Float64, nx)

    bc = zeros(Float64, 4)

    epsr::Float64 = 4.0  # relative permittivity
    cb::Vector{Float64} = dielectric(nx, epsr)

    for t in Int32.(1:ns)
        # calculate the Ex field
        ex[2:nx] .+= cb[2:nx] .* (hy[1:nx-1] - hy[2:nx])
        # put a Gaussian pulse in the middle
        ex[2] += gaussian(t, 40, 12.0)
        # absorbing boundary conditions
        ex[1], bc[1], bc[2] = bc[1], bc[2], ex[2]
        ex[nx], bc[4], bc[3] = bc[4], bc[3], ex[nx-1]
        # calculate the Hy field
        hy[1:nx-1] .+= 0.5 * (ex[1:nx-1] - ex[2:nx])
    end

    visualize(ns, nx, epsr, cb, ex)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
