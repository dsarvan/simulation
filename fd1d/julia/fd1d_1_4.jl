#!/usr/bin/env julia
# File: fd1d_1_4.jl
# Name: D.Saravanan
# Date: 26/02/2025

""" Simulation of a propagating sinusoidal striking a dielectric medium """

import PyPlot as plt

plt.matplotlib.style.use("classic")
plt.matplotlib.style.use("../pyplot.mplstyle")


function visualize(ns::Int, nx::Int, epsr::Float64, cb::Array{Float64}, ex::Array{Float64})::Nothing
    fig, ax = plt.subplots(figsize=(8,3), gridspec_kw=Dict("hspace"=>0.2))
    fig.suptitle(raw"FDTD simulation of a sinusoidal striking dielectric material")
    medium = epsr > 1 ? (div.(0.5,cb).-1)/(epsr-1)*1e3 : div.(0.5,cb).-1
    medium[medium.==0] .= -1e3
    ax.plot(ex, color="black", linewidth=1)
    ax.fill_between(0:nx-1, medium, medium[1], color="y", alpha=0.3)
    ax.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    ax.set(xticks=0:round(Int, div(nx,10)/10)*10:nx)
    ax.set(xlabel=raw"$z\;(cm)$", ylabel=raw"$E_x\;(V/m)$")
    ax.text(0.02, 0.90, raw"$T$ = "*"$ns", transform=ax.transAxes)
    ax.text(0.90, 0.90, raw"$\epsilon_r$ = "*"$epsr", transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("fd1d_1_4.png", dpi=100)
end


function sinusoidal(t::Int32, ds::Float64, freq::Float64)::Float64
    dt::Float64 = ds/6e8  # time step (s)
    return sin(2*pi*freq*dt*t)
end


function dielectric(nx::Int, epsr::Float64)::Array{Float64}
    cb = 0.5 .+ zeros(Float64, nx)
    cb[div(nx,2)+1:nx] .= 0.5/epsr
    return cb
end


function main()

    nx::Int = 512  # number of grid points
    ns::Int = 740  # number of time steps

    ex = zeros(Float64, nx)
    hy = zeros(Float64, nx)

    bc = zeros(Float64, 4)

    ds::Float64 = 0.01  # spatial step (m)
    dt::Float64 = ds/6e8  # time step (s)
    epsr::Float64 = 4.0  # relative permittivity
    cb::Array{Float64} = dielectric(nx, epsr)

    for t in Int32.(1:ns)
        # calculate the Ex field
        @views ex[2:nx] .+= cb[2:nx] .* (hy[1:nx-1] .- hy[2:nx])
        # put a sinusoidal wave at the low end
        ex[2] += sinusoidal(t, 0.01, 700e6)
        # absorbing boundary conditions
        ex[1], bc[1], bc[2] = bc[1], bc[2], ex[2]
        ex[nx], bc[4], bc[3] = bc[4], bc[3], ex[nx-1]
        # calculate the Hy field
        @views hy[1:nx-1] .+= 0.5 .* (ex[1:nx-1] .- ex[2:nx])
    end

    visualize(ns, nx, epsr, cb, ex)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
