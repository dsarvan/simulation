#!/usr/bin/env julia
# File: fdtd14.jl
# Name: D.Saravanan
# Date: 26/02/2025

""" Simulation of a propagating sinusoidal striking a dielectric medium """

import PyPlot as pyplot

pyplot.matplotlib.style.use("classic")
pyplot.matplotlib.style.use("pyplot.mplstyle")

const F64Array = Array{Float64}


function visualize(ns::Int, nx::Int, epsr::Float64, cb::F64Array, ex::F64Array)
    fig, axs = pyplot.subplots(figsize=(8,3), frameon=true)
    fig.suptitle("FDTD simulation of a sinusoidal striking dielectric material")
    medium = epsr > 1 ? findall(x->x!=0,0.5./cb.-1) : 0.5./cb.-1
    axs.plot(0:nx-1, ex, color="k", linewidth=1.0)
    axs.axvspan(medium[1], medium[end], color="y", alpha=0.3)
    axs.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    axs.set(xticks=0:Int(ceil(nx/500)*25):nx)
    axs.set(xlabel="\$z\\;(cm)\$", ylabel="\$E_x\\;(V/m)\$")
    axs.text(0.02, 0.90, "\$T\$ = $ns", transform=axs.transAxes)
    axs.text(0.90, 0.90, "\$\\epsilon_r\$ = $epsr", transform=axs.transAxes)
    fig.subplots_adjust(bottom=0.2, hspace=0.45)
    fig.savefig("fdtd14.png", dpi=100)
end


function sinusoidal(ts::Int, ds::Float64, freq::Float64)
    dt::Float64 = ds/6e8  # time step (s)
    return sin(2*pi*freq*dt*ts)
end


function dielectric(nx::Int, epsr::Float64)
    cb = fill(0.5::Float64, nx)
    cb[nx÷2+1:nx] .= 0.5/epsr
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
    cb::F64Array = dielectric(nx, epsr)

    for ts in map(Int,1:ns)
        # calculate the Ex field
        @views ex[2:nx] .+= cb[2:nx] .* (hy[1:nx-1] .- hy[2:nx])
        # put a sinusoidal wave at the low end
        ex[2] += sinusoidal(ts, 0.01, 700e6)
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
