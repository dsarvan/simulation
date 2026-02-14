#!/usr/bin/env julia
# File: test15.jl
# Name: D.Saravanan
# Date: 27/02/2025

""" Simulation of a propagating sinusoidal wave of 700 MHz striking a lossy
dielectric with a dielectric constant of 4 and conductivity of 0.04 (S/m) """

import PyPlot as pyplot
using Printf

pyplot.matplotlib.style.use("classic")
pyplot.matplotlib.style.use("pyplot.mplstyle")

const F32Array = Array{Float32}


function visualize(ns::Int, nx::Int, epsr::Float32, sigma::Float32, cb::F32Array, ex::F32Array)
    fig, axs = pyplot.subplots(figsize=(8,3), frameon=true)
    fig.suptitle("FDTD simulation of a sinusoidal striking lossy dielectric material")
    medium = epsr > 1 ? findall(x->x!=0,0.5./cb.-1) : 0.5./cb.-1
    axs.plot(0:nx-1, ex, color="k", linewidth=1.0)
    axs.axvspan(medium[1], medium[end], color="y", alpha=0.3)
    axs.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    axs.set(xticks=0:Int(ceil(nx/500)*50):nx)
    axs.set(xlabel="\$z\\;(cm)\$", ylabel="\$E_x\\;(V/m)\$")
    axs.text(0.02, 0.90, "\$T\$ = $ns", transform=axs.transAxes)
    axs.text(0.90, 0.90, "\$\\epsilon_r\$ = $epsr", transform=axs.transAxes)
    axs.text(0.85, 0.80, "\$\\sigma\$ = $sigma \$S/m\$", transform=axs.transAxes)
    fig.subplots_adjust(bottom=0.2, hspace=0.45)
    fig.savefig("test15.png", dpi=100)
end


function sinusoidal(ts::Int, ds::Float32, freq::Float32)
    dt::Float32 = ds/6e8  # time step (s)
    return sin(2*pi*freq*dt*ts)
end


function dielectric(nx::Int, dt::Float32, epsr::Float32, sigma::Float32)
    ca = fill(1.0f0::Float32, nx)
    cb = fill(0.5f0::Float32, nx)
    eps0::Float32 = 8.854e-12  # vaccum permittivity (F/m)
    epsf::Float32 = dt*sigma/(2*eps0*epsr)
    ca[nx÷2+1:nx] .= (1 - epsf)/(1 + epsf)
    cb[nx÷2+1:nx] .= 0.5/(epsr*(1 + epsf))
    return ca, cb
end


function main()

    nx::Int = 38000  # number of grid points
    ns::Int = 40000  # number of time steps

    ex = zeros(Float32, nx)
    hy = zeros(Float32, nx)

    bc = zeros(Float32, 4)

    ds::Float32 = 0.01  # spatial step (m)
    dt::Float32 = ds/6e8  # time step (s)
    epsr::Float32 = 4.0  # relative permittivity
    sigma::Float32 = 0.04  # conductivity (S/m)
    ca, cb = dielectric(nx, dt, epsr, sigma)

    stime = time_ns()

    for ts in map(Int,1:ns)
        # calculate the Ex field
        @views ex[2:nx] .= ca[2:nx] .* ex[2:nx] .+ cb[2:nx] .* (hy[1:nx-1] .- hy[2:nx])
        # put a sinusoidal wave at the low end
        ex[2] += sinusoidal(ts, 0.01f0, 700f6)
        # absorbing boundary conditions
        ex[1], bc[1], bc[2] = bc[1], bc[2], ex[2]
        ex[nx], bc[4], bc[3] = bc[4], bc[3], ex[nx-1]
        # calculate the Hy field
        @views hy[1:nx-1] .+= 0.5 .* (ex[1:nx-1] .- ex[2:nx])
    end

    ntime = time_ns()
    @printf("Total compute time on CPU: %.3f s\n", (ntime - stime)/1e9)

    println(ex[1:50])
    visualize(ns, nx, epsr, sigma, cb, ex)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
