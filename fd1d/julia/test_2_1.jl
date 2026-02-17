#!/usr/bin/env julia
# File: test21.jl
# Name: D.Saravanan
# Date: 28/02/2025

""" Simulation of a propagating sinusoidal wave of 700 MHz striking a lossy
dielectric with a dielectric constant of 4 and conductivity of 0.04 (S/m) """

import PyPlot as pyplot
using Base:@kwdef
using Printf

pyplot.matplotlib.style.use("classic")
pyplot.matplotlib.style.use("pyplot.mplstyle")

const F32Array = Array{Float32}


function visualize(ns::Int, nx::Int, epsr::Float32, sigma::Float32, nax::F32Array, ex::F32Array)
    fig, axs = pyplot.subplots(figsize=(8,3), frameon=true)
    fig.suptitle("FDTD simulation of a sinusoidal striking lossy dielectric material")
    medium = epsr > 1 ? findall(x->x!=0,1.0./nax.-1) : 1.0./nax.-1
    axs.plot(0:nx-1, ex, color="k", linewidth=1.0)
    axs.axvspan(medium[1], medium[end], color="y", alpha=0.3)
    axs.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    axs.set(xticks=0:Int(ceil(nx/500)*50):nx)
    axs.set(xlabel="\$z\\;(cm)\$", ylabel="\$E_x\\;(V/m)\$")
    axs.text(0.02, 0.90, "\$T\$ = $ns", transform=axs.transAxes)
    axs.text(0.90, 0.90, "\$\\epsilon_r\$ = $epsr", transform=axs.transAxes)
    axs.text(0.85, 0.80, "\$\\sigma\$ = $sigma \$S/m\$", transform=axs.transAxes)
    fig.subplots_adjust(bottom=0.2, hspace=0.45)
    fig.savefig("test21.png", dpi=100)
end



@kwdef struct Medium
    nax::F32Array; nbx::F32Array;
end


function sinusoidal(ts::Int, ds::Float32, freq::Float32)
    dt::Float32 = ds/6e8  # time step (s)
    return sin(2*pi*freq*dt*ts)
end


function dxfield(ts::Int, nx::Int, hy::F32Array, dx::F32Array)
    # calculate the electric flux density Dx
    @views dx[2:nx] .+= 0.5 .* (hy[1:nx-1] .- hy[2:nx])
    # put a sinusoidal wave at the low end
    dx[2] += sinusoidal(ts, 0.01f0, 700f6)
end


function exfield(nx::Int, md::Medium, ix::F32Array, dx::F32Array, ex::F32Array)
    # calculate the Ex field from Dx
    @views ex[2:nx] .= md.nax[2:nx] .* (dx[2:nx] .- ix[2:nx])
    @views ix[2:nx] .+= md.nbx[2:nx] .* ex[2:nx]
end


function hyfield(nx::Int, bc::F32Array, ex::F32Array, hy::F32Array)
    # absorbing boundary conditions
    ex[1], bc[1], bc[2] = bc[1], bc[2], ex[2]
    ex[nx], bc[4], bc[3] = bc[4], bc[3], ex[nx-1]
    # calculate the Hy field
    @views hy[1:nx-1] .+= 0.5 .* (ex[1:nx-1] .- ex[2:nx])
end


function dielectric(nx::Int, dt::Float32, epsr::Float32, sigma::Float32)
    md = Medium(
        nax = fill(1.0f0::Float32, nx),
        nbx = fill(0.0f0::Float32, nx),
    )
    eps0::Float32 = 8.854e-12  # vaccum permittivity (F/m)
    md.nax[nx÷2+1:nx] .= 1/(epsr + sigma*dt/eps0)
    md.nbx[nx÷2+1:nx] .= sigma*dt/eps0
    return md
end


function main()

    nx::Int = 38000  # number of grid points
    ns::Int = 40000  # number of time steps

    dx = zeros(Float32, nx)
    ex = zeros(Float32, nx)
    ix = zeros(Float32, nx)
    hy = zeros(Float32, nx)

    bc = zeros(Float32, 4)

    ds::Float32 = 0.01  # spatial step (m)
    dt::Float32 = ds/6e8  # time step (s)
    epsr::Float32 = 4.0  # relative permittivity
    sigma::Float32 = 0.04  # conductivity (S/m)
    md::Medium = dielectric(nx, dt, epsr, sigma)

    stime = time_ns()

    for ts in map(Int,1:ns)
        dxfield(ts, nx, hy, dx)
        exfield(nx, md, ix, dx, ex)
        hyfield(nx, bc, ex, hy)
    end

    ntime = time_ns()
    @printf("Total compute time on CPU: %.3f s\n", (ntime - stime)/1e9)

    println(ex[1:50])
    visualize(ns, nx, epsr, sigma, md.nax, ex)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
