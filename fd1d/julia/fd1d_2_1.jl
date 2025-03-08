#!/usr/bin/env julia
# File: fd1d_2_1.jl
# Name: D.Saravanan
# Date: 28/02/2025

""" Simulation of a propagating sinusoidal wave of 700 MHz striking a lossy
dielectric with a dielectric constant of 4 and conductivity of 0.04 (S/m) """

import PyPlot as plt

plt.matplotlib.style.use("classic")
plt.matplotlib.style.use("../pyplot.mplstyle")


function visualize(ns::Int, nx::Int, epsr::Float64, sigma::Float64, nax::Vector{Float64}, ex::Vector{Float64})::Nothing
    fig, ax = plt.subplots(figsize=(8,3), gridspec_kw=Dict("hspace" => 0.2))
    fig.suptitle(raw"FDTD simulation of a sinusoidal striking lossy dielectric material")
    medium = epsr > 1 ? (1 .- nax)/(1 - nax[end])*1e3 : (1 .- nax)
    medium[medium.==0] .= -1e3
    ax.plot(ex, color="black", linewidth=1)
    ax.fill_between(0:nx-1, medium, medium[1], color="y", alpha=0.3)
    ax.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    ax.set(xticks=0:round(Int, div(nx,10)/10)*10:nx)
    ax.set(xlabel=raw"$z\;(cm)$", ylabel=raw"$E_x\;(V/m)$")
    ax.text(0.02, 0.90, raw"$T$ = "*"$ns", transform=ax.transAxes)
    ax.text(0.90, 0.90, raw"$\epsilon_r$ = "*"$epsr", transform=ax.transAxes)
    ax.text(0.85, 0.80, raw"$\sigma$ = "*"$sigma"*raw" $S/m$", transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("fd1d_2_1.png", dpi=100)
end


struct medium
    nax::Vector{Float64}
    nbx::Vector{Float64}
end


function sinusoidal(t::Int32, ds::Float64, freq::Float64)::Float64
    dt::Float64 = ds/6e8  # time step (s)
    return sin(2 * pi * freq * dt * t)
end


function dxfield(t::Int32, nx::Int, dx::Vector{Float64}, hy::Vector{Float64})
    # calculate the electric flux density Dx
    @views dx[2:nx] .+= 0.5 .* (hy[1:nx-1] .- hy[2:nx])
    # put a sinusoidal wave at the low end
    dx[2] += sinusoidal(t, 0.01, 700e6)
end


function exfield(nx::Int, md::medium, dx::Vector{Float64}, ix::Vector{Float64}, ex::Vector{Float64})
    # calculate the Ex field from Dx
    @views ex[2:nx] .= md.nax[2:nx] .* (dx[2:nx] .- ix[2:nx])
    @views ix[2:nx] .= ix[2:nx] .+ md.nbx[2:nx] .* ex[2:nx]
end


function hyfield(nx::Int, ex::Vector{Float64}, hy::Vector{Float64}, bc::Vector{Float64})
    # absorbing boundary conditions
    ex[1], bc[1], bc[2] = bc[1], bc[2], ex[2]
    ex[nx], bc[4], bc[3] = bc[4], bc[3], ex[nx-1]
    # calculate the Hy field
    @views hy[1:nx-1] .+= 0.5 .* (ex[1:nx-1] .- ex[2:nx])
end


function dielectric(nx::Int, dt::Float64, epsr::Float64, sigma::Float64)::medium
    md = medium(
        ones(Float64, nx),
        zeros(Float64, nx),
    )
    eps0::Float64 = 8.854e-12  # vaccum permittivity (F/m)
    md.nax[div(nx,2)+1:nx] .= 1/(epsr + (sigma * dt/eps0))
    md.nbx[div(nx,2)+1:nx] .= sigma * dt/eps0
    return md
end


function main()

    nx::Int = 512  # number of grid points
    ns::Int = 740  # number of time steps

    dx = zeros(Float64, nx)
    ex = zeros(Float64, nx)
    ix = zeros(Float64, nx)
    hy = zeros(Float64, nx)

    bc = zeros(Float64, 4)

    ds::Float64 = 0.01  # spatial step (m)
    dt::Float64 = ds/6e8  # time step (s)
    epsr::Float64 = 4.0  # relative permittivity
    sigma::Float64 = 0.04  # conductivity (S/m)
    md::medium = dielectric(nx, dt, epsr, sigma)

    for t in Int32.(1:ns)
        dxfield(t, nx, dx, hy)
        exfield(nx, md, dx, ix, ex)
        hyfield(nx, ex, hy, bc)
    end

    visualize(ns, nx, epsr, sigma, md.nax, ex)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
