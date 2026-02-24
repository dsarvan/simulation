#!/usr/bin/env julia
# File: test23.jl
# Name: D.Saravanan
# Date: 02/03/2025

""" Simulation of a pulse striking a frequency-dependent dielectric material and
implements the discrete Fourier transform with a Gaussian pulse as its source """

import PyPlot as pyplot
using Base:@kwdef
using Printf

pyplot.matplotlib.style.use("classic")
pyplot.matplotlib.style.use("pyplot.mplstyle")

const F32Array = Array{Float32}


function visualize(ns::Int, nx::Int, epsr::Float32, sigma::Float32, nax::F32Array, ex::F32Array)
    fig, axs = pyplot.subplots(figsize=(8,3), frameon=true)
    fig.suptitle("FDTD simulation of a pulse striking Debye dielectric material")
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
    fig.savefig("test23.png", dpi=100)
end


function amplitude(ns::Int, nx::Int, epsr::Float32, sigma::Float32, nax::F32Array, amp::F32Array)
    fig, axs = pyplot.subplots(figsize=(8,3), frameon=true)
    fig.suptitle("The discrete Fourier transform with pulse as its source")
    medium = epsr > 1 ? findall(x->x!=0,1.0./nax.-1) : 1.0./nax.-1
    axs.plot(0:nx-1, amp, color="k", linewidth=1.0)
    axs.axvspan(medium[1], medium[end], color="y", alpha=0.3)
    axs.set(xlim=(0, nx-1), ylim=(-0.2, 2.2))
    axs.set(xticks=0:Int(ceil(nx/500)*50):nx)
    axs.set(xlabel="\$z\\;(cm)\$", ylabel="\$Amplitude\$")
    axs.text(0.02, 0.90, "\$T\$ = $ns", transform=axs.transAxes)
    axs.text(0.90, 0.90, "\$\\epsilon_r\$ = $epsr", transform=axs.transAxes)
    axs.text(0.85, 0.80, "\$\\sigma\$ = $sigma \$S/m\$", transform=axs.transAxes)
    fig.subplots_adjust(bottom=0.2, hspace=0.45)
    fig.savefig("testamplt23.png", dpi=100)
end


@kwdef struct FTrans
    rpt::F32Array; ipt::F32Array;
    rin::F32Array; iin::F32Array;
end


@kwdef struct Medium
    nax::F32Array; nbx::F32Array;
    ncx::F32Array; ndx::F32Array;
end


function gaussian(ts::Int, t0::Int, sigma::Float32)
    return exp(-0.5*((ts - t0)/sigma)^2)
end


function fourier(ts::Int, nf::Int, nx::Int, dt::Float32, freq::F32Array, ex::F32Array, ft::FTrans)
    # calculate the Fourier transform of input source
    @views ft.rin[1:nf] .+= cos.(2*pi*freq[1:nf]*dt*ts) .* ex[11]
    @views ft.iin[1:nf] .-= sin.(2*pi*freq[1:nf]*dt*ts) .* ex[11]
    # calculate the Fourier transform of Ex field
    @views ft.rpt[1:nf,1:nx] .+= cos.(2*pi*freq[1:nf]*dt*ts) .* ex[1:nx]'
    @views ft.ipt[1:nf,1:nx] .-= sin.(2*pi*freq[1:nf]*dt*ts) .* ex[1:nx]'
end


function dxfield(ts::Int, nx::Int, hy::F32Array, dx::F32Array)
    # calculate the electric flux density Dx
    @views dx[2:nx] .+= 0.5 .* (hy[1:nx-1] .- hy[2:nx])
    # put a Gaussian pulse at the low end
    dx[2] += gaussian(ts, 50, 10.0f0)
end


function exfield(nx::Int, md::Medium, ix::F32Array, sx::F32Array, dx::F32Array, ex::F32Array)
    # calculate the Ex field from Dx
    @views ex[2:nx] .= md.nax[2:nx] .* (dx[2:nx] .- ix[2:nx] .- md.ncx[2:nx] .* sx[2:nx])
    @views ix[2:nx] .+= md.nbx[2:nx] .* ex[2:nx]
    @views sx[2:nx] .= md.ncx[2:nx] .* sx[2:nx] .+ md.ndx[2:nx] .* ex[2:nx]
end


function hyfield(nx::Int, bc::F32Array, ex::F32Array, hy::F32Array)
    # absorbing boundary conditions
    ex[1], bc[1], bc[2] = bc[1], bc[2], ex[2]
    ex[nx], bc[4], bc[3] = bc[4], bc[3], ex[nx-1]
    # calculate the Hy field
    @views hy[1:nx-1] .+= 0.5 .* (ex[1:nx-1] .- ex[2:nx])
end


function dielectric(nx::Int, dt::Float32, chi::Float32, tau::Float32, epsr::Float32, sigma::Float32)
    md = Medium(
        nax = fill(1.0f0::Float32, nx),
        nbx = fill(0.0f0::Float32, nx),
        ncx = fill(0.0f0::Float32, nx),
        ndx = fill(0.0f0::Float32, nx),
    )
    eps0::Float32 = 8.854e-12  # vaccum permittivity (F/m)
    md.nax[nx÷2+1:nx] .= 1/(epsr + sigma*dt/eps0 + chi*dt/tau)
    md.nbx[nx÷2+1:nx] .= sigma*dt/eps0
    md.ncx[nx÷2+1:nx] .= exp(-dt/tau)
    md.ndx[nx÷2+1:nx] .= chi*dt/tau
    return md
end


function main()

    nx::Int = 38000  # number of grid points
    ns::Int = 40000  # number of time steps

    dx = zeros(Float32, nx)
    ex = zeros(Float32, nx)
    ix = zeros(Float32, nx)
    sx = zeros(Float32, nx)
    hy = zeros(Float32, nx)

    bc = zeros(Float32, 4)

    ds::Float32 = 0.01  # spatial step (m)
    dt::Float32 = ds/6e8  # time step (s)
    chi::Float32 = 2.0  # relaxation susceptibility
    tau::Float32 = 0.001e-6  # relaxation time (s)
    epsr::Float32 = 2.0  # relative permittivity
    sigma::Float32 = 0.01  # conductivity (S/m)
    md::Medium = dielectric(nx, dt, chi, tau, epsr, sigma)

    # frequency 50 MHz, 200 MHz, 500 MHz
    freq = Float32[50e6, 200e6, 500e6]
    nf::Int = length(freq)  # number of frequencies

    ft = FTrans(
        rpt = zeros(Float32, (nf,nx)),
        ipt = zeros(Float32, (nf,nx)),
        rin = zeros(Float32, (nf, 1)),
        iin = zeros(Float32, (nf, 1)),
    )

    amplt = zeros(Float32, (nf,nx))
    phase = zeros(Float32, (nf,nx))

    stime = time_ns()

    for ts in map(Int,1:ns)
        dxfield(ts, nx, hy, dx)
        exfield(nx, md, ix, sx, dx, ex)
        fourier(ts, nf, nx, dt, freq, ex, ft)
        hyfield(nx, bc, ex, hy)
    end

    # calculate the amplitude and phase at each frequency
    amplt = hypot.(ft.rin,ft.iin).\1 .* hypot.(ft.rpt,ft.ipt)
    phase = atan.(ft.ipt,ft.rpt) .- atan.(ft.iin,ft.rin)

    ntime = time_ns()
    @printf("Total compute time on CPU: %.3f s\n", (ntime - stime)/1e9)

    println(ex[1:50])
    visualize(ns, nx, epsr, sigma, md.nax, ex)
    amplitude(ns, nx, epsr, sigma, md.nax, amplt[3,:])
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
