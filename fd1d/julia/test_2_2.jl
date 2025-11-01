#!/usr/bin/env julia
# File: test_2_2.jl
# Name: D.Saravanan
# Date: 01/03/2025

""" Simulation of a pulse striking a dielectric medium and implements
the discrete Fourier transform with a Gaussian pulse as its source """

import PyPlot as plt
using Printf

plt.matplotlib.style.use("classic")
plt.matplotlib.style.use("../pyplot.mplstyle")


function visualize(ns::Int, nx::Int, epsr::Float32, sigma::Float32, nax::Array{Float32}, ex::Array{Float32})::Nothing
    fig, ax = plt.subplots(figsize=(8,3), gridspec_kw=Dict("hspace"=>0.2))
    fig.suptitle("FDTD simulation of a pulse striking dielectric material")
    medium = epsr > 1 ? findall(x->x!=0,1.0./nax.-1) : 1.0./nax.-1
    ax.plot(0:nx-1, ex, color="k", linewidth=1.0)
    ax.axvspan(medium[1], medium[end], color="y", alpha=0.3)
    ax.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    ax.set(xticks=0:Int(ceil(nx/500)*50):nx)
    ax.set(xlabel="\$z\\;(cm)\$", ylabel="\$E_x\\;(V/m)\$")
    ax.text(0.02, 0.90, "\$T\$ = $ns", transform=ax.transAxes)
    ax.text(0.90, 0.90, "\$\\epsilon_r\$ = $epsr", transform=ax.transAxes)
    ax.text(0.85, 0.80, "\$\\sigma\$ = $sigma \$S/m\$", transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("test_2_2.png", dpi=100)
end


function amplitude(ns::Int, nx::Int, epsr::Float32, sigma::Float32, nax::Array{Float32}, amp::Array{Float32})::Nothing
    fig, ax = plt.subplots(figsize=(8,3), gridspec_kw=Dict("hspace"=>0.2))
    fig.suptitle("The discrete Fourier transform with pulse as its source")
    medium = epsr > 1 ? findall(x->x!=0,1.0./nax.-1) : 1.0./nax.-1
    ax.plot(0:nx-1, amp, color="k", linewidth=1.0)
    ax.axvspan(medium[1], medium[end], color="y", alpha=0.3)
    ax.set(xlim=(0, nx-1), ylim=(-0.2, 2.2))
    ax.set(xticks=0:Int(ceil(nx/500)*50):nx)
    ax.set(xlabel="\$z\\;(cm)\$", ylabel="\$Amplitude\$")
    ax.text(0.02, 0.90, "\$T\$ = $ns", transform=ax.transAxes)
    ax.text(0.90, 0.90, "\$\\epsilon_r\$ = $epsr", transform=ax.transAxes)
    ax.text(0.85, 0.80, "\$\\sigma\$ = $sigma \$S/m\$", transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("test_amp_2_2.png", dpi=100)
end


struct medium
    nax::Array{Float32}
    nbx::Array{Float32}
end


struct ftrans
    r_pt::Array{Float32}
    i_pt::Array{Float32}
    r_in::Array{Float32}
    i_in::Array{Float32}
end


function gaussian(t::Int32, t0::Int, sigma::Float32)::Float32
    return exp(-0.5*((t - t0)/sigma)^2)
end


function fourier(t::Int32, nf::Int, nx::Int, dt::Float32, freq::Array{Float32}, ex::Array{Float32}, ft::ftrans)
    # calculate the Fourier transform of input source
    @views ft.r_in[1:nf] .+= cos.(2*pi*freq[1:nf]*dt*t) .* ex[11]
    @views ft.i_in[1:nf] .-= sin.(2*pi*freq[1:nf]*dt*t) .* ex[11]
    # calculate the Fourier transform of Ex field
    @views ft.r_pt[1:nf,1:nx] .+= cos.(2*pi*freq[1:nf]*dt*t) .* ex[1:nx]'
    @views ft.i_pt[1:nf,1:nx] .-= sin.(2*pi*freq[1:nf]*dt*t) .* ex[1:nx]'
end


function dxfield(t::Int32, nx::Int, dx::Array{Float32}, hy::Array{Float32})
    # calculate the electric flux density Dx
    @views dx[2:nx] .+= 0.5 .* (hy[1:nx-1] .- hy[2:nx])
    # put a Gaussian pulse at the low end
    dx[2] += gaussian(t, 50, 10.0f0)
end


function exfield(nx::Int, md::medium, dx::Array{Float32}, ix::Array{Float32}, ex::Array{Float32})
    # calculate the Ex field from Dx
    @views ex[2:nx] .= md.nax[2:nx] .* (dx[2:nx] .- ix[2:nx])
    @views ix[2:nx] .+= md.nbx[2:nx] .* ex[2:nx]
end


function hyfield(nx::Int, ex::Array{Float32}, hy::Array{Float32}, bc::Array{Float32})
    # absorbing boundary conditions
    ex[1], bc[1], bc[2] = bc[1], bc[2], ex[2]
    ex[nx], bc[4], bc[3] = bc[4], bc[3], ex[nx-1]
    # calculate the Hy field
    @views hy[1:nx-1] .+= 0.5 .* (ex[1:nx-1] .- ex[2:nx])
end


function dielectric(nx::Int, dt::Float32, epsr::Float32, sigma::Float32)::medium
    md = medium(
        fill(1.0f0::Float32, nx),
        fill(0.0f0::Float32, nx),
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
    sigma::Float32 = 0.0  # conductivity (S/m)
    md::medium = dielectric(nx, dt, epsr, sigma)

    # frequency 100 MHz, 200 MHz, 500 MHz
    freq = Float32[100e6, 200e6, 500e6]
    nf::Int = length(freq)  # number of frequencies

    ft = ftrans(
        zeros(Float32, (nf, nx)),
        zeros(Float32, (nf, nx)),
        zeros(Float32, (nf, 1)),
        zeros(Float32, (nf, 1)),
    )

    amplt = zeros(Float32, (nf, nx))
    phase = zeros(Float32, (nf, nx))

    stime = time_ns()

    for t in Int32.(1:ns)
        dxfield(t, nx, dx, hy)
        exfield(nx, md, dx, ix, ex)
        fourier(t, nf, nx, dt, freq, ex, ft)
        hyfield(nx, ex, hy, bc)
    end

    # calculate the amplitude and phase at each frequency
    amplt = hypot.(ft.r_in,ft.i_in).\1 .* hypot.(ft.r_pt,ft.i_pt)
    phase = atan.(ft.i_pt,ft.r_pt) .- atan.(ft.i_in,ft.r_in)

    ntime = time_ns()
    @printf("Total compute time on CPU: %.3f s\n", (ntime - stime)/1e9)

    println(ex[1:50])
    visualize(ns, nx, epsr, sigma, md.nax, ex)
    amplitude(ns, nx, epsr, sigma, md.nax, amplt[3,:])
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
