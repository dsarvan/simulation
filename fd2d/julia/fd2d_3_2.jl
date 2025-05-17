#!/usr/bin/env julia
# File: fd2d_3_2.jl
# Name: D.Saravanan
# Date: 04/03/2025

""" Simulation of a propagating sinusoidal in free space in the transverse
magnetic (TM) mode with the two-dimensional perfectly matched layer (PML) """

import PyPlot as plt
using Base.Threads
using LoopVectorization

plt.matplotlib.style.use("classic")
plt.matplotlib.style.use("../pyplot.mplstyle")
meshgrid(xs, ys) = xs .* ones(length(ys))', ones(length(xs)) .* ys'


function surfaceplot(ns::Int, nx::Int, ny::Int, ez::Array{Float64})::Nothing
    fig, ax = plt.subplots(subplot_kw=Dict("projection" => "3d"))
    fig.suptitle(raw"FDTD simulation of a sinusoidal in free space with PML")
    xv, yv = meshgrid(0:ny-1, 0:nx-1)
    ax.plot_surface(yv, xv, ez', rstride=1, cstride=1, cmap="gray", lw=0.25)
    ax.text2D(0.1, 0.7, raw"$T$ = "*"$ns", transform=ax.transAxes)
    ax.set(xlim=(0, nx), ylim=(0, ny), zlim=(0, 1))
    ax.set(xlabel=raw"$x\;(cm)$", ylabel=raw"$y\;(cm)$", zlabel=raw"$E_z\;(V/m)$")
    ax.zaxis.set_rotate_label(false); ax.view_init(elev=20.0, azim=45)
    plt.savefig("fd2d_surface_3_2.png", dpi=100)
end


function contourplot(ns::Int, nx::Int, ny::Int, ez::Array{Float64})::Nothing
    fig, ax = plt.subplots(figsize=(4,4), gridspec_kw=Dict("hspace" => 0.2))
    fig.suptitle(raw"FDTD simulation of a sinusoidal in free space with PML")
    xv, yv = meshgrid(0:ny-1, 0:nx-1)
    ax.contourf(yv, xv, ez', cmap="gray", alpha=0.75)
    ax.contour(yv, xv, ez', colors="k", linewidths=0.25)
    ax.set(xlim=(0, nx-1), ylim=(0, ny-1), aspect="equal")
    ax.set(xlabel=raw"$x\;(cm)$", ylabel=raw"$y\;(cm)$")
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("fd2d_contour_3_2.png", dpi=100)
end


struct pmlayer
    fx1::Array{Float64}
    fx2::Array{Float64}
    fx3::Array{Float64}
    fy1::Array{Float64}
    fy2::Array{Float64}
    fy3::Array{Float64}
    gx2::Array{Float64}
    gx3::Array{Float64}
    gy2::Array{Float64}
    gy3::Array{Float64}
end


function sinusoidal(t::Int32, ds::Float64, freq::Float64)::Float64
    dt::Float64 = ds/6e8  # time step (s)
    return sin(2 * pi * freq * dt * t)
end


function pmlparam(npml::Int, nx::Int, ny::Int, pml::pmlayer)
    """ calculate the two-dimensional perfectly matched layer (PML) parameters """
    for n in 1:npml
        xm = 0.33 * ((npml-n+1)/npml)^3
        xn = 0.33 * ((npml-n+1-0.5)/npml)^3
        pml.fx1[n] = pml.fx1[nx-n] = pml.fy1[n] = pml.fy1[ny-n] = xn
        pml.fx2[n] = pml.fx2[nx-n] = pml.fy2[n] = pml.fy2[ny-n] = 1/(1+xn)
        pml.gx2[n] = pml.gx2[nx+1-n] = pml.gy2[n] = pml.gy2[ny+1-n] = 1/(1+xm)
        pml.fx3[n] = pml.fx3[nx-n] = pml.fy3[n] = pml.fy3[ny-n] = (1-xn)/(1+xn)
        pml.gx3[n] = pml.gx3[nx+1-n] = pml.gy3[n] = pml.gy3[ny+1-n] = (1-xm)/(1+xm)
    end
end


function dfield(t::Int32, nx::Int, ny::Int, pml::pmlayer, dz::Array{Float64}, hx::Array{Float64}, hy::Array{Float64})
    """ calculate the electric flux density Dz """
    @threads for j in 2:ny
        @turbo for i in 2:nx
            dz[i,j] = pml.gx3[i] * pml.gy3[j] * dz[i,j] + pml.gx2[i] * pml.gy2[j] * 0.5 * (hy[i,j] - hy[i-1,j] - hx[i,j] + hx[i,j-1])
        end
    end
    # put a sinusoidal source at a point that is offset four cells
    # from the center of the problem space in each direction
    dz[div(nx,2)-4,div(ny,2)-4] = sinusoidal(t, 0.01, 1500e6)
end


function efield(nx::Int, ny::Int, naz::Array{Float64}, dz::Array{Float64}, ez::Array{Float64})
    """ calculate the Ez field from Dz """
    @threads for j in 1:ny
        @turbo for i in 1:nx
            ez[i,j] = naz[i,j] * dz[i,j]
        end
    end
end


function hfield(nx::Int, ny::Int, pml::pmlayer, ez::Array{Float64}, ihx::Array{Float64}, ihy::Array{Float64}, hx::Array{Float64}, hy::Array{Float64})
    """ calculate the Hx and Hy field """
    @threads for j in 1:ny-1
        @turbo for i in 1:nx-1
            ihx[i,j] += ez[i,j] - ez[i,j+1]
            ihy[i,j] += ez[i,j] - ez[i+1,j]
            hx[i,j] = pml.fy3[j] * hx[i,j] + pml.fy2[j] * (0.5 * ez[i,j] - 0.5 * ez[i,j+1] + pml.fx1[i] * ihx[i,j])
            hy[i,j] = pml.fx3[i] * hy[i,j] - pml.fx2[i] * (0.5 * ez[i,j] - 0.5 * ez[i+1,j] + pml.fy1[j] * ihy[i,j])
        end
    end
end


function main()

    nx::Int = 60  # number of grid points
    ny::Int = 60  # number of grid points

    ns::Int = 100  # number of time steps

    dz = zeros(Float64, (nx, ny))
    ez = zeros(Float64, (nx, ny))
    hx = zeros(Float64, (nx, ny))
    hy = zeros(Float64, (nx, ny))

    ihx = zeros(Float64, (nx, ny))
    ihy = zeros(Float64, (nx, ny))

    naz = ones(Float64, (nx, ny))

    ds::Float64 = 0.01  # spatial step (m)
    dt::Float64 = ds/6e8  # time step (s)

    pml = pmlayer(
        zeros(Float64, nx),
        ones(Float64, nx),
        ones(Float64, nx),
        zeros(Float64, ny),
        ones(Float64, ny),
        ones(Float64, ny),
        ones(Float64, nx),
        ones(Float64, nx),
        ones(Float64, ny),
        ones(Float64, ny),
    )

    npml::Int = 8  # pml thickness
    pmlparam(npml, nx, ny, pml)

    for t in Int32.(1:ns)
        dfield(t, nx, ny, pml, dz, hx, hy)
        efield(nx, ny, naz, dz, ez)
        hfield(nx, ny, pml, ez, ihx, ihy, hx, hy)
    end

    surfaceplot(ns, nx, ny, ez)
    contourplot(ns, nx, ny, ez)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
