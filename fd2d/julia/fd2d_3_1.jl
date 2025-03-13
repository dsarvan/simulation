#!/usr/bin/env julia
# File: fd2d_3_1.jl
# Name: D.Saravanan
# Date: 03/03/2025

""" Simulation of a pulse in free space in the transverse magnetic (TM) mode """

import PyPlot as plt
using Printf

plt.matplotlib.style.use("classic")
plt.matplotlib.style.use("../pyplot.mplstyle")


meshgrid(xs, ys) = [xs[i] for i in 1:length(xs), j in 1:length(ys)], [ys[j] for i in 1:length(xs), j in 1:length(ys)]


function surfaceplot(ns::Int, nx::Int, ny::Int, ez::Matrix{Float64})::Nothing
    fig, ax = plt.subplots(subplot_kw=Dict("projection" => "3d"))
    fig.suptitle(raw"FDTD simulation of a pulse in free space")
    xv, yv = meshgrid(0:nx-1, 0:ny-1)
    ax.plot_surface(xv, yv, ez', rstride=1, cstride=1, cmap="gray", lw=0.25)
    ax.text2D(0.8, 0.7, raw"$T$ = "*"$ns", transform=ax.transAxes)
    ax.set(xlim=(0, nx), ylim=(0, ny), zlim=(0, 1))
    ax.set(xlabel=raw"$x\;(cm)$", ylabel=raw"$y\;(cm)$", zlabel=raw"$E_z\;(V/m)$")
    ax.zaxis.set_rotate_label(false)
    ax.view_init(elev=20.0, azim=45)
    plt.savefig("fd2d_surface_3_1.png")
end


function contourplot(ns::Int, nx::Int, ny::Int, ez::Matrix{Float64})::Nothing
    fig, ax = plt.subplots(figsize=(4,4), gridspec_kw=Dict("hspace" => 0.2))
    fig.suptitle(raw"FDTD simulation of a pulse in free space")
    xv, yv = meshgrid(0:nx-1, 0:ny-1)
    ax.contourf(xv, yv, ez', cmap="gray", alpha=0.75)
    ax.contour(xv, yv, ez', colors="k", linewidths=0.25)
    ax.set(xlim=(0, nx-1), ylim=(0, ny-1), aspect="equal")
    ax.set(xlabel=raw"$x\;(cm)$", ylabel=raw"$y\;(cm)$")
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("fd2d_contour_3_1.png")
end


function gaussian(t::Int32, t0::Int, sigma::Float64)::Float64
    return exp(-0.5 * ((t - t0)/sigma)^2)
end


function dfield(t::Int32, nx::Int, ny::Int, dz::Matrix{Float64}, hx::Matrix{Float64}, hy::Matrix{Float64})
    """ calculate the electric flux density Dz """
    @views dz[2:ny,2:nx] .+= 0.5 .* (hy[2:ny,2:nx] .- hy[1:ny-1,2:nx] .- hx[2:ny,2:nx] .+ hx[2:ny,1:nx-1])
    # put a Gaussian pulse in the middle
    dz[div(ny,2)+1,div(nx,2)+1] = gaussian(t, 20, 6.0)
end


function efield(nx::Int, ny::Int, naz::Matrix{Float64}, dz::Matrix{Float64}, ez::Matrix{Float64})
    """ calculate the Ez field from Dz """
    @views ez[2:ny,2:nx] .= naz[2:ny,2:nx] .* dz[2:ny,2:nx]
end


function hfield(nx::Int, ny::Int, ez::Matrix{Float64}, hx::Matrix{Float64}, hy::Matrix{Float64})
    """ calculate the Hx and Hy field """
    @views hx[1:ny-1,1:nx-1] .+= 0.5 .* (ez[1:ny-1,1:nx-1] .- ez[1:ny-1,2:nx])
    @views hy[1:ny-1,1:nx-1] .+= 0.5 .* (ez[2:ny,1:nx-1] .- ez[1:ny-1,1:nx-1])
end


function main()

    nx::Int = 60  # number of grid points
    ny::Int = 60  # number of grid points

    ns::Int = 70  # number of time steps

    dz = zeros(Float64, (ny, nx))
    ez = zeros(Float64, (ny, nx))
    hx = zeros(Float64, (ny, nx))
    hy = zeros(Float64, (ny, nx))

    naz = ones(Float64, (ny, nx))

    for t in Int32.(1:ns)
        dfield(t, nx, ny, dz, hx, hy)
        efield(nx, ny, naz, dz, ez)
        hfield(nx, ny, ez, hx, hy)
    end

    surfaceplot(ns, nx, ny, ez)
    contourplot(ns, nx, ny, ez)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
