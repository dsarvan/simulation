#!/usr/bin/env julia
# File: test_3_1.jl
# Name: D.Saravanan
# Date: 03/03/2025

""" Simulation of a pulse in free space in the transverse magnetic (TM) mode """

import PyPlot as plt
using Printf

plt.matplotlib.style.use("classic")
plt.matplotlib.style.use("../pyplot.mplstyle")
meshgrid(xs, ys) = xs'.+ 0 .*ys, ys.+ 0 .*xs'


function surfaceplot(ns::Int, nx::Int, ny::Int, ez::Array{Float32})::Nothing
    fig, ax = plt.subplots(subplot_kw=Dict("projection"=>"3d"))
    fig.suptitle(raw"FDTD simulation of a pulse in free space")
    yv, xv = meshgrid(0:ny-1, 0:nx-1)
    ax.plot_surface(xv, yv, ez, rstride=1, cstride=1, cmap="gray", lw=0.25)
    ax.text2D(0.1, 0.7, raw"$T$ = "*"$ns", transform=ax.transAxes)
    ax.set(xlim=(0, nx), ylim=(0, ny), zlim=(0, 1))
    ax.set(xlabel=raw"$x\;(cm)$", ylabel=raw"$y\;(cm)$", zlabel=raw"$E_z\;(V/m)$")
    ax.zaxis.set_rotate_label(false); ax.view_init(elev=20.0, azim=45)
    plt.show()
end


function contourplot(ns::Int, nx::Int, ny::Int, ez::Array{Float32})::Nothing
    fig, ax = plt.subplots(figsize=(4,4), gridspec_kw=Dict("hspace"=>0.2))
    fig.suptitle(raw"FDTD simulation of a pulse in free space")
    yv, xv = meshgrid(0:ny-1, 0:nx-1)
    ax.contourf(xv, yv, ez, cmap="gray", alpha=0.75)
    ax.contour(xv, yv, ez, colors="k", linewidths=0.25)
    ax.set(xlim=(0, nx-1), ylim=(0, ny-1), aspect="equal")
    ax.set(xlabel=raw"$x\;(cm)$", ylabel=raw"$y\;(cm)$")
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.show()
end


function gaussian(t::Int32, t0::Int, sigma::Float32)::Float32
    return exp(-0.5*((t - t0)/sigma)^2)
end


function dfield(t::Int32, nx::Int, ny::Int, dz::Array{Float32}, hx::Array{Float32}, hy::Array{Float32})
    """ calculate the electric flux density Dz """
    @views dz[2:nx,2:ny] .+= 0.5 .* (hy[2:nx,2:ny] .- hy[1:nx-1,2:ny] .- hx[2:nx,2:ny] .+ hx[2:nx,1:ny-1])
    # put a Gaussian pulse in the middle
    dz[div(nx,2)+1,div(ny,2)+1] = gaussian(t, 20, 6.0f0)
end


function efield(nx::Int, ny::Int, naz::Array{Float32}, dz::Array{Float32}, ez::Array{Float32})
    """ calculate the Ez field from Dz """
    @views ez[1:nx,1:ny] .= naz[1:nx,1:ny] .* dz[1:nx,1:ny]
end


function hfield(nx::Int, ny::Int, ez::Array{Float32}, hx::Array{Float32}, hy::Array{Float32})
    """ calculate the Hx and Hy field """
    @views hx[1:nx-1,1:ny-1] .+= 0.5 .* (ez[1:nx-1,1:ny-1] .- ez[1:nx-1,2:ny])
    @views hy[1:nx-1,1:ny-1] .-= 0.5 .* (ez[1:nx-1,1:ny-1] .- ez[2:nx,1:ny-1])
end


function main()

    nx::Int = 1024  # number of grid points
    ny::Int = 1024  # number of grid points

    ns::Int = 5000  # number of time steps

    dz = zeros(Float32, (nx, ny))
    ez = zeros(Float32, (nx, ny))
    hx = zeros(Float32, (nx, ny))
    hy = zeros(Float32, (nx, ny))

    naz = ones(Float32, (nx, ny))

    stime = time_ns()

    for t in Int32.(1:ns)
        dfield(t, nx, ny, dz, hx, hy)
        efield(nx, ny, naz, dz, ez)
        hfield(nx, ny, ez, hx, hy)
    end

    ntime = time_ns()
    @printf("Total compute time on CPU: %.3f s\n", (ntime - stime)/1e9)

    println(ez[3,:][1:50])
    surfaceplot(ns, nx, ny, ez)
    contourplot(ns, nx, ny, ez)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
