#!/usr/bin/env julia
# File: test_3_2.jl
# Name: D.Saravanan
# Date: 04/03/2025

""" Simulation of a propagating sinusoidal in free space in the transverse
magnetic (TM) mode with the two-dimensional perfectly matched layer (PML) """

import PyPlot as plt
using Printf

plt.matplotlib.style.use("classic")
plt.matplotlib.style.use("../pyplot.mplstyle")
meshgrid(xs, ys) = xs'.+ 0 .*ys, ys.+ 0 .*xs'


function surfaceplot(ns::Int, nx::Int, ny::Int, ez::Array{Float32})::Nothing
    fig, ax = plt.subplots(subplot_kw=Dict("projection"=>"3d"))
    fig.suptitle(raw"FDTD simulation of a sinusoidal in free space with PML")
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
    fig.suptitle(raw"FDTD simulation of a sinusoidal in free space with PML")
    yv, xv = meshgrid(0:ny-1, 0:nx-1)
    ax.contourf(xv, yv, ez, cmap="gray", alpha=0.75)
    ax.contour(xv, yv, ez, colors="k", linewidths=0.25)
    ax.set(xlim=(0, nx-1), ylim=(0, ny-1), aspect="equal")
    ax.set(xlabel=raw"$x\;(cm)$", ylabel=raw"$y\;(cm)$")
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.show()
end


struct pmlayer
    fx1::Array{Float32}
    fx2::Array{Float32}
    fx3::Array{Float32}
    fy1::Array{Float32}
    fy2::Array{Float32}
    fy3::Array{Float32}
    gx2::Array{Float32}
    gx3::Array{Float32}
    gy2::Array{Float32}
    gy3::Array{Float32}
end


function sinusoidal(t::Int32, ds::Float32, freq::Float32)::Float32
    dt::Float32 = ds/6e8  # time step (s)
    return sin(2*pi*freq*dt*t)
end


function dfield(t::Int32, nx::Int, ny::Int, pml::pmlayer, dz::Array{Float32}, hx::Array{Float32}, hy::Array{Float32})
    """ calculate the electric flux density Dz """
    @views dz[2:nx,2:ny] .= pml.gx3[2:nx] .* pml.gy3[2:ny]' .* dz[2:nx,2:ny] .+ pml.gx2[2:nx] .* pml.gy2[2:ny]' .* 0.5 .* (hy[2:nx,2:ny] .- hy[1:nx-1,2:ny] .- hx[2:nx,2:ny] .+ hx[2:nx,1:ny-1])
    # put a sinusoidal source at a point that is offset four cells
    # from the center of the problem space in each direction
    dz[div(nx,2)-4,div(ny,2)-4] = sinusoidal(t, 0.01f0, 1500f6)
end


function efield(nx::Int, ny::Int, naz::Array{Float32}, dz::Array{Float32}, ez::Array{Float32})
    """ calculate the Ez field from Dz """
    @views ez[1:nx,1:ny] .= naz[1:nx,1:ny] .* dz[1:nx,1:ny]
end


function hfield(nx::Int, ny::Int, pml::pmlayer, ez::Array{Float32}, ihx::Array{Float32}, ihy::Array{Float32}, hx::Array{Float32}, hy::Array{Float32})
    """ calculate the Hx and Hy field """
    curl_em = ez[1:nx-1,1:ny-1] .- ez[1:nx-1,2:ny]
    curl_en = ez[1:nx-1,1:ny-1] .- ez[2:nx,1:ny-1]
    @views ihx[1:nx-1,1:ny-1] .+= curl_em
    @views ihy[1:nx-1,1:ny-1] .+= curl_en
    @views hx[1:nx-1,1:ny-1] .= pml.fy3[1:ny-1]' .* hx[1:nx-1,1:ny-1] .+ pml.fy2[1:ny-1]' .* (0.5 .* curl_em .+ pml.fx1[1:nx-1] .* ihx[1:nx-1,1:ny-1])
    @views hy[1:nx-1,1:ny-1] .= pml.fx3[1:nx-1] .* hy[1:nx-1,1:ny-1] .- pml.fx2[1:nx-1] .* (0.5 .* curl_en .+ pml.fy1[1:ny-1]' .* ihy[1:nx-1,1:ny-1])
end


function pmlparam(nx::Int, ny::Int, npml::Int, pml::pmlayer)::Nothing
    """ calculate the two-dimensional perfectly matched layer (PML) parameters """
    for n in 1:npml
        xm = 0.33*((npml-n+1)/npml)^3
        xn = 0.33*((npml-n+1-0.5)/npml)^3
        pml.fx1[n] = pml.fx1[nx-n] = pml.fy1[n] = pml.fy1[ny-n] = xn
        pml.fx2[n] = pml.fx2[nx-n] = pml.fy2[n] = pml.fy2[ny-n] = 1/(1+xn)
        pml.gx2[n] = pml.gx2[nx+1-n] = pml.gy2[n] = pml.gy2[ny+1-n] = 1/(1+xm)
        pml.fx3[n] = pml.fx3[nx-n] = pml.fy3[n] = pml.fy3[ny-n] = (1-xn)/(1+xn)
        pml.gx3[n] = pml.gx3[nx+1-n] = pml.gy3[n] = pml.gy3[ny+1-n] = (1-xm)/(1+xm)
    end
end


function main()

    nx::Int = 1024  # number of grid points
    ny::Int = 1024  # number of grid points

    ns::Int = 5000  # number of time steps

    dz = zeros(Float32, (nx, ny))
    ez = zeros(Float32, (nx, ny))
    hx = zeros(Float32, (nx, ny))
    hy = zeros(Float32, (nx, ny))

    ihx = zeros(Float32, (nx, ny))
    ihy = zeros(Float32, (nx, ny))

    naz = ones(Float32, (nx, ny))

    pml = pmlayer(
        fill(0.0f0::Float32, nx),
        fill(1.0f0::Float32, nx),
        fill(1.0f0::Float32, nx),
        fill(0.0f0::Float32, ny),
        fill(1.0f0::Float32, ny),
        fill(1.0f0::Float32, ny),
        fill(1.0f0::Float32, nx),
        fill(1.0f0::Float32, nx),
        fill(1.0f0::Float32, ny),
        fill(1.0f0::Float32, ny),
    )

    npml::Int = 8  # pml thickness
    pmlparam(nx, ny, npml, pml)

    ds::Float32 = 0.01  # spatial step (m)
    dt::Float32 = ds/6e8  # time step (s)

    stime = time_ns()

    for t in Int32.(1:ns)
        dfield(t, nx, ny, pml, dz, hx, hy)
        efield(nx, ny, naz, dz, ez)
        hfield(nx, ny, pml, ez, ihx, ihy, hx, hy)
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
