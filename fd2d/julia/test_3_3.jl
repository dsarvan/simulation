#!/usr/bin/env julia
# File: test_3_3.jl
# Name: D.Saravanan
# Date: 05/03/2025

""" Simulation of a plane wave pulse propagating in free space in the transverse
magnetic (TM) mode with the two-dimensional perfectly matched layer (PML) """

import PyPlot as plt
using LoopVectorization
using Printf

plt.matplotlib.style.use("classic")
plt.matplotlib.style.use("../pyplot.mplstyle")
meshgrid(xs, ys) = xs'.+ 0 .*ys, ys.+ 0 .*xs'


function surfaceplot(ns::Int, nx::Int, ny::Int, ez::Array{Float32})::Nothing
    fig, ax = plt.subplots(subplot_kw=Dict("projection"=>"3d"))
    fig.suptitle(raw"FDTD simulation of a plane wave in free space with PML")
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
    fig.suptitle(raw"FDTD simulation of a plane wave in free space with PML")
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


function gaussian(t::Int32, t0::Int, sigma::Float32)::Float32
    return exp(-0.5*((t - t0)/sigma)^2)
end


function ezinct(ny::Int, ezi::Array{Float32}, hxi::Array{Float32}, bc::Array{Float32})
    """ calculate the incident Ez """
    @tturbo for j in 2:ny
        ezi[j] += 0.5 * (hxi[j-1] - hxi[j])
    end
    # absorbing boundary conditions
    ezi[1], bc[1], bc[2] = bc[1], bc[2], ezi[2]
    ezi[ny], bc[4], bc[3] = bc[4], bc[3], ezi[ny-1]
end


function dfield(t::Int32, nx::Int, ny::Int, pml::pmlayer, ezi::Array{Float32}, dz::Array{Float32}, hx::Array{Float32}, hy::Array{Float32})
    """ calculate the electric flux density Dz """
    @tturbo for j in 2:ny
        for i in 2:nx
            dz[i,j] = pml.gx3[i] * pml.gy3[j] * dz[i,j] + pml.gx2[i] * pml.gy2[j] * 0.5 * (hy[i,j] - hy[i-1,j] - hx[i,j] + hx[i,j-1])
        end
    end
    # put a Gaussian pulse at the low end
    ezi[4] = gaussian(t, 20, 8.0f0)
end


function inctdz(nx::Int, ny::Int, npml::Int, hxi::Array{Float32}, dz::Array{Float32})
    """ incident Dz values """
    @tturbo for i in npml:nx-npml+1
        dz[i,npml] += 0.5 * hxi[npml-1]
        dz[i,ny-npml+1] -= 0.5 * hxi[ny-npml+1]
    end
end


function efield(nx::Int, ny::Int, naz::Array{Float32}, dz::Array{Float32}, ez::Array{Float32})
    """ calculate the Ez field from Dz """
    @tturbo for j in 1:ny
        for i in 1:nx
            ez[i,j] = naz[i,j] * dz[i,j]
        end
    end
end


function hxinct(ny::Int, ezi::Array{Float32}, hxi::Array{Float32})
    """ calculate the incident Hx """
    @tturbo for j in 1:ny-1
        hxi[j] += 0.5 * (ezi[j] - ezi[j+1])
    end
end


function hfield(nx::Int, ny::Int, pml::pmlayer, ez::Array{Float32}, ihx::Array{Float32}, ihy::Array{Float32}, hx::Array{Float32}, hy::Array{Float32})
    """ calculate the Hx and Hy field """
    @tturbo for j in 1:ny-1
        for i in 1:nx-1
            ihx[i,j] += ez[i,j] - ez[i,j+1]
            ihy[i,j] += ez[i,j] - ez[i+1,j]
            hx[i,j] = pml.fy3[j] * hx[i,j] + pml.fy2[j] * (0.5 * ez[i,j] - 0.5 * ez[i,j+1] + pml.fx1[i] * ihx[i,j])
            hy[i,j] = pml.fx3[i] * hy[i,j] - pml.fx2[i] * (0.5 * ez[i,j] - 0.5 * ez[i+1,j] + pml.fy1[j] * ihy[i,j])
        end
    end
end


function incthx(nx::Int, ny::Int, npml::Int, ezi::Array{Float32}, hx::Array{Float32})
    """ incident Hx values """
    @tturbo for i in npml:nx-npml+1
        hx[i,npml-1] += 0.5 * ezi[npml]
        hx[i,ny-npml+1] -= 0.5 * ezi[ny-npml+1]
    end
end


function incthy(nx::Int, ny::Int, npml::Int, ezi::Array{Float32}, hy::Array{Float32})
    """ incident Hy values """
    @tturbo for j in npml:ny-npml+1
        hy[npml-1,j] -= 0.5 * ezi[j]
        hy[nx-npml+1,j] += 0.5 * ezi[j]
    end
end


function pmlparam(nx::Int, ny::Int, npml::Int, pml::pmlayer)::Nothing
    """ calculate the two-dimensional perfectly matched layer (PML) parameters """
    for n in 1:npml
        xm = 0.33*((npml-n+1)/npml)^3
        xn = 0.33*((npml-n+0.5)/npml)^3
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

    ezi = zeros(Float32, ny)
    hxi = zeros(Float32, ny)

    dz = zeros(Float32, (nx, ny))
    ez = zeros(Float32, (nx, ny))
    hx = zeros(Float32, (nx, ny))
    hy = zeros(Float32, (nx, ny))

    ihx = zeros(Float32, (nx, ny))
    ihy = zeros(Float32, (nx, ny))

    naz = ones(Float32, (nx, ny))

    bc = zeros(Float32, 4)

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
        ezinct(ny, ezi, hxi, bc)
        dfield(t, nx, ny, pml, ezi, dz, hx, hy)
        inctdz(nx, ny, npml, hxi, dz)
        efield(nx, ny, naz, dz, ez)
        hxinct(ny, ezi, hxi)
        hfield(nx, ny, pml, ez, ihx, ihy, hx, hy)
        incthx(nx, ny, npml, ezi, hx)
        incthy(nx, ny, npml, ezi, hy)
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
