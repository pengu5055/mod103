"""
The aim is to plot a contour plot of each benchmark function we used in the test,
and to also plot their global minima.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib as mpl
import palettable as pl
import cmasher as cmr
from constants import *

functions = to_benchmark
bounds = boundaries
minima = optimal_points

# --- Plot 1 ---
if False:
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    
    ax[0, 0].set_title("Rosenbrock")
    X, Y = np.meshgrid(np.linspace(*bounds["nonlin_rosenbrock2"][0], 1000), np.linspace(*bounds["nonlin_rosenbrock2"][1], 1000))
    Z = functions[0](X, Y)
    cm = pl.cartocolors.sequential.DarkMint_7.mpl_colormap
    norm = mpl.colors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    ax[0, 0].contourf(X, Y, Z, cmap=cm, levels=100, norm=norm)
    ax[0, 0].contour(X, Y, Z, colors='black', levels=50, linewidths=0.5)
    ax[0, 0].scatter(*minima["nonlin_rosenbrock2"], marker="D", color="#f20a72", label="Global minimum")
    cbar = fig.colorbar(mappable, ax=ax[0, 0])
    
    
    ax[0, 1].set_title("Ackley")
    X, Y = np.meshgrid(np.linspace(*bounds["nonlin_ackley2"][0], 1000), np.linspace(*bounds["nonlin_ackley2"][1], 1000))
    Z = functions[1](X, Y)
    cm = pl.cartocolors.sequential.Emrld_7.mpl_colormap
    norm = mpl.colors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    ax[0, 1].contourf(X, Y, Z, cmap=cm, levels=100, norm=norm)
    ax[0, 1].contour(X, Y, Z, colors='black', levels=35, linewidths=0.5)
    ax[0, 1].scatter(*minima["nonlin_ackley2"], marker="D", color="#f20a72", label="Global minimum")
    cbar = fig.colorbar(mappable, ax=ax[0, 1])
    
    
    ax[1, 0].set_title("Bukin N6")
    X, Y = np.meshgrid(np.linspace(*bounds["nonlin_bukinN62"][0], 1000), np.linspace(*bounds["nonlin_bukinN62"][1], 1000))
    Z = functions[2](X, Y)
    cm = pl.cartocolors.sequential.PinkYl_7.mpl_colormap
    norm = mpl.colors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    ax[1, 0].contourf(X, Y, Z, cmap=cm, levels=100, norm=norm)
    ax[1, 0].contour(X, Y, Z, colors='black', levels=35, linewidths=0.5)
    ax[1, 0].scatter(*minima["nonlin_bukinN62"], marker="D", color="#f20a72", label="Global minimum")
    cbar = fig.colorbar(mappable, ax=ax[1, 0])
    
    
    ax[1, 1].set_title("Cross-in-tray")
    X, Y = np.meshgrid(np.linspace(*bounds["nonlin_cross_in_tray2"][0], 1000), np.linspace(*bounds["nonlin_cross_in_tray2"][1], 1000))
    Z = functions[3](X, Y)
    cm = pl.cartocolors.sequential.BurgYl_7.mpl_colormap
    norm = mpl.colors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    ax[1, 1].contourf(X, Y, Z, cmap=cm, levels=100, norm=norm)
    ax[1, 1].contour(X, Y, Z, colors='black', levels=35, linewidths=0.5)
    mm = [(1.34941, -1.34941), (1.34941, 1.34941), (-1.34941, 1.34941), (-1.34941, -1.34941)]
    for i in mm:
        ax[1, 1].scatter(*i, marker="D", color="#f20a72", label="Global minimum")
    cbar = fig.colorbar(mappable, ax=ax[1, 1])
    
    for i in range(2):
        for j in range(2):
            ax[i, j].set_xlabel(r"$x$")
            ax[i, j].set_ylabel(r"$y$")
            # ax[i, j].set_aspect("equal")
            # ax[i, j].legend()
    plt.subplots_adjust(wspace=0.17, left=0.07, right=0.98, 
                        top=0.95, bottom=0.15)
    plt.show()

# --- Plot 2 ---
if False:
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    
    ax[0, 0].set_title("Eggholder")
    X, Y = np.meshgrid(np.linspace(*bounds["nonlin_eggholder2"][0], 1000), np.linspace(*bounds["nonlin_eggholder2"][1], 1000))
    Z = functions[4](X, Y)
    cm = pl.cartocolors.sequential.Purp_7.mpl_colormap
    norm = mpl.colors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    ax[0, 0].contourf(X, Y, Z, cmap=cm, levels=100, norm=norm)
    ax[0, 0].contour(X, Y, Z, colors='black', levels=20, linewidths=0.5)
    ax[0, 0].scatter(*minima["nonlin_eggholder2"], marker="D", color="#f20a72", label="Global minimum")
    cbar = fig.colorbar(mappable, ax=ax[0, 0])
    
    ax[0, 1].set_title("Himmelblau")
    X, Y = np.meshgrid(np.linspace(*bounds["nonlin_himmelblau2"][0], 1000), np.linspace(*bounds["nonlin_himmelblau2"][1], 1000))
    Z = functions[5](X, Y)
    cm = pl.cartocolors.sequential.Emrld_7.mpl_colormap
    norm = mpl.colors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    ax[0, 1].contourf(X, Y, Z, cmap=cm, levels=100, norm=norm)
    ax[0, 1].contour(X, Y, Z, colors='black', levels=35, linewidths=0.5)
    mm = [(3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)]
    for i in mm:
        ax[0, 1].scatter(*i, marker="D", color="#f20a72", label="Global minimum")
    cbar = fig.colorbar(mappable, ax=ax[0, 1])
    
    ax[1, 0].set_title("Holder table")
    X, Y = np.meshgrid(np.linspace(*bounds["nonlin_holder_table2"][0], 1000), np.linspace(*bounds["nonlin_holder_table2"][1], 1000))
    Z = functions[6](X, Y)
    cm = pl.cartocolors.sequential.Burg_7.mpl_colormap
    norm = mpl.colors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    ax[1, 0].contourf(X, Y, Z, cmap=cm, levels=100, norm=norm)
    ax[1, 0].contour(X, Y, Z, colors='black', levels=35, linewidths=0.5)
    mm = [(8.05502, 9.66459), (-8.05502, 9.66459), (8.05502, -9.66459), (-8.05502, -9.66459)]
    for i in mm:
        ax[1, 0].scatter(*i, marker="D", color="#f20a72", label="Global minimum")
    cbar = fig.colorbar(mappable, ax=ax[1, 0])
    
    ax[1, 1].set_title("Levi N13")
    X, Y = np.meshgrid(np.linspace(*bounds["nonlin_leviN132"][0], 1000), np.linspace(*bounds["nonlin_leviN132"][1], 1000))
    Z = functions[7](X, Y)
    cm = pl.cartocolors.sequential.Mint_7.mpl_colormap
    norm = mpl.colors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    ax[1, 1].contourf(X, Y, Z, cmap=cm, levels=100, norm=norm)
    ax[1, 1].contour(X, Y, Z, colors='black', levels=20, linewidths=0.5)
    ax[1, 1].scatter(*minima["nonlin_leviN132"], marker="D", color="#f20a72", label="Global minimum")
    cbar = fig.colorbar(mappable, ax=ax[1, 1])
    
    for i in range(2):
        for j in range(2):
            ax[i, j].set_xlabel(r"$x$")
            ax[i, j].set_ylabel(r"$y$")
            # ax[i, j].set_aspect("equal")
            # ax[i, j].legend()
    
    plt.subplots_adjust(wspace=0.17, left=0.07, right=0.98,
                        top=0.95, bottom=0.15)
    plt.show()

# --- Plot 3 ---
if True:
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    ax[0, 0].set_title("Rastrigin")
    X, Y = np.meshgrid(np.linspace(*bounds["nonlin_rastrigin2"][0], 1000), np.linspace(*bounds["nonlin_rastrigin2"][1], 1000))
    Z = functions[8](X, Y)
    cm = pl.colorbrewer.sequential.BuPu_7.mpl_colormap
    norm = mpl.colors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    ax[0, 0].contourf(X, Y, Z, cmap=cm, levels=100, norm=norm)
    ax[0, 0].contour(X, Y, Z, colors='black', levels=20, linewidths=0.5)
    ax[0, 0].scatter(*minima["nonlin_rastrigin2"], marker="D", color="#f20a72", label="Global minimum")
    cbar = fig.colorbar(mappable, ax=ax[0, 0])

    ax[0, 1].set_title("Schaffer N2")
    X, Y = np.meshgrid(np.linspace(*bounds["nonlin_schafferN22"][0], 1000), np.linspace(*bounds["nonlin_schafferN22"][1], 1000))
    Z = functions[9](X, Y)
    cm = pl.colorbrewer.sequential.Purples_7.mpl_colormap
    norm = mpl.colors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    ax[0, 1].contourf(X, Y, Z, cmap=cm, levels=100, norm=norm)
    ax[0, 1].contour(X, Y, Z, colors='black', levels=1, linewidths=0.5)
    ax[0, 1].scatter(*minima["nonlin_schafferN22"], marker="D", color="#f20a72", label="Global minimum")
    cbar = fig.colorbar(mappable, ax=ax[0, 1])
    ax[0, 1].set_xlim(-10, 10)
    ax[0, 1].set_ylim(-10, 10)

    ax[1, 0].set_title("Schaffer N4")
    X, Y = np.meshgrid(np.linspace(*bounds["nonlin_schafferN42"][0], 1000), np.linspace(*bounds["nonlin_schafferN42"][1], 1000))
    Z = functions[10](X, Y)
    cm = pl.colorbrewer.sequential.YlGn_7.mpl_colormap
    norm = mpl.colors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    ax[1, 0].contourf(X, Y, Z, cmap=cm, levels=100, norm=norm)
    ax[1, 0].contour(X, Y, Z, colors='black', levels=1, linewidths=0.5)
    mm = [
            (0, 1.253131828792882),
            (0, -1.253131828792882),
            (1.253131828792882, 0),
            (-1.253131828792882, 0),
        ]
    for i in mm:
        ax[1, 0].scatter(*i, marker="D", color="#f20a72", label="Global minimum")
    cbar = fig.colorbar(mappable, ax=ax[1, 0])
    ax[1, 0].set_xlim(-10, 10)
    ax[1, 0].set_ylim(-10, 10)

    ax[1, 1].set_title("Sphere")
    X, Y = np.meshgrid(np.linspace(*bounds["nonlin_sphere2"][0], 1000), np.linspace(*bounds["nonlin_sphere2"][1], 1000))
    Z = functions[11](X, Y)
    cm = pl.colorbrewer.sequential.YlOrRd_7.mpl_colormap
    norm = mpl.colors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    ax[1, 1].contourf(X, Y, Z, cmap=cm, levels=100, norm=norm)
    ax[1, 1].contour(X, Y, Z, colors='black', levels=20, linewidths=0.5)
    ax[1, 1].scatter(*minima["nonlin_sphere2"], marker="D", color="#f20a72", label="Global minimum")
    cbar = fig.colorbar(mappable, ax=ax[1, 1])

    for i in range(2):
        for j in range(2):
            ax[i, j].set_xlabel(r"$x$")
            ax[i, j].set_ylabel(r"$y$")
            # ax[i, j].set_aspect("equal")
            # ax[i, j].legend()

    plt.subplots_adjust(wspace=0.17, left=0.07, right=0.98,
                        top=0.95, bottom=0.15)
    plt.show()
