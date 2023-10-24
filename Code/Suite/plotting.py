"""
A file to house all plotting functions.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr

def plot_2D(init_params, converged_params, func, extent=5):
    # Unpack the parameters
    try:
        n_vars = func.__code__.co_argcount - len(func.__defaults__)
    
    except TypeError:
        # If the function has no default arguments
        n_vars = func.__code__.co_argcount
    unpacked_data = np.split(converged_params, n_vars)

    # Create a grid of points
    x_range = np.linspace(-extent, extent, 1000)
    y_range = np.linspace(-extent, extent, 1000)

    X, Y = np.meshgrid(x_range, y_range)

    # Evaluate the function at each point
    Z = func(X, Y)

    # Plot the function and the optimum
    fig, ax = plt.subplots()

    norm  = mpl.colors.Normalize(vmin=Z.min(), vmax=Z.max())
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmr.cosmic)


    ax.contourf(X, Y, Z, cmap=cmr.cosmic, levels=100, norm=norm)
    ax.contour(X, Y, Z, colors='black', levels=20, linewidths=0.5)
    ax.scatter(init_params[0], init_params[1], marker="D", color="#28fc8f", label="Starting points")
    ax.scatter(*unpacked_data, marker="D", color="#f20a72", label="Converged points")

    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label(r"$f(x, y)$")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect("equal")
    ax.legend()

    return fig, ax

