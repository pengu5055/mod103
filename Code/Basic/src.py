"""
Source functions go here so that they can be imported into other files
and that code is cleaner.
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
import matplotlib as mpl
import cmasher as cmr

# Define function to minimize
CONST = 1 / (4 * np.pi * 8.85*10**(-12))

def potential(phi_arr, theta_arr):
    """
    Defines the electrostatic potential energy of a system of
    charges on a unit sphere. This is the function that we want
    to minimize.

    Parameters
    ----------
    phi_arr : numpy.ndarray or Iterable
        Array of phi values.
    theta_arr : numpy.ndarray or Iterable
        Array of theta values.

    Returns
    -------
    U : numpy.float64
        Potential energy of the system.
    """
    # Calculate the distance between each pair of points
    coords = np.vstack((phi_arr, theta_arr))
    dist= np.zeros((coords.shape[1], coords.shape[1]))

    # Iterate over columns
    for i in range(coords.shape[1]):
        phi_current = coords[0, i]
        theta_current = coords[1, i]
        # Calculate the distance to all other points
        x_current, y_current, z_current = sphere2cart(phi_current, theta_current)

        for j in range(coords.shape[1]):
            phi_other = coords[0, j]
            theta_other = coords[1, j]

            x_other, y_other, z_other = sphere2cart(phi_other, theta_other)

            # Calculate the distance between the two points
            dist[i, j] = np.sqrt((x_current - x_other)**2 + (y_current - y_other)**2 + (z_current - z_other)**2)
            
    # Calculate the potential energy        
    U = np.sum(1 / dist)  # * CONST

    return U


def wrapped_potential(x):
    """
    Wrap potential to only take one argument.
    Which is then unpacked into phi_arr and theta_arr.
    This is not entirely necessary, but it makes it
    easier to use the minimize function.

    Parameters
    ----------
    x : numpy.ndarray
        2D array of phi and theta values.

    Returns
    -------
    U : numpy.float64
        Potential energy of the system.
    """
    two_m = x.size
    m = int(two_m / 2)
    phi_arr = x[:m]
    theta_arr = x[m:]
    return potential(phi_arr, theta_arr)


def plot_unit_sphere(n):
    """
    Plot a unit sphere. This is used to visualize the
    Thomson problem. This is the skeleton of a plot,
    so you're meant to add to it.
    
    Parameters
    ----------
    n : int
        Number of points to use in each direction.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes._subplots.Axes3DSubplot
        Axes object.
    """
    phi = np.linspace(0, 2*np.pi, n)
    theta = np.linspace(0, np.pi, n)

    # Create a meshgrid
    X = np.outer(np.cos(phi), np.sin(theta))
    Y = np.outer(np.sin(phi), np.sin(theta))
    Z = np.outer(np.ones(n), np.cos(theta))

    # Create color array
    C = np.zeros((n, n)) + 0.5

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, facecolors=cmr.cosmic(C), rstride=1,
                    cstride=1, linewidth=0, antialiased=False, alpha=0.3)

    ax.set_title("Thomson Problem")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_aspect("equal")
    
    return fig, ax


def sphere2cart(phi_arr, theta_arr, r=1):
    """
    Convert spherical coordinates to cartesian coordinates.

    Parameters
    ----------
    phi_arr : numpy.ndarray or Iterable
        Array of phi values.
    theta_arr : numpy.ndarray or Iterable
        Array of theta values.

    Returns
    -------
    x : numpy.ndarray
        Array of x values.
    y : numpy.ndarray
        Array of y values.
    z : numpy.ndarray
        Array of z values.
    """
    x = r * np.cos(phi_arr) * np.sin(theta_arr)
    y = r * np.sin(phi_arr) * np.sin(theta_arr)
    z = r * np.ones(phi_arr.size) * np.cos(theta_arr)
    
    return x, y, z
