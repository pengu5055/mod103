"""
Solve Thomson problem using various optimization methods.
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
import matplotlib as mpl
import cmasher as cmr
import gurobipy as gp
from gurobipy import GRB
from src import *

# --- Initialize ---
# Minimize the function for a given number of additional points m
# By default one charge is always placed at the north pole
m = 1
n = 50

# Give the rest random starting points
# phi_arr = np.append(phi_arr, np.random.rand(m - 1) * 2*np.pi)
# theta_arr = np.append(theta_arr, np.random.rand(m - 1) * np.pi)

# Uniformly distribute points around the equator
phi_arr = np.linspace(0, 2*np.pi, m)
theta_arr = np.ones(m) * np.pi / 2

# Create starting guess vector (concatenate phi and theta)
x0 = np.concatenate((phi_arr, theta_arr))

# --- Optimization ---
res = minimize(
    fun=wrapped_potential,
    x0=x0,
    method="Powell",
    tol=1e-6,
)

# Unpack the result
phi_arr = res.x[:m]
theta_arr = res.x[m:]

# --- Plot function that was minimized ---
fig, ax = plot_potential2D(x0, res.x)

plt.title("Potential energy as a function of the second charge")
plt.show()


# --- Plotting ---
fig, ax = plot_unit_sphere(n)

x, y, z = sphere2cart(phi_arr, theta_arr)

ax.scatter(0, 0, 1, marker="o", color="green", label="North pole charge")
ax.scatter(x, y, z, marker="o", color="red", label="Free charges")

x = np.append(x, 0)
y = np.append(y, 0)
z = np.append(z, 1)

# Connect points with dashed lines
for i in range(m + 1):
    for j in range(m + 1):
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], linestyle="--", color="#696969")

ax.set_title(f"Thomson Problem for m = {m}")
plt.legend()
plt.show()


    