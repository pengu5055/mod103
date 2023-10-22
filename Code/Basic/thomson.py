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
# Minimize the function for a given number of points m
m = 2
n = 50

# Fix one point at the north pole
phi_arr = np.array([0])
theta_arr = np.array([0])
# Give the rest random starting points
phi_arr = np.append(phi_arr, np.random.rand(m - 1) * 2*np.pi)
theta_arr = np.append(theta_arr, np.random.rand(m - 1) * np.pi)

# Create starting guess vector (concatenate phi and theta)
x0 = np.concatenate((phi_arr, theta_arr))

# --- Optimization ---
res = minimize(
    fun=wrapped_potential,
    x0=x0,
    method="Nelder-Mead",
    tol=1e-6,
)

# Unpack the result
phi_arr = res.x[:m]
theta_arr = res.x[m:]

# --- Plotting ---
fig, ax = plot_unit_sphere(n)

x, y, z = sphere2cart(phi_arr, theta_arr)

ax.scatter(x, y, z, marker="o", color="red", label="Starting points")

# Connect points with dashed lines
# for i in range(m):
#     for j in range(m):
#         ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], linestyle="--", color="#696969")


plt.legend()
plt.show()


    