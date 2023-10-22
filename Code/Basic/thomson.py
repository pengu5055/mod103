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


# Minimize the function for a given number of points m
m = 2
n = 50


# Give random starting points
phi_arr = np.random.rand(m) * 2*np.pi
theta_arr = np.random.rand(m) * np.pi

# 



# Plot the configuration
fig, ax = plot_unit_sphere(n)

x, y, z = sphere2cart(phi_arr, theta_arr)

ax.scatter(x, y, z, marker="o", color="red", label="Starting points")

# Connect points with dashed lines
for i in range(m):
    for j in range(m):
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], linestyle="--", color="#696969")


plt.legend()
plt.show()


    