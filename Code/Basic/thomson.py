"""
Solve Thomson problem using various optimization methods.
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib as mpl
import cmasher as cmr
import gurobipy as gp
from gurobipy import GRB
from src import *

# --- Initialize ---
# Minimize the function for a given number of additional points m
# By default one charge is always placed at the north pole
m = 9
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

ax.scatter(0, 0, 1, marker="o", s=30, color="#28fc8f", label="North pole charge")
ax.scatter(x, y, z, marker="o", s=30, color="#f20a72", label="Free charges")

x = np.append(x, 0)
y = np.append(y, 0)
z = np.append(z, 1)

# Connect points with dashed lines
for i in range(m + 1):
    for j in range(m + 1):
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 
                linestyle="--", color="#696969", linewidth=1)

ax.set_title(f"Thomson Problem for m = {m} + 1")
plt.legend()
plt.subplots_adjust(left=0, bottom=0.03, right=1, top=0.93)

# DEBUG
# artists = ax.get_children()
# axes = {a.axes for a in artists}

def init_anim():
    ax.view_init(20, 0)
    return fig,

def animate(i):
    ax.view_init(30, i)
    return fig,

anim = FuncAnimation(fig, animate, frames=360, interval=20, blit=False)
writer= FFMpegWriter(fps=30)
anim.save("./Videos/charges_10.mp4", writer=writer, dpi=400)
plt.show()
