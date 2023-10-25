"""
The aim here is to solve the stoplight variational problem that
was given in mod101 just using nonlinear optimization this time.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from src import *
import palettable as pl
import cmasher as cmr

# --- Default Parameters ---
n = 100  # Number of time steps
v_0 = 5  # Initial velocity
l = 100 # Length till stoplight
t_step = 1  # Time step
KAPPA = 0.1  # Penalty parameter

# --- Parameter Sweeps ---
t_step = np.linspace(0, 1, 10)

output = []
for i, k in enumerate(t_step):
    print(f"Solving for parameter {i + 1} of {len(t_step)}")
    solution = stoplight_solver(n, v_0, l, k, KAPPA)
    output.append(solution)
# --- Plot ---
cm = pl.cartocolors.qualitative.Prism_10.mpl_colormap
colors = cmr.take_cmap_colors(cm, len(t_step), return_fmt="hex", cmap_range=(0.2, 0.8))

fig, ax = plt.subplots()

# Plot the function
for i in range(len(output)):
    solution = output[i]
    ax.plot(solution, label=f"$t_0={round(t_step[i],2)}$", color=colors[i])

ax.set_xlabel("Time step")
ax.set_ylabel("Velocity")
ax.set_title("Stoplight problem with different time steps")
plt.legend(loc="lower left")
plt.grid(color="#4d4d4d", alpha=0.1)
plt.show()

