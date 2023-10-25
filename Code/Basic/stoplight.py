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
KAPPA = np.linspace(0, 1, 10)

output = []
for i, k in enumerate(KAPPA):
    print(f"Solving for parameter {i + 1} of {len(KAPPA)}")
    solution = stoplight_solver(n, v_0, l, t_step, k)
    output.append(solution)
# --- Plot ---
cm = pl.cartocolors.sequential.PurpOr_4.mpl_colormap
colors = cmr.take_cmap_colors(cm, len(KAPPA), return_fmt="hex", cmap_range=(0.2, 0.8))

fig, ax = plt.subplots()

# Plot the function
for i in range(len(output)):
    solution = output[i]
    ax.plot(solution, label=f"\kappa={KAPPA[i]}", color=colors[i])

ax.set_xlabel("Time step")
ax.set_ylabel("Velocity")
ax.set_title("Stoplight problem with different penalty parameters")
plt.legend()
plt.show()

