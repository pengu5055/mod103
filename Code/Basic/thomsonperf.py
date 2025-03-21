"""
Benchmark the performance of various local minimization algorithms
on the Thomson problem.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from src import *
import cmasher as cmr
import palettable as pl

# --- Initialize ---
methods = [
    "Nelder-Mead",
    "Powell",
    "CG",
    "BFGS",
]

m = 1  # I know the exact solution for m = 1
n = 50

# Spread charges evenly around the equator
phi_arr = np.linspace(0, 2*np.pi, m)
theta_arr = np.ones(m) * np.pi / 2

# Create starting guess vector (concatenate phi and theta)
x0 = np.concatenate((phi_arr, theta_arr))


results = []

for method in methods:
# --- Optimization ---
    res = minimize(
        fun=wrapped_potential,
        x0=x0,
        method=method,
        tol=1e-6,
    )

    results.append(res.x)

results = np.array(results)

# This only applies for m = 1
abs_err_x = np.abs(0 - results[:, 0] / np.pi)
abs_err_y = np.abs(1 - results[:, 1] / np.pi)

max_abs_err = []
for err_x, err_y in zip(abs_err_x, abs_err_y):
    if err_x > err_y:
        max_abs_err.append(err_x)
    else:
        max_abs_err.append(err_y)

fig, ax = plt.subplots()

# Create bar chart
cm = pl.cartocolors.sequential.PurpOr_4.mpl_colormap
colors = cmr.take_cmap_colors(cm, len(methods), return_fmt="hex", cmap_range=(0.2, 0.8))

ax.barh(
    methods,
    max_abs_err,
    color=colors,
)

# Set bar labels
ax.bar_label(ax.containers[0], fmt="%.2e",
             padding=2, color="#4B4453", fontsize=12)


# Set y ticks to method names
ax.set_yticklabels(methods)

ax.set_title("Absolute error of the 2 charge Thomson problem")
ax.set_ylabel("Abs. error")
ax.set_xlabel("Method")
ax.set_xlim(1e-12, 1e-1)
ax.set_xscale("log")
plt.subplots_adjust(left=0.22)
plt.show()





