"""
Now that we have Results/global_benchmarks.h5, we can visualize the results.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import cmasher as cmr
from constants import *


# Load the results
data = pd.read_hdf("Results/rel-err_benchmark.h5", key="Benchmarks")
# The aim is to plot a heatmap of the absolute error for each function
# and each method.

error_matrix = np.zeros((len(to_benchmark), len(methods)))

for i, func in enumerate(to_benchmark):
    name = func.__name__
    # print(f"Function: {name}")
    for j, method in enumerate(methods):
        # print(f"\t\tMethod: {method}")
        cond = data.index.isin([name]) & (data["Method"] == method)
        err = data.loc[cond, "Absolute error"].values[0]
        error_matrix[i, j] = np.mean(err, axis=0)

# Plot the heatmap
fig, ax = plt.subplots()

cm = cmr.get_sub_cmap("cmr.infinity_s", 0.1, 0.9)

norm = mpl.colors.SymLogNorm(10e-15, vmin=error_matrix.min() + 1e-16, vmax=error_matrix.max()/3)
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cm)

print(len(methods)/len(to_benchmark))

# Extent set so that the heatmap is square and aligned with the plot grid
im = ax.imshow(error_matrix, extent=[1.5, 3*len(methods) + 1.5, 0, len(to_benchmark) + 0],
               interpolation='none', cmap=cm, norm=norm)

# Add the colorbar
cbar = fig.colorbar(im)
cbar.set_label(r"Abs. error")

# Add the function names

# Try and set grid before setting ticks since then grid 
# will be drawn over the matrix elements
plt.grid(True, color="w", linestyle="--", linewidth=1)

names = [str(func.__name__)
         .removeprefix("nonlin_")
         .removesuffix("2") for func in to_benchmark]
names = names[::-1]
names = ["".join([name[i].upper() if i == 0 else name[i] for i in range(len(name))])
           .replace("_", " ")
           .replace("N", " N") for name in names]

methods = [method.replace("_", " ") for method in methods]


ax.set_xticks(np.arange(3*len(methods), step=3) + 1.5) 
ax.set_yticks(np.arange(len(to_benchmark)))

ax.set_xticklabels(methods)
# Reverse the order of the y-ticks so that the functions are in the same order
ax.set_yticklabels(names)

ax.set_xlabel("Method")
ax.set_ylabel("Function")

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(to_benchmark)):
    for j in range(len(methods)):
        text = ax.text(
            # Corrections for grid alignment and for center justification
            3*j + 1.5 + 1.5,
            11 - i + 0.5,
            f"{error_matrix[i, j]:.2e}",
            ha="center",
            va="center",
            color="w",
        )

ax.set_title("Absolute error of the converged points")
ax.set_aspect("equal")
plt.axis("square")
plt.subplots_adjust(left=0.18, bottom=0.2, right=0.98, top=0.93)
plt.show()


