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
data = pd.read_hdf("Results/global_benchmarks.h5", key="Benchmarks")
print(data.columns)

# The aim is to plot a heatmap of the absolute error for each function
# and each method.

error_matrix = np.zeros((len(to_benchmark), len(methods)))

for i, func in enumerate(to_benchmark):
    name = func.__name__
    print(f"Function: {name}")
    for j, method in enumerate(methods):
        print(f"\t\tMethod: {method}")
        cond = data.index.isin([name]) & (data["Method"] == method)
        err = data.loc[cond, "Absolute error"].values[0]
        print(err)
        error_matrix[i, j] = np.mean(err, axis=0)

# Plot the heatmap
fig, ax = plt.subplots()

norm = mpl.colors.Normalize(vmin=error_matrix.min(), vmax=error_matrix.max())
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmr.ember)

print(len(methods)/len(to_benchmark))

im = ax.imshow(error_matrix, extent=[0, 3*len(methods), 0, len(to_benchmark)],
               cmap=cmr.ember, norm=norm)

# Add the colorbar
cbar = fig.colorbar(mappable, ax=ax)
cbar.set_label(r"Abs. error")

# Add the function names

names = [str(func.__name__).removeprefix("nonlin_") for func in to_benchmark]

ax.set_xticks(np.arange(3*len(methods), step=3) + 1.5) 
ax.set_yticks(np.arange(len(to_benchmark)))

ax.set_xticklabels(methods)
ax.set_yticklabels(names)

ax.set_xlabel("Method")
ax.set_ylabel("Function")

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(to_benchmark)):
    for j in range(len(methods)):
        text = ax.text(
            3*j + 1.5,
            i + 0.5,
            f"{error_matrix[i, j]:.2e}",
            ha="center",
            va="center",
            color="w",
        )

ax.set_title("Absolute error of the converged points")
ax.set_aspect("equal")
plt.axis("square")
plt.show()


