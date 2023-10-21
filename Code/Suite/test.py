"""
Test the gradient descent algorithm.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
from graddesc import gradient_descent
from functions import rastrigin2, nonlin_rastrigin2

# Define the function to be optimized
def gradient(x):
    return np.gradient(nonlin_rastrigin2(x))

# Define a random starting point
scaling = 5.12
sign = np.random.choice([-1, 1], size=2)

start = np.array([sign[0] * np.random.rand() * scaling, sign[1] * np.random.rand() * scaling])

# Optimize the function
x_opt = gradient_descent(
    gradient=gradient,
    start=start,
    learn_rate=0.2,
    n_iter=1000,
    tolerance=1e-06,
)

# Plot the function and the optimum
x_range = np.linspace(-5.12, 5.12, 1000)
y_range = np.linspace(-5.12, 5.12, 1000)

X, Y = np.meshgrid(x_range, y_range)
Z = nonlin_rastrigin2(np.array([X, Y]))

fig, ax = plt.subplots(figsize=(6, 5))

norm = mpl.colors.Normalize(vmin=Z.min(), vmax=Z.max())
mappable = mpl.cm.ScalarMappable(norm=norm, cmap='cmr.cosmic')

ax.contourf(X, Y, Z, levels=100, cmap='cmr.cosmic')
ax.contour(X, Y, Z, levels=100, cmap='cmr.cosmic')
ax.plot(x_opt[0], x_opt[1], marker="X", color="#eb3464", label="Converged point")
ax.plot(start[0], start[1], marker="D", color="#5df09f", label="Starting point")

fig.colorbar(mappable, ax=ax)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("DIY Gradient Descent")
ax.set_aspect("equal")
plt.legend(loc="upper left")
plt.show()


