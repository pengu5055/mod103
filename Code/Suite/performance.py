"""
We will use this file to test the performance of various optimization methods.
These have been mentioned in README.md. The benchmark to use are the 
pathological functions defined in functions.py.
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
from functions import *
from plotting import *
from perftest import *


# --- Initialize ---
points = benchmark_func(nonlin_rastrigin2, np.array([1, 1]))

correct = np.array([0, 0])

abs_error = np.abs(points - correct)
print(f"Converged points {points}")
print(f"Absolute error {abs_error}")

# fig, ax = plot_2D(np.array([1, 1]), points[0], nonlin_rastrigin2)
# plt.show()

