"""
We will use this file to test the performance of various optimization methods.
These have been mentioned in README.md. The benchmark to use are the 
pathological functions defined in functions.py.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
from functions import *
from plotting import *
from perftest import *


# --- Initialize ---

to_benchmark = [
    nonlin_rosenbrock2,
    nonlin_ackley2,
    nonlin_bukinN62,
    nonlin_cross_in_tray2,
    nonlin_eggholder2,
    nonlin_himmelblau2,
    nonlin_holder_table2,
    nonlin_leviN132,
    nonlin_rastrigin2,
    nonlin_schafferN22,
    nonlin_schafferN42,
    nonlin_sphere2,
]

start_points = {
    "rosenbrock": np.array([1, 1]),
    "ackley": np.array([1, 1]),
    "bukin": np.array([1, 1]),
    "cross_in_tray": np.array([1, 1]),
    "eggholder": np.array([1, 1]),
    "himmelblau": np.array([1, 1]),
    "holder_table": np.array([1, 1]),
    "levi": np.array([1, 1]),
    "rastrigin": np.array([1, 1]),
    "schaffer2": np.array([1, 1]),
    "schaffer4": np.array([1, 1]),
    "sphere": np.array([1, 1]),
}

data = pd.DataFrame(columns=["Function", "Method", "Converged point", "Absolute error"])

# --- Benchmark ---

for func in to_benchmark:
    points = benchmark_func(func, np.array([1, 1]))

points = benchmark_func(nonlin_rastrigin2, np.array([1, 1]))

correct = np.array([0, 0])

abs_error = np.abs(points - correct)
print(f"Converged points {points}")
print(f"Absolute error {abs_error}")



