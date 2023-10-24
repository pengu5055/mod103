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
    "nonlin_rosenbrock2": np.array([3, 2]),
    "nonlin_ackley2": np.array([1, 1]),
    "nonlin_bukinN62": np.array([1, 1]),
    "nonlin_cross_in_tray2": np.array([5, -4]),
    "nonlin_eggholder2": np.array([0, 0]),
    "nonlin_himmelblau2": np.array([0, 0]),
    "nonlin_holder_table2": np.array([0, 0]),
    "nonlin_leviN132": np.array([5, 5]),
    "nonlin_rastrigin2": np.array([3, 4]),
    "nonlin_schafferN22": np.array([20, 10]),
    "nonlin_schafferN42": np.array([20, 10]),
    "nonlin_sphere2": np.array([50, 75]),
}

optimal_points = {
    "nonlin_rosenbrock2": np.array([1, 1]),
    "nonlin_ackley2": np.array([0, 0]),
    "nonlin_bukinN62": np.array([-10, 1]),
    "nonlin_cross_in_tray2": np.array([1.34941, -1.34941]),
    "nonlin_eggholder2": np.array([512, 404.2319]),
    "nonlin_himmelblau2": np.array([3, 2]),
    "nonlin_holder_table2": np.array([8.05502, 9.66459]),
    "nonlin_leviN132": np.array([1, 1]),
    "nonlin_rastrigin2": np.array([0, 0]),
    "nonlin_schafferN22": np.array([0, 0]),
    "nonlin_schafferN42": np.array([0, 1.25313]),
    "nonlin_sphere2": np.array([0, 0]),
}

boundaries = {
    "nonlin_rosenbrock2": [(-5, 5), (-5, 5)],
    "nonlin_ackley2": [(-5, 5), (-5, 5)],
    "nonlin_bukinN62": [(-15, -5), (-3, 3)],
    "nonlin_cross_in_tray2": [(-10, 10), (-10, 10)],
    "nonlin_eggholder2": [(-700, 700), (-700, 700)],
    "nonlin_himmelblau2": [(-5, 5), (-5, 5)],
    "nonlin_holder_table2": [(-1000, 1000), (-1000, 1000)],
    "nonlin_leviN132": [(-10, 10), (-10, 10)],
    "nonlin_rastrigin2": [(-5.12, 5.12), (-5.12, 5.12)],
    "nonlin_schafferN22": [(-100, 100), (-100, 100)],
    "nonlin_schafferN42": [(-100, 100), (-100, 100)],
    "nonlin_sphere2": [(-100, 100), (-100, 100)],
}

methods = [
    "Basin_hopping",
    "Differential_evolution",
    "Dual_annealing",
    "SHGO",
]

columns = ["Method", "Start point", "Converged point", "Absolute error"]
index = ["Function"]

# Create a dataframe to store the results
data = pd.DataFrame()

# --- Benchmark ---

for func in to_benchmark:
    name = func.__name__
    print(f"Benchmarking function: {name}")
    points = benchmark_func(func, start_points[name], boundaries[name])

    # Calculate the absolute error
    abs_error = np.abs(points - optimal_points[name])

    # Add the results to the dataframe
    for i, point in enumerate(points):
        output = pd.DataFrame(
            [
                [
                    methods[i],
                    start_points[name],
                    point,
                    abs_error[i],
                ]
            ],
            columns=columns,
            index=[name],
        )

        data = pd.concat([data, output])

data.to_hdf("Results/global_benchmarks.h5", key="Benchmarks", mode="w", complevel=9)

print(data)



