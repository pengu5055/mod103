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
from constants import *


# --- Initialize ---
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



