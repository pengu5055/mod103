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

    # Check if the function is multi-modal
    if name == "nonlin_himmelblau2":
        # Take the global minimum that yields the lowest absolute error
        possibilities = np.array([
            (3, 2),
            (-2.805118, 3.131312),
            (-3.779310, -3.283186),
            (3.584428, -1.848126),
        ])

        # Calculate the relative error for each point
        smallest = np.inf
        for point in possibilities:
            himmelblau_abs_error = np.abs(points - point)
            
            if np.max(himmelblau_abs_error) < smallest:
                smallest = np.max(himmelblau_abs_error)
                optimal_points[name] = point
        
    elif name == "nonlin_crossintray2":
        possibilities = np.array([
            (1.34941, -1.34941),
            (1.34941, 1.34941),
            (-1.34941, 1.34941),
            (-1.34941, -1.34941),
        ])

        smallest = np.inf
        for point in possibilities:
            crossintray_abs_error = np.abs(points - point)

            if np.max(crossintray_abs_error) < smallest:
                smallest = np.max(crossintray_abs_error)
                optimal_points[name] = point
    
    elif name == "nonlin_schafferN42":
        possibilities = np.array([
            (0, 1.253131828792882),
            (0, -1.253131828792882),
            (1.253131828792882, 0),
            (-1.253131828792882, 0),
        ])

        smallest = np.inf
        for point in possibilities:
            schaffer_abs_error = np.abs(points - point)

            if np.max(schaffer_abs_error) < smallest:
                smallest = np.max(schaffer_abs_error)
                optimal_points[name] = point
    
    elif name == "nonlin_holder_table2":
        possibilities = np.array([(8.05502, 9.66459), (-8.05502, 9.66459), (8.05502, -9.66459), (-8.05502, -9.66459)])

        smallest = np.inf
        for point in possibilities:
            holder_abs_error = np.abs(points - point)

            if np.max(holder_abs_error) < smallest:
                smallest = np.max(holder_abs_error)
                optimal_points[name] = point
    
    
    # --- Calculate the absolute error ---

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

data.to_hdf("Results/rel-err_benchmark_v2.h5", key="Benchmarks", mode="w", complevel=9)

print(data)



