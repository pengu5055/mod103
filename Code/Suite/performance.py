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
from typing import Iterable, Callable


# --- Define functions ---
def wrap_func(data, func:Callable=None):
    # Could possibly be done with a decorator, but more complicated
    try:
        n_vars = func.__code__.co_argcount - len(func.__defaults__)
    
    except TypeError:
        # If the function has no default arguments
        n_vars = func.__code__.co_argcount
    unpacked_data = np.split(data, n_vars)

    return func(*unpacked_data)
        
    
        


def benchmark_func(
        functions: Iterable[Callable],
        start: Iterable[float],
):
    """

    """
    # Check that functions is an iterable
    if not isinstance(functions, Iterable):
        raise TypeError("functions must be an iterable.")
    
    # Check that start is an iterable
    if not isinstance(start, Iterable):
        raise TypeError("start must be an iterable.")


    methods = [
        "Nelder-Mead",
        "Powell",
        "CG",
        "BFGS",
        # "Newton-CG",    
    ]
    # + Gurobi (not a scipy method)


    results = []

    for i, func in enumerate(functions):
        to_minimize = lambda x: func(x)
        column = []

        for method in methods:
            res = minimize(
                fun=wrap_func,
                args=(to_minimize),
                x0=start,
                method=method,
                tol=1e-6,
            )
            column.append(res.x)

        results.append(column)

    return np.array(results)

# --- Initialize ---
points = benchmark_func([nonlin_rastrigin2], np.array([1, 1]))

correct = np.array([0, 0])

abs_error = np.abs(points - correct)
print(f"Converged points {points}")
print(f"Absolute error {abs_error}")



