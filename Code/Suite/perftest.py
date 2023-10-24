"""
A file to house all performance testing functions.
"""
import numpy as np
from scipy.optimize import basinhopping, differential_evolution, dual_annealing, shgo, Bounds
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
from functions import *
from plotting import *
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

def wrap_2D(data, func:Callable=None):
    unpacked_data = np.split(data, 2)
    return func(*unpacked_data)
        
    
        


def benchmark_func(
        functions: Callable,
        start: Iterable[float],
        bounds: Iterable[Iterable[float]]=[(-5, 5), (-5, 5)],
):
    """
    A function to benchmark the performance of various optimization methods.

    Parameters
    ----------
    functions : Callable
        A function to minimize.
    start : Iterable[float]
        An iterable of starting points.
    bounds : Iterable[Iterable[float]], optional
        Bounds for the optimization, by default [(-5, 5), (-5, 5)].

    Returns
    -------
    results : list
        A list of lists containing the results of the optimization
        for each method.
    """
    # Check that start is an iterable
    if not isinstance(start, Iterable):
        raise TypeError("start must be an iterable.")

    results = []

    func = functions
    to_minimize = lambda x: wrap_2D(x, func=func)
    column = []

    print("Using method: Basin hopping")
    res = basinhopping(
        func=to_minimize,
        x0=start,
        minimizer_kwargs={"method": "Powell"},
        niter=100,
        disp=False,
    )
    column.append(res.x)

    print("Using method: Differential evolution")

    # Create bounds

    res = differential_evolution(
        func=to_minimize,
        bounds=bounds,
        disp=False,
    )
    column.append(res.x)

    print("Using method: Dual annealing")
    res = dual_annealing(
        func=to_minimize,
        bounds=bounds,
    )
    column.append(res.x)

    print("Using method: SHGO")
    res = shgo(
        func=to_minimize,
        bounds=bounds,
    )
    column.append(res.x)

    results.append(column)

    return results