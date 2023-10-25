"""
Store constant values for the suite.
"""
import numpy as np
from functions import *

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
    "nonlin_rosenbrock2": [(-2, 2), (-1, 3)],
    "nonlin_ackley2": [(-5, 5), (-5, 5)],
    "nonlin_bukinN62": [(-15, -5), (-3, 3)],
    "nonlin_cross_in_tray2": [(-10, 10), (-10, 10)],
    "nonlin_eggholder2": [(-700, 700), (-700, 700)],
    "nonlin_himmelblau2": [(-5, 5), (-5, 5)],
    "nonlin_holder_table2": [(-10, 10), (-10, 10)],
    "nonlin_leviN132": [(-5, 7), (-5, 7)],
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