"""
The aim here is to solve the stoplight variational problem that
was given in mod101 just using nonlinear optimization this time.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from src import *

# --- Initialize ---
n = 100  # Number of time steps
v_0 = 5  # Initial velocity
l = 100 # Length till stoplight
t_step = 1  # Time step
KAPPA = 0.1  # Penalty parameter

solution = stoplight_solver(n, v_0, l, t_step, KAPPA)

# --- Plot ---
fig, ax = plt.subplots()

# Plot the function
plt.plot(solution)
plt.show()

