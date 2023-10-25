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

to_minimize = lambda v: stoplight(v, v_0, t_step)

# Add constraints
constraint = lambda v: 1 + np.exp(KAPPA * (np.sum([0.5 * v[i] if i == 0 or i == len(v) - 1 else v[i] for i in range(len(v))]) - l/t_step))

# positivity = 

func = lambda v: to_minimize(v) + constraint(v)

# Define as n dimensional problem. One for each time step or rather v_i
t = np.linspace(0, 1, n)
res = minimize(
    fun=func,
    x0=np.ones(n),
    method="Powell",
    tol=1e-6,
)


# --- Plot ---
fig, ax = plt.subplots()

# Plot the function
plt.plot(res.x)
plt.show()

