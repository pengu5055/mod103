"""
The idea here is to plot the stoplight potential in 2D. Since
the first term is well defined we can take 3 terms total as that will
produce 2 dimensions.
"""
import numpy as np
import matplotlib.pyplot as plt
from src import *
import palettable as pl
import cmasher as cmr

# --- Default Parameters ---
n = 100  # Number of time steps
v_0 = 5  # Initial velocity
l = 100 # Length till stoplight
t_step = 1  # Time step
KAPPA = 0.1  # Penalty parameter

func = lambda v: 1/2 * (v_0 - 0)/(t_step)