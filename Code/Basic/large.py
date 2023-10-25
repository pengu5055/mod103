"""
Try and use Gurobi to solve the problem on a larger scale.
NOTE: Currently broken, as little time left to fix it.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from src import *
import cmasher as cmr
import palettable as pl
import gurobipy as gp
from gurobipy import GRB

# --- Initialize ---
m = 9
n = 50

# Uniformly distribute points around the equator
phi_arr = np.linspace(0, 2*np.pi, m)
theta_arr = np.ones(m) * np.pi / 2

# Create starting guess vector (concatenate phi and theta)
x0 = np.concatenate((phi_arr, theta_arr))

# --- Potential definition ---
def potential(x, y, z):
    U = np.zeros((m, m + 1))
    for i in range(m + 1):  # +1 due to needing previous point
        for j in range(m + 1):  # +1 due to fixed charge at north pole
            # Calculate the distance between the two points
            U[i, j] = 1/(np.sqrt((x[i] - x[i - 1])**2 + \
                                 (y[i] - y[i - 1])**2 + \
                                 (z[i] - z[i - 1])**2))
        
        # Calculate the distance to the fixed charge at the north pole
        U[i, -1] = 1/(np.sqrt((x[i] - 0)**2 + \
                              (y[i] - 0)**2 + \
                              (z[i] - 1)**2))
            
    # Calculate the potential energy        
    U = np.sum(U)  # * CONST

    return U

# --- Optimization ---
# Create model
model = gp.Model("thomson")

# Create variables
phi = model.addVars(m, lb=0, ub=2*np.pi, name="phi")
theta = model.addVars(m, lb=0, ub=np.pi, name="theta")
x = model.addVars(m + 1, lb=-1, ub=1, name="x")
y = model.addVars(m + 1, lb=-1, ub=1, name="y")
z = model.addVars(m + 1, lb=-1, ub=1, name="z")

# Auxiliary variables for non-convex terms
# 1 : cos(phi)
# 2 : sin(theta)
# 3 : sin(phi)
# 4 : cos(theta)

u1 = model.addVars(m, lb=-1, ub=1, name="u1")
v1 = model.addVars(m, lb=-1, ub=1, name="v1")
u2 = model.addVars(m, lb=-1, ub=1, name="u2")
v2 = model.addVars(m, lb=-1, ub=1, name="v2")
u3 = model.addVars(m, lb=-1, ub=1, name="u3")
v3 = model.addVars(m, lb=-1, ub=1, name="v3")
u4 = model.addVars(m, lb=-1, ub=1, name="u4")
v4 = model.addVars(m, lb=-1, ub=1, name="v4")

root = model.addVars(m, lb=0, ub=1, name="root")


# Add constraint
model.addConstr(phi[0] == 0)
model.addConstr(theta[0] == np.pi/2)

# Add constraints that relate the variables
# They need to be linearized!
b = 2 * np.pi
model.addConstr(u1 * v1 == gp.quicksum([u1, v1, -u1, -v1]) / 4)  # Linearize cosine term
model.addConstr(u1 * b >= 1)
model.addConstr(u1*v1 <= b)
model.addConstr(u2 * v2 == u2 * v2 - (u2 * u2 * u2 * v2 * v2 * v2)/6)  # Linearize cosine term
model.addConstr(u2*v2 <= b)
model.addConstr(u3 * v3 == u3 * v3 - (u3 * u3 * u3 * v3 * v3 * v3)/6)  # Linearize cosine term
model.addConstr(u3*v3 <= b)
model.addConstr(u4 * v4 == gp.quicksum([u4, v4, -u4, -v4]) / 4)  # Linearize cosine term
model.addConstr(u4 * b >= 1)
model.addConstr(u4*v4 <= b)

# Add constraints that relate the variables pt.2
model.addConstr(x == gp.quicksum([u1[i]*v1[i] * u2[i]*v2[i] for i in range(m)]))
model.addConstr(y == gp.quicksum([u2[i]*v2[i] * u3[i]*v3[i] for i in range(m)]))
model.addConstr(z == gp.quicksum([u4[i]*v4[i] for i in range(m)]))

# Add constraints that relate the variables pt.3
model.addConstr(root = gp.quicksum([x[i]**2 + y[i]**2 + z[i]**2 for i in range(m)]))


# Set objective
model.setObjective(potential(x, y, z), GRB.MINIMIZE)

# Optimize model
model.optimize()

# Unpack the result
phi_arr = np.array([phi[i].x for i in range(m)])
theta_arr = np.array([theta[i].x for i in range(m)])

# --- Plotting ---
fig, ax = plot_unit_sphere(n)

x, y, z = sphere2cart(phi_arr, theta_arr)

ax.scatter(0, 0, 1, marker="o", s=30, color="#28fc8f", label="North pole charge")
ax.scatter(x, y, z, marker="o", s=30, color="#f20a72", label="Free charges")
plt.show()


