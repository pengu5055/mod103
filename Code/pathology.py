"""
Because I may not like myself very much, I will try and look at very 
pathological functions or rather functions that are difficult to
optimize and are thus used as benchmarks for optimization algorithms.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
import gurobipy as gp
from gurobipy import GRB

# Global parameters
A = 10
n = 2

def rastrigin(x1, x2, u1, v1, u2, v2, A=A):
    return A*n + x1**2 - A*u1*v1+ x2**2 - A*u2*v2

def rastrigin_plotable(x, A=A):
    return A*n + x[0]**2 - A*np.cos(2*np.pi*x[0]) + x[1]**2 - A*np.cos(2*np.pi*x[1])

# Create a new model
m = gp.Model('rastrigin')

# Allow non-convex objectives / quadratic constraints
m.params.NonConvex = 2

# Create variables
x1 = m.addVar(lb=-5.12, ub=5.12, name="x1")
x2 = m.addVar(lb=-5.12, ub=5.12, name="x2")

# Create additional variables for the non-convex terms
u1 = m.addVar(lb=-1, ub=1, name="u1")
v1 = m.addVar(lb=-1, ub=1, name="v1")
u2 = m.addVar(lb=-1, ub=1, name="u2")
v2 = m.addVar(lb=-1, ub=1, name="v2")

# Add constraints to relate the variables
b = 2 * np.pi

m.addConstr(u1 * v1 == gp.quicksum([u1, v1, -u1, -v1]) / 4)  # Linearize cosine term
m.addConstr(u1 * b >= 1)
m.addConstr(u1*v1 <= b)
m.addConstr(u2 * v2 == gp.quicksum([u2, v2, -u2, -v2]) / 4)  # Linearize cosine term
m.addConstr(u2 * b >= 1)
m.addConstr(u2*v2 <= b)

# Set objective
m.setObjective(rastrigin(x1, x2, u1, v1, u2, v2), GRB.MINIMIZE)

m.optimize()

if m.status == GRB.OPTIMAL:
    print('x: %g' % x1.x)
    print('y: %g' % x2.x)
    print('obj: %g' % m.objVal)

else:
    print('No solution!')
    quit()

# Plot the function and the optimum
x_range = np.linspace(-5.12, 5.12, 1000)
y_range = np.linspace(-5.12, 5.12, 1000)

X, Y = np.meshgrid(x_range, y_range)
Z = rastrigin_plotable(np.array([X, Y]))

fig, ax = plt.subplots(figsize=(6, 5))

norm = mpl.colors.Normalize(vmin=Z.min(), vmax=Z.max())
mappable = mpl.cm.ScalarMappable(norm=norm, cmap='cmr.cosmic')



# Calculate isocontours and plot function
ax.contourf(X, Y, Z, cmap=cmr.cosmic, levels=100)
ax.contour(X, Y, Z, levels=10, colors="black", linewidths=0.75)

ax.plot(x1.x, x2.x, marker="X", color="#f20a72", markersize=10, label="Optimum")

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("Rastrigin function")
ax.set_aspect("equal")
plt.colorbar(mappable, ax=ax)
ax.legend()
plt.show()