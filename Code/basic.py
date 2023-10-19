"""
This is my entry point into the current project.
The idea is to first familiarize myself with some methods 
of optimization and perhaps visualize them.

I will try and combine Pyomo with Gurobi for this purpose.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
import gurobipy as gp
from gurobipy import GRB

# Global parameters
a = 1
b = 100

# Demo function 1/xy
def f(x, y, a=a, b=b):
    return (a - x)**2 + b*(y - x**2)**2


# Create a new model
m = gp.Model('test')

# Allow non-convex objectives / quadratic constraints
m.params.NonConvex = 2 

# Create variables
x = m.addVar(lb=-3, ub=3, name="x")
y = m.addVar(lb=-3, ub=3, name="y")

# Create additional variables for the non-convex terms
z = m.addVar(name="z") # x**2
w = m.addVar(name="w") # x*y

# Add constraints to relate the variables
m.addConstr(z == x**2)
m.addConstr(w == x*y)

# Set objective
m.setObjective((a - x)**2 + b*(y**2 - 2*w + z**2), GRB.MINIMIZE)

m.optimize()

if m.status == GRB.OPTIMAL:
    print('x: %g' % x.x)
    print('y: %g' % y.x)
    print('z: %g' % z.x)
    print('w: %g' % w.x)
    print('obj: %g' % m.objVal)

else:
    print('No solution!')
    quit()

# Plot the function and the optimum
x_range = np.linspace(-1, 1, 1000)
y_range = np.linspace(-1, 1, 1000)

X, Y = np.meshgrid(x_range, y_range)

Z = f(X, Y)

fig, ax = plt.subplots()
norm = mpl.colors.LogNorm(vmin=Z.min()+10e-6, vmax=Z.max()+10e-6)
mappable = mpl.cm.ScalarMappable(norm=norm, cmap='cmr.sapphire')

ax.contourf(X, Y, Z, 20, cmap='cmr.sapphire', norm=norm)
ax.scatter(x.x, y.x, color="#f20a72", label="Optimum")

plt.colorbar(mappable, ax=ax)
plt.legend(loc="upper left")
plt.title(f"Rosenbrock 'banana' function, a={a}, b={b}")
plt.xlabel("x")
plt.ylabel("y")
plt.subplots_adjust(right=1, top=0.93)
plt.show()