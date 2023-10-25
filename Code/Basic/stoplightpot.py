"""
The idea here is to plot the stoplight potential in 2D. Since
the first term is well defined we can take 3 terms total as that will
produce 2 dimensions.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib as mpl
from src import *
import palettable as pl
import cmasher as cmr

# --- Default Parameters ---
n = 100  # Number of time steps
v_0 = 5  # Initial velocity
l = 100 # Length till stoplight
t_step = 1  # Time step
KAPPA = 0.001  # Penalty parameter

lag = lambda v: 1/2 * ((v_0 - 0)/t_step)**2 + \
                       ((v[0] - v_0)/t_step)**2 + \
                 1/2 * ((v[1] - v[0])/t_step)**2

constraint = lambda v: 1 + np.exp(KAPPA * (1/2*v_0 + v[0] + 1/2*v[1] - l/t_step))

func = lambda v: lag(v) + constraint(v)

# --- Plot ---
fig, ax = plt.subplots()

v1 = np.linspace(-1000, 1000, 10000)
v2 = np.linspace(-1000, 1000, 10000)

V1, V2 = np.meshgrid(v1, v2)

Z = func([V1, V2])

cm = pl.cartocolors.sequential.Magenta_7.mpl_colormap#.reversed()
norm = mpl.colors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cm)

ax.contourf(V1, V2, Z, cmap=cm, levels=100, norm=norm)
ax.contour(V1, V2, Z, colors='black', levels=35, linewidths=0.5)

cbar = fig.colorbar(mappable, ax=ax)

ax.set_title("Stoplight Lagrangian w/o distance penalty term")
ax.set_xlabel(r"$v_1$")
ax.set_ylabel(r"$v_2$")

textstr = '\n'.join((
    r'$v_0=%.2f$' % (v_0, ),
    r'$l=%.2f$' % (l, ),
    r'$\Delta t=%.2f$' % (t_step, ),
    r'$\kappa=%.2f$' % (KAPPA)))

props = dict(boxstyle='round', facecolor='#c7b5c3', alpha=0.5)

# place a text box in upper left in axes coords
ax.text(0.03, 0.25, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.show()
