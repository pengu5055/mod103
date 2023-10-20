"""
This python file contains the definitions of the functions used in the
test suite for the optimization algorithms. It is essentially a summary
of this page:

https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""
import numpy as np

def rastrigin2(x1, x2, u1, v1, u2, v2, A=20):
    """
    Defines the Rastrigin function in 2D. This is a pathological function
    that looks like a bunch of hills and valleys in a grid pattern (at 
    least for n=2). 

    https://en.wikipedia.org/wiki/Rastrigin_function

    Parameters:
        x1, x2: Variables
        u1, v1, u2, v2: Auxiliary variables such that u*v = cos(2*pi*x)
        A: Constant

    Returns:
        The value of the Rastrigin function at the given point.    
    """


    return 2*A + x1**2 - A*u1*v1+ x2**2 - A*u2*v2

def plot_rastrigin2(x, A=20):
    """
    Defines the Rastrigin function in 2D, but in a way that is easier to
    plot. The reason why this is necessary is because the Rastrigin function
    is not convex and thus cannot be used in Gurobi. So we need to linearize
    it.

    https://en.wikipedia.org/wiki/Rastrigin_function

    Parameters:
        x: A 2D vector
        A: Constant

    Returns:
        The value of the Rastrigin function at the given point.
    """

    return 2*A + x[0]**2 - A*np.cos(2*np.pi*x[0]) + x[1]**2 - A*np.cos(2*np.pi*x[1])


def ackley2(z1, z2, a=20, b=0.2, c=2*np.pi):
    """
    Defines the Ackley function in 2D. This is a pathological function
    that looks like a bunch of hills and valleys in a grid pattern (at 
    least for n=2) but it is different from the Rastrigin function.

    https://en.wikipedia.org/wiki/Ackley_function

    Parameters:
        z1: Auxiliary variable such that z = e^{...} = 
            1 + 1/2 * {...} = exp(0.5*(cos(2*pi*x1) + cos(2*pi*x2))
        z2: Auxiliary variable such that z = e^{...} = 
            1 + 1/2 * {...} = exp(-0.2*sqrt(0.5*(x1**2 + x2**2)))

    Returns:
        The value of the Ackley function at the given point.    
    """
    return -20 * z2 - z1 + a + np.e + 20


def plot_ackley2(x, y):
    """
    Defines the Ackley function in 2D, but in a way that is easier to
    plot. The reason why this is necessary is because the Ackley function
    is not convex and thus cannot be used in Gurobi. So we need to linearize
    it.

    https://en.wikipedia.org/wiki/Ackley_function

    Parameters:
        x, y: Variables
    
    Returns:
        The value of the Ackley function at the given point.
    """
    return -20 * np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) - np.exp(0.5*(np.cos(2*np.pi*x) 
               + np.cos(2*np.pi*y))) + np.e + 20

def sphere2(x, y):
    """
    Defines the Sphere function in 2D. 

    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Parameters:
        x, y: Variables

    Returns:
        The value of the Sphere function at the given point.    
    """
    return x**2 + y**2

def plot_sphere2(x, y):
    """
    Defines the Sphere function in 2D, but in a way that is easier to
    plot. The reason why this is necessary is because the Sphere function
    is not convex and thus cannot be used in Gurobi. So we need to linearize
    it.

    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Parameters:
        x, y: Variables

    Returns:
        The value of the Sphere function at the given point.    
    """
    return x**2 + y**2

def rosenbrock2(x, y, v, a=1, b=100):
    """
    Defines the Rosenbrock function in 2D. 

    https://en.wikipedia.org/wiki/Rosenbrock_function

    Parameters:
        x, y: Variables
        v: Auxiliary variable such that v = x**2
        a, b: Constants

    Returns:
        The value of the Rosenbrock function at the given point.    
    """
    return (a - x)**2 + b*(y - v)**2

def rosenbrock2_plot(x, y, a=1, b=100):
    """
    Defines the Rosenbrock function in 2D, but in a way that is easier to
    plot. The reason why this is necessary is because the Rosenbrock function
    is not convex and thus cannot be used in Gurobi. So we need to linearize
    it.

    https://en.wikipedia.org/wiki/Rosenbrock_function

    Parameters:
        x, y: Variables
        a, b: Constants

    Returns:
        The value of the Rosenbrock function at the given point.
    """
    return (a - x)**2 + b*(y - x**2)**2


def bukinN62(z1, z2):
    """
    Defines the Bukin function N. 6 in 2D.

    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Parameters:
        z1: Auxiliary variable such that z1 = |y - 0.01*x**2|
        z2: Auxiliary variable such that z2 = |x + 10|
    
    Returns:
        The value of the Bukin function N. 6 at the given point.
    """
    return 100*np.sqrt(z1) + 0.01*z2