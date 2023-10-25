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

        f(x, y) = f(0, 0) = 0

    https://en.wikipedia.org/wiki/Rastrigin_function

    Parameters:
        x1, x2: Variables
        u1, v1, u2, v2: Auxiliary variables such that u*v = cos(2*pi*x)
        A: Constant

    Returns:
        The value of the Rastrigin function at the given point.    
    """
    return 2*A + x1**2 - A*u1*v1+ x2**2 - A*u2*v2

def nonlin_rastrigin2(x, y, A=20):
    """
    Defines the Rastrigin function in 2D, but in a way that is easier to
    plot. The reason why this is necessary is because the Rastrigin function
    is not convex and thus cannot be used in Gurobi. So we need to linearize
    it.

        f(x, y) = f(0, 0) = 0

    https://en.wikipedia.org/wiki/Rastrigin_function

    Parameters:
        x: A 2D vector
        A: Constant

    Returns:
        The value of the Rastrigin function at the given point.
    """

    return 2*A + x**2 - A*np.cos(2*np.pi*x) + y**2 - A*np.cos(2*np.pi*y)


def ackley2(z1, z2, a=20, b=0.2, c=2*np.pi):
    """
    Defines the Ackley function in 2D. This is a pathological function
    that looks like a bunch of hills and valleys in a grid pattern (at 
    least for n=2) but it is different from the Rastrigin function.

        f(x, y) = f(0, 0) = 0    

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


def nonlin_ackley2(x, y):
    """
    Defines the Ackley function in 2D, but in a way that is easier to
    plot. The reason why this is necessary is because the Ackley function
    is not convex and thus cannot be used in Gurobi. So we need to linearize
    it.

        f(x, y) = f(0, 0) = 0

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

        f(x, y) = f(0, 0) = 0

    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Parameters:
        x, y: Variables

    Returns:
        The value of the Sphere function at the given point.    
    """
    return x**2 + y**2


def nonlin_sphere2(x, y):
    """
    Defines the Sphere function in 2D, but in a way that is easier to
    plot. The reason why this is necessary is because the Sphere function
    is not convex and thus cannot be used in Gurobi. So we need to linearize
    it.

        f(x, y) = f(0, 0) = 0

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
    Has a global minimum at (1, 1).

        f(x, y) = f(1, 1) = 0

    https://en.wikipedia.org/wiki/Rosenbrock_function

    Parameters:
        x, y: Variables
        v: Auxiliary variable such that v = x**2
        a, b: Constants

    Returns:
        The value of the Rosenbrock function at the given point.    
    """
    return (a - x)**2 + b*(y - v)**2


def nonlin_rosenbrock2(x, y, a=1, b=100):
    """
    Defines the Rosenbrock function in 2D, but in a way that is easier to
    plot. The reason why this is necessary is because the Rosenbrock function
    is not convex and thus cannot be used in Gurobi. So we need to linearize
    it. Has a global minimum at (0, 0).

        f(x, y) = f(1, 1) = 0

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
    Defines the Bukin function N. 6 in 2D. Which has a global minimum at
    (-10, 1) and is shaped like a valley with a steep cliff on the left
    side.

        f(x, y) = f(-10, 1) = 0

    The variables z1 and z2 are auxiliary variables that are defined as:

        z1 = |y - 0.01*x**2|
        z2 = |x + 10|

    More information can be found here:
        https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Parameters:
        z1: Auxiliary variable such that z1 = |y - 0.01*x**2|
        z2: Auxiliary variable such that z2 = |x + 10|
    
    Returns:
        The value of the Bukin function N. 6 at the given point.
    """
    return 100*np.sqrt(z1) + 0.01*z2


def nonlin_bukinN62(x, y):
    """
    Defines the Bukin function N. 6 in 2D, but in a way that is easier to
    plot. The reason why this is necessary is because the Bukin function N. 6
    is not convex and thus cannot be used in Gurobi. So we need to linearize
    it.

        f(x, y) = f(-10, 1) = 0

    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Parameters:
        x, y: Variables

    Returns:
        The value of the Bukin function N. 6 at the given point.
    """
    return 100*np.sqrt(np.abs(y - 0.01*x**2)) + 0.01*np.abs(x + 10)


def leviN132(z1, z2, z3, z4, z5):
    """
    Defines the Levi function N. 13 in 2D. Which has a global minimum at
    (1, 1) and is shaped like a ribbed bowl.

        f(x, y) = f(1, 1) = 0

    More information can be found here:
        https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Parameters:
        z1: Auxiliary variable such that z1 = np.sin(3*np.pi*x)**2
        z2: Auxiliary variable such that z2 = np.sin(3*np.pi*y)**2
        z3: Auxiliary variable such that z3 = np.sin(2*np.pi*y)**2
        z4: Auxiliary variable such that z4 = (x - 1)**2
        z5: Auxiliary variable such that z5 = (y - 1)**2

    """
    return z1 + z4 * (1 + z2) + z5 * (1 + z3)


def nonlin_leviN132(x, y):
    """
    Defines the Levi function N. 13 in 2D, but in a way that is easier to
    plot. The reason why this is necessary is because the Levi function N. 13
    is not convex and thus cannot be used in Gurobi. So we need to linearize
    it.
        
         f(x, y) = f(1, 1) = 0

    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Parameters:
        x, y: Variables
    
    Returns:
        The value of the Levi function N. 13 at the given point.
    """
    return np.sin(3*np.pi*x)**2 + (x - 1)**2 * (1 + np.sin(3*np.pi*y)**2) + \
              (y - 1)**2 * (1 + np.sin(2*np.pi*y)**2)


def himmelblau2(x1, x2, v1, v2):
    """
    Defines the Himmelblau function in 2D. Which has four global minima at
    (3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)
    and is shaped like a bowl with four holes in it. This function is
    particularly interesting because it has multiple local minima ie. it is
    multi-modal.

        f(x, y) = f(3, 2) = f(-2.805118, 3.131312) = f(-3.779310, -3.283186) = 
                  f(3.584428, -1.848126) = 0

    More information can be found here:
        https://en.wikipedia.org/wiki/Himmelblau%27s_function

    Parameters:
        x1, x1: Variables
        v1: Auxiliary variable such that v1 = x**2
        v2: Auxiliary variable such that v2 = y**2
    
    Returns:
        The value of the Himmelblau function at the given point.
    """
    return (v1 + x2 - 11)**2 + (x1 + v2 - 7)**2


def nonlin_himmelblau2(x, y):
    """
    Defines the Himmelblau function in 2D, but in a way that is easier to
    plot. The reason why this is necessary is because the Himmelblau function
    is not convex and thus cannot be used in Gurobi. So we need to linearize
    it.

        f(x, y) = f(3, 2) = f(-2.805118, 3.131312) = f(-3.779310, -3.283186) = 
                  f(3.584428, -1.848126) = 0

    https://en.wikipedia.org/wiki/Himmelblau%27s_function

    Parameters:
        x, y: Variables
    
    Returns:
        The value of the Himmelblau function at the given point.
    """
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def cross_in_tray2(w):
    """
    Defines the Cross-in-tray function in 2D. Which has four global minima at
    (1.34941, -1.34941), (1.34941, 1.34941), (-1.34941, 1.34941), (-1.34941, -1.34941)
    and is shaped like a bowl with four holes in it. This function is
    particularly interesting because it has multiple local minima ie. it is
    multi-modal.

        f(x, y) = f(1.34941, -1.34941) = f(1.34941, 1.34941) = 
                  f(-1.34941, 1.34941) = f(-1.34941, -1.34941) = -2.06261

    More information can be found here:
        https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Parameters:
        z1: Auxiliary variable such that z1 = np.sin(x)
        z2: Auxiliary variable such that z2 = np.sin(y)
        z3: Auxiliary variable such that z3 = |100 - sqrt(x**2 + y**2)/pi|
        w: Auxiliary variable such that w = |z1*z2*exp(z3) + 1|
    
    Returns:
        The value of the Cross-in-tray function at the given point.
    """
    return -0.0001*(w + 1)**0.1


def nonlin_cross_in_tray2(x, y):
    """
    Defines the Cross-in-tray function in 2D, but in a way that is easier to
    plot. The reason why this is necessary is because the Cross-in-tray function
    is not convex and thus cannot be used in Gurobi. So we need to linearize
    it.

        f(x, y) = f(1.34941, -1.34941) = f(1.34941, 1.34941) = 
                  f(-1.34941, 1.34941) = f(-1.34941, -1.34941) = -2.06261

    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Parameters:
        x, y: Variables
    
    Returns:
        The value of the Cross-in-tray function at the given point.
    """
    return -0.0001*(np.abs(np.sin(x)*np.sin(y)*np.exp(np.abs(100 - np.sqrt(x**2 + y**2)/np.pi))) + 1)**0.1


def eggholder2(x, y, z1, z2):
    """
    Defines the Eggholder function in 2D. Which has a global minima at
    (512, 404.2319) and is like a crisscross pattern.

        f(x, y) = f(512, 404.2319) = -959.6407

    More information can be found here:
        https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Parameters:
        x, y: Variables
        z1: Auxiliary variable such that z1 = sin(sqrt(|x/2 + (y + 47)|))
        z2: Auxiliary variable such that z2 = sin(sqrt(|x - (y + 47)|))
    
    Returns:
        The value of the Eggholder function at the given point.
    """
    return -(y + 47)*z1 - x*z2


def nonlin_eggholder2(x, y):
    """
    Defines the Eggholder function in 2D, but in a way that is easier to
    plot. The reason why this is necessary is because the Eggholder function
    is not convex and thus cannot be used in Gurobi. So we need to linearize
    it.

        f(x, y) = f(512, 404.2319) = -959.6407

    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Parameters:
        x, y: Variables
    
    Returns:
        The value of the Eggholder function at the given point.
    """
    return -(y + 47)*np.sin(np.sqrt(np.abs(x/2 + (y + 47)))) - x*np.sin(np.sqrt(np.abs(x - (y + 47))))


def holder_table2(w):
    """
    Defines the Holder table function in 2D. Which has a global minima at
    (8.05502, 9.66459) and is shaped like a picnic blanket or a table cloth.

        f(x, y) = f(8.05502, 9.66459) = -19.2085

        [(8.05502, 9.66459), (-8.05502, 9.66459), (8.05502, -9.66459), (-8.05502, -9.66459)]
    
    More information can be found here:
        https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Parameters:
        w: Auxiliary variable such that 
            w = -|sin(x)*cos(y)*exp(|1 - sqrt(x**2 + y**2)/pi)|

    Returns:
        The value of the Holder table function at the given point.
    """

def nonlin_holder_table2(x, y):
    """
    Defines the Holder table function in 2D, but in a way that is easier to
    plot. The reason why this is necessary is because the Holder table function
    is not convex and thus cannot be used in Gurobi. So we need to linearize
    it.

        f(x, y) = f(8.05502, 9.66459) = -19.2085

    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Parameters:
        x, y: Variables
    
    Returns:
        The value of the Holder table function at the given point.
    """
    return -np.abs(np.sin(x)*np.cos(y)*np.exp(np.abs(1 - np.sqrt(x**2 + y**2)/np.pi)))


def schafferN22(w1, w2):
    """
    Defines the Schaffer function N. 2 in 2D. Which has a global minima at
    (0, 0) and is shaped like a bunch of dots in a grid pattern.

        f(x, y) = f(0, 0) = 0

    More information can be found here:
        https://en.wikipedia.org/wiki/Test_functions_for_optimization
    
    Parameters:
        w1: Auxiliary variable such that w1 = sin(x**2 - y**2)**2
        w2: Auxiliary variable such that w2 = 1/(1 + 0.001*(x**2 + y**2))**2
    """
    return 0.5 + (w1 - 0.5)/(w2)


def nonlin_schafferN22(x, y):
    """
    Defines the Schaffer function N. 2 in 2D, but in a way that is easier to
    plot. The reason why this is necessary is because the Schaffer function N. 2
    is not convex and thus cannot be used in Gurobi. So we need to linearize
    it.

        f(x, y) = f(0, 0) = 0

    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    
    Parameters:
        x, y: Variables
    
    Returns:
        The value of the Schaffer function N. 2 at the given point.
    """
    return 0.5 + (np.sin(x**2 - y**2) - 0.5)/(1 + 0.001*(x**2 + y**2))**2


def schafferN42(w1, w2):
    """
    Defines the Schaffer function N. 4 in 2D. Which has a global minima at
    (0, 1.25313) and is shaped like a bunch of dots in a grid pattern.

        f(x, y) = f(0, 1.25313) = f(0, -1.25313) = f(1.25313, 0) = f(-1.25313, 0) 
            = 0.292579

        1.253131828792882 -> Coordinate 
        0.292578632035980 -> Value

    More information can be found here:
        https://en.wikipedia.org/wiki/Test_functions_for_optimization
    
    Parameters:
        w1: Auxiliary variable such that w1 = cos(sin(x**2 - y**2))**2
        w2: Auxiliary variable such that w2 = 1/(1 + 0.001*(x**2 + y**2))**2
    """
    return 0.5 + (w1 - 0.5)/(w2)


def nonlin_schafferN42(x, y):
    """
    Defines the Schaffer function N. 4 in 2D, but in a way that is easier to
    plot. The reason why this is necessary is because the Schaffer function N. 4
    is not convex and thus cannot be used in Gurobi. So we need to linearize
    it.

        f(x, y) = f(0, 1.25313) = f(0, -1.25313) = f(1.25313, 0) = f(-1.25313, 0) 
            = 0.292579

        1.253131828792882 -> Coordinate 
        0.292578632035980 -> Value

    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    
    Parameters:
        x, y: Variables
    
    Returns:
        The value of the Schaffer function N. 4 at the given point.
    """
    return 0.5 + (np.cos(np.sin(x**2 - y**2))**2 - 0.5)/(1 + 0.001*(x**2 + y**2))**2
