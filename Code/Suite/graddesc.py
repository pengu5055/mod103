"""
The aim here is to create a basic gradient descent algorithm
to optimize a function. This is so that I can learn how some of
these minimization algorithms work and I can compare my results
to the results of the more advanced algorithms.
"""
import numpy as np
from typing import Callable, Iterable


def gradient_descent(
        gradient: Callable,
        start: Iterable[float],
        learn_rate: float,
        n_iter:int = 50,
        tolerance:float = 1e-06,
    ):
    """
    Gradient descent algorithm for optimization of a function.
    """
    vector = np.array(start)

    for _ in range(n_iter):
        diff = -learn_rate * np.array(gradient(vector))
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff

    return vector
