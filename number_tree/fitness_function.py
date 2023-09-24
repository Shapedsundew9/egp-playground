"""Number tree evolution functions.

The objective is to evolve an expression from two integer values, X & Y, and ONLY the + and - operators
to calculate a target value, Z.

i.e. aX - bX + cY - dY = Z

For example, given the values X=3 and Y=5, and a target value of Z=7, a valid expression could be
one of:
    a. 5 + 5 - 3 = 7 = 2Y - X (optimal i.e least + and - operations)
    b. 3 + 3 + 3 - 5 + 3 = 7 = 4X - Y
    c. 3 - 3 + 5 - 3 + 5 = 7 = 2Y + X - 2X
    etc.

Whilst contrived this problem is a good test of the evolution algorithm with easy to understand results.
"""
from typing import Callable, Any
from numpy import single


# Data generation parameters
# Optimal solution is 8Y - 2X
X = 3
Y = 5
Z = 44


# This structure is required by Erasmus
EGP_PROBLEM_CONFIG: dict[str, Any] = {
    "name": "Number Tree",                              # Optional but recommended
    "description": __doc__,                             # Optional but recommended
    "inputs": ["int", "int"],                           # Required
    "outputs": ["int"],                                 # Required
    "creator": "22c23596-df90-4b87-88a4-9409a0ea764f",  # Optional
}


def fitness_function(individual: Callable[[int, int], int]) -> single:
    """The fitness function is about the 'correctness' of the solution. It is the only
    required function in this module and must take a Callable and return the fitness as
    numoy single where:

    0.0 <= fitness <= 1.0

    For this problem the fitness is 1 - the square of the error in Z over Z**2.
    """
    return single(max(1.0 - ((Z - individual(X, Y)) / Z) ** 2, 1.0))
