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
from logging import DEBUG, Logger, NullHandler, getLogger
from random import randint, seed
from typing import Callable, cast

import matplotlib.pyplot as plt
from egp_population.population import population
from numpy import arange, array, int64, meshgrid, single, sqrt, bool_, full
from numpy.typing import NDArray
from tqdm import trange

_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG: bool = _logger.isEnabledFor(DEBUG)


# Data generation parameters
# Optimal solution is 8Y - 2X
X = 3
Y = 5
Z = 44


def preload_function() -> None:
    """In this case we don't need to preload anything.
    Data is generated.
    """


def fitness_function(individual: Callable[[int, int], int]) -> single:
    """The fitness function is about the 'correctness' of the solution.
    See the survivability function for what constitutes the best solution.

    0.0 <= fitness <= 1.0

    For this problem the fitness is 1 - the square of the error in Z over Z**2.
    """
    return single(max(1.0 - ((Z - individual(X, Y)) / Z) ** 2, 1.0))


def survivability_function(
    populous: population,
) -> tuple[NDArray[single], NDArray[bool_]] | tuple[NDArray[int64], NDArray[single], NDArray[bool_]]:
    """The survivability function is used to determine which solutions continue to be evolved & which will be deleted.
    It is called after each epoch and has 2 ways to define the next epoch:
        1. Set 'active' to True for the individuals to continue to be evolved. Survivability is ignored.
        2. Set 'active' to False for all individuals and 'active_size' individuals with the highest survivability will be evolved.
    NOTE: In the case of ties in survivability crossing the 'active_size' limit the individuals with the same survivability
    will be chosen at random until the limit is reached.

    Erasmus does not delete individuals unless memory is exhausted (so the total population can grow to be very large).
    The intention is not to close off evolutionary paths that may ultimately lead to better solutions. When forced to
    delete individuals only ones with no existing descendants will be deleted. The ones with the lowest survivability first.
    In the case of ties individuals will be chosen at random.

    Trivially, and by default, survivability == fitness.
        0.0 <= survivability <= 1.0
    However, the power of survivability is that it allows a more complex definition of what constitutes the best solution without
    obfuscating the fitness function and can be evaluated in the context of the population rather than the just the individual.
    For example, solution diversity can be encouraged by increasing the survivabilty of individuals that are structurally
    different. In the early stages of evolution a sub-population may show increased fitness and start to dominate the active
    population but run into a deadend, successively failing to improve. If survivability were based only on an individuals
    performance a slightly less fit individual with a different structure and different potential may be deleted before it has a
    chance to be evolved and that time/energy/cost would be lost.

    Returns
    -------
    tuple[array[single], array[bool]] = (survivability, active) for the whole population
    tuple[array[int64], array[single], array[bool]] = (ref, survivability, active) for a subset of the population
    """
    survivability: NDArray[single] = cast(NDArray[single], populous['fitness'])
    active: NDArray[bool_] = full(survivability.shape, False, dtype=bool_)
    return (survivability, active)


def random_solution(x: int, y: int, limit: int = 4 * Z) -> tuple[list[int], list[int]]:
    """Generate a random solution to the problem."""
    path: tuple[list[int], list[int]] = ([0], [0])
    z: int = 0
    while z != Z:
        path[0].append(path[0][-1])
        path[1].append(path[1][-1])
        match randint(0, 3):
            case 0:
                path[0][-1] += x
            case 1:
                path[0][-1] -= x
            case 2:
                path[1][-1] += y
            case 3:
                path[1][-1] -= y
        z = path[0][-1] + path[1][-1]
        if z > limit or z < -limit:
            path[0].pop()
            path[1].pop()
    return path


# Run as a script for testing
if __name__ == "__main__":
    # Set up problem space
    seed(1)
    z_limit = 2 * Z
    boundary = int(2 * Z / min(X, Y)) + 1
    x_space, y_space = meshgrid(arange(-boundary, boundary + 1) * X, arange(-boundary, boundary + 1) * Y)
    dz_space = sqrt(abs(x_space + y_space - Z))

    # Generate a random path to the solution
    random_path = array(random_solution(X, Y, z_limit))

    # Plot the random path
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 10)
    c = ax.pcolormesh(x_space, y_space, dz_space, cmap="RdBu", vmin=0, vmax=dz_space.max())
    ax.set_title(f"A Random Path to Solution. X={X}, Y={Y}, Z={Z}, length={len(random_path[0]) - 1}.")
    ax.axis([x_space.min(), x_space.max(), y_space.min(), y_space.max()])
    fig.colorbar(c, ax=ax)
    ax.plot(random_path[0], random_path[1], linewidth=2, color="black", label="random path")
    ax.plot([0], [0], marker="x", markersize=5, color="green", label="origin")
    ax.plot(
        random_path[0][-1],
        random_path[1][-1],
        markersize=5,
        marker="o",
        color="yellow",
        label="solution",
    )
    plt.savefig("random_path.png")

    # Generate a population of random solutions
    path_lengths: list[int] = [len(random_solution(X, Y, z_limit)[0]) for _ in trange(10000, desc="Generating random paths")]
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 10)
    ax.hist(path_lengths, bins=range(0, max(path_lengths) + 10, 10), color="blue")
    ax.set_title(f"Random Path Length Distribution. X={X}, Y={Y}, Z={Z}. 10k samples, bin width = 10.")
    ax.set_xlabel("Path length")
    ax.set_ylabel("Frequency")
    plt.savefig("random_path_distribution.png")
