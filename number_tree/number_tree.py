"""Number tree problem."""
from logging import DEBUG, Logger, NullHandler, getLogger
from random import randint, seed

from  matplotlib import pyplot as plt
from numpy import arange, array, meshgrid, sqrt
from tqdm import trange

from fitness_function import X, Y, Z


_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG: bool = _logger.isEnabledFor(DEBUG)


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
