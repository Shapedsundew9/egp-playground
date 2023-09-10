"""Linear regression evolution functions.

The objective is to evolve a linear regression model that predicts m and c in
y = mx + c given a set of x and y values.

Fitness is calculated as the sum of the squared errors between the evolved m & c
and that found by scipy linregress.
"""
from typing import Callable
from math import pi
from logging import DEBUG, Logger, NullHandler, getLogger
from numpy import array, tan, mean, empty, real
from numpy.random import default_rng, Generator
from scipy.stats import linregress


_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG: bool = _logger.isEnabledFor(DEBUG)


# Data generation parameters
# 'seed' is the seed for the random number generator
# See generate_x() for details of 'scale' & N
# See add_noise() for details of 'noise'
# 'samples' is the number of samples in the data set
# 'threshold' is the fitness threshold that must be met or exceeded to move to the next difficulty level
# 'data' is the generated samples & expected results for the given difficulty level
PARAMS = {
    "trivial": {
        "seed": 1,
        "scale": 10,
        "N": 10,
        "noise": 0.01,
        "samples": 10,
        "threshold": 0.500,
        "data": None,
    },
    "easy": {
        "seed": 2,
        "scale": 100,
        "N": 100,
        "noise": 0.10,
        "samples": 100,
        "threshold": 0.850,
        "data": None,
    },
    "medium": {
        "seed": 3,
        "scale": 1000,
        "N": 1000,
        "noise": 0.25,
        "samples": 200,
        "threshold": 0.950,
        "data": None,
    },
    "hard": {
        "seed": 4,
        "scale": 10000,
        "N": 5000,
        "noise": 0.50,
        "samples": 300,
        "threshold": 0.990,
        "data": None,
    },
    "extreme": {
        "seed": 5,
        "scale": 100000,
        "N": 10000,
        "noise": 1.00,
        "samples": 1000,
        "threshold": 0.999,
        "data": None,
    },
}


def preload_function():
    """In this case we don't need to preload anything.
    Data is generated.
    """


def generate_x(
    scale: real, N: int, rng: Generator, narrow: bool = False
) -> tuple[array]:
    """Generate the N x values for the linear regression model.

    -scale <= x <= scale

    If narrow is True, then x is narrowed to be between two random values
    in the range -scale to scale.
    """
    if narrow:
        x1 = rng.uniform(-scale, scale)
        x2 = rng.uniform(-scale, scale)
        x = rng.uniform(min(x1, x2), max(x1, x2), N)
    else:
        x = rng.uniform(-scale, scale, N)
    return x


def add_noise(x: array, y: array, noise: real, rng: Generator) -> tuple[array, array]:
    """Add noise to the x & y values.

    The noise added is generated from a uniform distribution and scaled by a factor
    between two random points in the range -noise and noise.
    """
    n1 = rng.uniform(-noise, noise)
    n2 = rng.uniform(-noise, noise)
    x += rng.uniform(min(n1, n2), max(n1, n2), len(x)) * (x.max() - x.min()) / len(x)
    y += rng.uniform(min(n1, n2), max(n1, n2), len(y)) * (y.max() - y.min()) / len(y)
    return x, y


def sample(num: int, difficulty: str) -> tuple[array, array, real, real]:
    """Lazy generation of data for the given difficulty level and sample number."""
    param_set = PARAMS[difficulty]
    if param_set["data"] is None:
        rng = default_rng(param_set["seed"])
        _logger.info(f"Generating data for {difficulty} data samples...")
        param_set["data"] = {
            "x": empty((param_set["samples"], param_set["N"]), dtype=real),
            "y": empty((param_set["samples"], param_set["N"]), dtype=real),
            "f_m": empty((param_set["samples"],), dtype=real),
            "f_c": empty((param_set["samples"],), dtype=real),
        }
        for sample_num in range(param_set["samples"]):
            m = tan(rng.uniform(-pi, pi))
            c = rng.uniform(-param_set["scale"], param_set["scale"])
            x = generate_x(param_set["scale"], param_set["N"], rng)
            y = m * x + c
            x, y = add_noise(x, y, param_set["noise"], rng)
            f_m, f_c, _, _, _ = linregress(x, y)
            param_set["data"]["x"][sample_num] = x
            param_set["data"]["y"][sample_num] = y
            param_set["data"]["f_m"][sample_num] = f_m
            param_set["data"]["f_c"][sample_num] = f_c
        _logger.info(f"Data for {difficulty} data samples generated.")
    return (
        param_set["data"]["x"][num],
        param_set["data"]["y"][num],
        param_set["data"]["f_m"][num],
        param_set["data"]["f_c"][num],
    )


def fitness_function(individual: Callable[[array, array], tuple[real, real]]) -> real:
    """The individual is a function that takes an array of x values and an array of
     y values and returns a tuple of m and c.

    The fitness function is the sum of the squared errors between the evolved m & c
    and that found by scipy linregress.

    To reduce computation time, we use a subset of the data to calculate the fitness
    increasing the number of samples if the fitness is good enough.

    For reproducability we use the same data for each individual.
    """
    sample_fitness = []
    for difficulty, param_set in PARAMS.items():
        if _LOG_DEBUG:
            _logger.debug(f"Running fitness_function with {difficulty} parameters")
        for sample_num in range(param_set["samples"]):
            x, y, f_m, f_c = sample(sample_num, difficulty)
            i_m, i_c = individual(x, y)
            sample_fitness.append(1.0 - ((i_m - f_m) ** 2 + (i_c - f_c) ** 2) / 2.0)
        fitness = 1.0 - mean((1.0 - sample_fitness) ** 2)
        if fitness < param_set["threshold"]:
            break
    return fitness


if __name__ == "__main__":
    # Do some characterization of the data sets
    # For each difficulty level:
    #    1. Plot the distribution of graident angles
    #    2. Plot the distribution of intercepts
    #    3. Plot the distribution of gradients on a unit circle as arrows
    #    4. Plot the x-ranges sequentially starting with lowest xmin
    #    5. Plot the r**2 value distribution
    pass
