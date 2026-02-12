"""
Experiment design approaches for generating sets of samples and their
concentrations given limits and an algorithm.
"""

import scipy
import pandas as pd
import numpy as np
from itertools import combinations
from typing import Annotated, Literal, TypeVar
import numpy.typing as npt

DType = TypeVar("DType", bound=np.generic)
Array1 = Annotated[npt.NDArray[DType], Literal[1]]
Array2 = Annotated[npt.NDArray[DType], Literal[2]]


def box_behnken_design(
    low: Array1[np.float64],
    high: Array1[np.float64],
    centers: int = 1,
) -> Array2[np.float64]:
    """
    Generate a Box-Behnken design matrix using the ranges given.

    Parameters
    ----------
    low: Array1[np.float64],
        Parameter lower limits.
    high: Array1[np.float64],
        Parameter upper limits.
    centers: int
        Number of center point replicates.

    Returns
    -------
    scaled_design: Array2[np.float64]
    """

    n_factors = low.shape[0]

    if n_factors < 3:
        raise ValueError("Box-Behnken requires at least 3 factors.")

    design_rows = []

    for i, j in combinations(range(n_factors), 2):
        two_factor_levels = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])

        for row in two_factor_levels:
            new_row = np.zeros(n_factors)
            new_row[i] = row[0]
            new_row[j] = row[1]
            design_rows.append(new_row)

    for _ in range(centers):
        design_rows.append(np.zeros(n_factors))

    design = np.array(design_rows)

    # center design around 0.5
    design = (design - design.min()) / (design - design.min()).max()

    # Scale the design to the bounds
    scaled_design = design * (high - low) + low

    return scaled_design


def full_factorial_design(
    low: Array1[np.float64],
    high: Array1[np.float64],
    levels: list[int] = [2],
) -> Array2[np.float64]:
    """
    Create a Full Factorial design scaled to the low/high concentration ranges
    supplied.

    Parameters
    ----------
    low: Array1[np.float64],
        Parameter lower limits.
    high: Array1[np.float64],
        Parameter upper limits.
    levels: int=1,
        Number of levels for the design (increments in each dimension, see also [pyDOE3 docs](https://pydoe3.readthedocs.io/en/latest/factorial.html#general-full-factorial))

    Returns
    -------
    scaled_design: Array2[np.float64]
    """

    grids = np.meshgrid(*[np.arange(level) for level in levels], indexing="ij")
    design = np.stack([g.flatten() for g in grids], axis=-1)

    # Scale the design to the bounds
    scaled_design = design * (high - low) + low

    return scaled_design


def latin_hypercube_design(
    low: Array1[np.float64],
    high: Array1[np.float64],
    n_samples: int = 10,
) -> Array2[np.float64]:
    """
    Create a design based on Latin Hypercube sampling, scaled to the low/high
    concentration ranges supplied.
    [Wikipedia page](https://en.wikipedia.org/wiki/Latin_hypercube_sampling)

    Parameters
    ----------
    low: Array1[np.float64],
        Parameter lower limits.
    high: Array1[np.float64],
        Parameter upper limits.
    n_samples: int=10,
        Number of samples to generate.

    Returns
    -------
    scaled_design: Array2[np.float64]
    """

    lhs_sampler = scipy.stats.qmc.LatinHypercube(low.shape[0])
    design = lhs_sampler.random(n_samples)

    # Scale the design
    scaled_design = design * (high - low) + low

    return scaled_design


def random_design(
    low: Array1[np.float64],
    high: Array1[np.float64],
    n_samples: int = 10,
    rng_seed: int = 42,
) -> Array2[np.float64]:
    """
    Create a design based on random sampling, scaled to the low/high
    concentration ranges supplied.

    Parameters
    ----------
    low: Array1[np.float64],
        Parameter lower limits.
    high: Array1[np.float64],
        Parameter upper limits.
    n_samples: int = 10,
        Number of samples to generate.
    rng_seed: int = 42,
        Seed for random number generator.

    Returns
    -------
    scaled_design: Array2[np.float64]
    """

    rng = np.random.default_rng(seed=rng_seed)

    design = rng.uniform(low=0.0, high=1.0, size=(n_samples, low.shape[0]))

    scaled_design = design * (high - low) + low

    return scaled_design


def sobol_design(
    low: Array1[np.float64],
    high: Array1[np.float64],
    n_samples: int = 10,
    rng_seed: int = 42,
) -> Array2[np.float64]:
    """
    Create a design based on Sobol sampling, scaled to the low/high
    concentration ranges supplied.
    [Wikipedia entry](https://en.wikipedia.org/wiki/Sobol_sequence)

    Parameters
    ----------
    low: Array1[np.float64],
        Parameter lower limits.
    high: Array1[np.float64],
        Parameter upper limits.
    n_samples: int = 10,
        Number of samples to generate.
    rng_seed: int = 42,
        Seed for random number generator.

    Returns
    -------
    scaled_design: Array2[np.float64]
    """

    use_points = 1 << (n_samples - 1).bit_length()

    if use_points != n_samples:
        print(
            f"""
            Using {use_points} points for Sobol sampling, rather than the
            specified {n_samples} (The balance properties of Sobol points
            require n to be a power of 2)."""
        )

    sobol_sampler = scipy.stats.qmc.Sobol(d=low.shape[0])
    design = sobol_sampler.random(use_points)

    scaled_design = design * (high - low) + low

    return scaled_design


def main():
    example_full_factorial_design()
    example_box_behnken_design()


def example_box_behnken_design():
    """
    Run an example design process.
    """
    print("Running example box behnken design strategy.")

    # Specify input information
    exp_code = "BB001"

    # Generate the design
    n_factors = 3
    factor_names = [f"factor_{x}" for x in range(n_factors)]
    low = np.full(n_factors, 0.1)
    high = np.full(n_factors, 1.2)

    scaled_design = box_behnken_design(
        low,
        high,
        centers=3,
    )

    # Create output
    df = pd.DataFrame(scaled_design, columns=[x for x in factor_names])
    sample_names = [f"{exp_code}_{i:03}" for i in range(scaled_design.shape[0])]
    df["sample_name"] = sample_names

    # Reorder columns
    cols = ["sample_name"] + [x for x in factor_names]
    df = df[cols]

    print(df.head())


def example_full_factorial_design():
    """
    Run an example design process.
    """
    print("Running example full_factorial_design strategy.")
    # Specify input information
    exp_code = "FF001"

    # Generate the design
    n_factors = 3
    factor_names = [f"factor_{x}" for x in range(n_factors)]
    low = np.full(n_factors, 0.1)
    high = np.full(n_factors, 1.2)

    scaled_design = full_factorial_design(
        low, high, levels=[5 for x in range(low.shape[0])]
    )

    # Create output
    df = pd.DataFrame(scaled_design, columns=[x for x in factor_names])
    sample_names = [f"{exp_code}_{i:03}" for i in range(scaled_design.shape[0])]
    df["sample_name"] = sample_names

    # Reorder columns
    cols = ["sample_name"] + [x for x in factor_names]
    df = df[cols]

    print(df.head())


if __name__ == "__main__":
    main()
