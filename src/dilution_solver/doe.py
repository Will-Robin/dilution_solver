import scipy
import pandas as pd
import numpy as np
import pyDOE3
import matplotlib.pyplot as plt
from typing import Annotated, Literal, TypeVar
import numpy.typing as npt

DType = TypeVar("DType", bound=np.generic)
Array1 = Annotated[npt.NDArray[DType], Literal[1]]
Array2 = Annotated[npt.NDArray[DType], Literal[2]]


def box_behnken_design(
    n_factors: int,
    low: Array1[np.float64],
    high: Array1[np.float64],
) -> Array2[np.float64]:
    # Generate the design
    design = pyDOE3.bbdesign(n_factors)
    # center design around 0.5
    design = (design - design.min()) / (design - design.min()).max()

    # Scale the design to the bounds
    scaled_design = design * (high - low) + low

    return scaled_design


def full_factorial_design(
    n_factors: int,
    low: Array1[np.float64],
    high: Array1[np.float64],
    levels: list[int] = [2],
) -> Array2[np.float64]:
    design = pyDOE3.fullfact(levels)

    # Scale the design to the bounds
    scaled_design = design * (high - low) + low

    return scaled_design


def latin_hypercube_design(
    n_factors: int,
    low: Array1[np.float64],
    high: Array1[np.float64],
    n_samples: int = 10,
) -> Array2[np.float64]:
    lhs_sampler = scipy.stats.qmc.LatinHypercube(low.shape[0])
    design = lhs_sampler.random(n_samples)

    # Scale the design to continuous space
    scaled_design = design * (high - low) + low

    return scaled_design


def random_design(
    n_factors: int,
    low: Array1[np.float64],
    high: Array1[np.float64],
    n_samples: int = 10,
    rng_seed: int = 42,
) -> Array2[np.float64]:
    np.random.seed(rng_seed)

    # Create a random grid scaled between 0 and 1
    design = np.random.rand(n_samples, low.shape[0])

    scaled_design = design * (high - low) + low

    return scaled_design


def sobol_design(
    n_factors: int,
    low: Array1[np.float64],
    high: Array1[np.float64],
    n_samples: int = 10,
    rng_seed: int = 42,
) -> Array2[np.float64]:
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
    print("Testing full_factorial_design strategy.")
    # Specify input information
    exp_code = "EXP001"

    # Generate the design
    n_factors = 3
    factor_names = [f"factor_{x}" for x in range(n_factors)]
    low = np.full(n_factors, 0.1)
    high = np.full(n_factors, 1.2)
    scaled_design = full_factorial_design(
        n_factors, low, high, levels=[5 for x in range(low.shape[0])]
    )

    # Create output
    df = pd.DataFrame(scaled_design, columns=[x for x in factor_names])
    sample_names = [f"{exp_code}_{i:03}" for i in range(scaled_design.shape[0])]
    df["sample_name"] = sample_names

    # Reorder columns
    cols = ["sample_name"] + [x for x in factor_names]
    df = df[cols]

    print(df.head())
    pd.plotting.scatter_matrix(df)
    plt.show()


if __name__ == "__main__":
    main()
