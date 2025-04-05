"""
WIP
"""
import numpy as np
from .routines import calculate_stock_volumes
from .routines import validate_or_optimize
from typing import Annotated, Literal, TypeVar
import numpy.typing as npt


DType = TypeVar("DType", bound=np.generic)

Array1 = Annotated[npt.NDArray[DType], Literal[1]]
Array2 = Annotated[npt.NDArray[DType], Literal[2]]


def pass_case(
    stock_c: Array1[np.float64],
    targets_c: Array2[np.float64],
    targets_v: Array2[np.float64],
    bounds: list[tuple[float]],
):
    """
    Parameters
    ----------
    stock_c: np.array
        Concentrations of stock solutions (n x 1 for n stocks).

    targets_c: np.array
        Concentrations of stock solutions (m x n for m samples and n stocks).

    targets_v: np.array
        Intended target solution volumes (m x 1 for m samples).

    bounds: list(tuple(float))
        Bounds for each stock concentration (n x 2 for n stocks).
    """

    print("\nRunning pass case test.")
    print("-----------------------")

    optimized_stock_c = validate_or_optimize(stock_c, targets_c, targets_v, bounds)

    optimised_stock_v, optimised_excess_volume = calculate_stock_volumes(
        optimized_stock_c, targets_c, targets_v
    )

    print("Optimized stock concentrations:\n", optimized_stock_c)
    print("Optimized stock volumes:\n", optimised_stock_v)
    print("Minimum stock volumes:\n", optimised_stock_v.sum(axis=0))
    print("Optimized solvent additions:\n", optimised_excess_volume)


def fail_case(
    stock_c: Array1[np.float64],
    targets_c: Array2[np.float64],
    targets_v: Array2[np.float64],
    bounds: list[tuple[float]],
):
    """
    Parameters
    ----------
    stock_c: np.array
        Concentrations of stock solutions (n x 1 for n stocks).

    targets_c: np.array
        Concentrations of stock solutions (m x n for m samples and n stocks).

    targets_v: np.array
        Intended target solution volumes (m x 1 for m samples).

    bounds: list(tuple(float))
        Bounds for each stock concentration (n x 2 for n stocks).
    """

    print("\nRunning fail case test.")
    print("-----------------------")
    # Provide proposed experiment design

    ## Stocks
    # Concentrations of four stock solutions, one for each component
    stock_c = np.array([0.2, 0.4, 0.5, 0.4])

    ## Targets
    targets_c = np.array(
        [
            [0.1, 0.4, 0.6, 0.3],  # sample 1
            [0.5, 0.3, 0.7, 1.2],  # sample 2
            [0.6, 0.8, 0.3, 0.1],  # sample 3
        ]
    )  # Concentrations of 3 samples, four components

    targets_v = np.ones((3)) * 0.1  # Volumes for 3 samples

    optimized_stock_c = validate_or_optimize(stock_c, targets_c, targets_v, bounds)

    optimised_stock_v, optimised_excess_volume = calculate_stock_volumes(
        optimized_stock_c, targets_c, targets_v
    )

    print("Optimized stock concentrations:\n", optimized_stock_c)
    print("Optimized stock volumes:\n", optimised_stock_v)
    print("Minimum stock volumes:\n", optimised_stock_v.sum(axis=0))
    print("Optimized solvent additions:\n", optimised_excess_volume)


def test():
    # Provide proposed experiment design

    ## Stocks
    # Concentrations of four stock solutions, one for each component
    stock_c = np.array([0.2, 0.4, 0.5, 0.4])
    # Bounds for stock_c to ensure values are physically realistic (non-negative)
    bounds = [(0.001, 4)] * len(stock_c)

    ## Targets
    targets_c = np.array(
        [
            [0.1, 0.4, 0.6, 0.3],  # sample 1
            [0.5, 0.3, 0.7, 1.2],  # sample 2
            [0.6, 0.8, 0.3, 0.1],  # sample 3
        ]
    )  # Concentrations of 3 samples, four components

    targets_v = np.ones((3)) * 0.1  # Volumes for 3 samples

    optimized_stock_c = validate_or_optimize(stock_c, targets_c, targets_v, bounds)

    optimised_stock_v, optimised_excess_volume = calculate_stock_volumes(
        optimized_stock_c, targets_c, targets_v
    )

    print("Optimized stock concentrations:\n", optimized_stock_c)
    print("Optimized stock volumes:\n", optimised_stock_v)
    print("Minimum stock volumes:\n", optimised_stock_v.sum(axis=0))
    print("Optimized solvent additions:\n", optimised_excess_volume)
