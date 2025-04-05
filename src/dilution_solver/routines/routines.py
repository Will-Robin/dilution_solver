"""
Functions for calculating desired stock solution concentrations fit to a set of
sample concentrations and volumes.
"""
import numpy as np
from scipy.optimize import minimize
from typing import Annotated, Literal, TypeVar
import numpy.typing as npt


DType = TypeVar("DType", bound=np.generic)

Array1 = Annotated[npt.NDArray[DType], Literal[1]]
Array2 = Annotated[npt.NDArray[DType], Literal[2]]


def calculate_stock_volumes(
    stock_c: Array1[np.float64],
    targets_c: Array2[np.float64],
    targets_v: Array1[np.float64],
) -> tuple[Array2[np.float64], Array1[np.float64]]:
    """
    Find out volumes required to make target concentrations (targets_c), given
    stock concentrations (stock_c), then calculate the difference between the
    stock volumes required and the target volumes.

    Parameters
    ----------
    stock_c: Array1[np.float64]
        Concentrations of stock solutions (n x 1 for n stocks).

    targets_c: Array2[np.float64]
        Concentrations of stock solutions (m x n for m samples and n stocks).

    targets_v: Array1[np.float64]
        Intended target solution volumes (m x 1 for m samples).

    Returns
    -------
    (stock_v, excess_volume):, Array2[np.float64]) Array1[np.float64]
        The calculated amounts of stock solution required, and the difference
        between the intended sample volumes and the total calculated
        stock volumes per sample.
    """

    # First work out how many moles are required for each compound in each
    # sample
    targets_m = targets_c * targets_v[:, np.newaxis]

    # Now determine volumes from stocks to add to get desired number of moles
    stock_v = targets_m / stock_c

    # Compare the sum of stock volumes to the target volume
    targets_proposed_v = stock_v.sum(axis=1)
    excess_volume = targets_v - targets_proposed_v

    return stock_v, excess_volume


def evaluate_stock_concentrations(
    stock_c: Array1[np.float64],
    targets_c: Array2[np.float64],
    targets_v: Array2[np.float64],
) -> float:
    """
    Calculate the error of a proposed stock concentration (stock_c) for an
    experimental design, costed according to the difference betweem the target
    volumes (targets_v) and the sum of calculated stock volumes (per sample).

    Parameters
    ----------
    stock_c: Array1[np.float64]
        Concentrations of stock solutions (n x 1 for n stocks).

    targets_c: Array1[np.float64]
        Concentrations of stock solutions (m x n for m samples and n stocks).

    targets_v: Array1[np.float64]
        Intended target solution volumes (m x 1 for m samples).

    Returns
    -------
    objective: float
        Positive sum of the negative parts of the difference between targets_v and
        the sample-wise sum of stock volumes.
    """
    # Calculate the excess_volume
    _, excess_volume = calculate_stock_volumes(stock_c, targets_c, targets_v)

    # Objective: minimize the negative values (penalize negative excess_volume)
    objective = -np.sum(np.minimum(excess_volume, 0))
    return objective


def optimize_stock_concentrations(
    stock_c: Array1[np.float64],
    targets_c: Array2[np.float64],
    targets_v: Array2[np.float64],
    bounds: list[tuple[float]],
) -> Array1[np.float64]:
    """
    Choose optimal stock concentrations based on specified target
    concentrations (targets_c) and volumes (targets_v) within bounds
    (bounds).

    Parameters
    ----------
    stock_c: Array1[np.float64]
        Concentrations of stock solutions (n x 1 for n stocks).

    targets_c: Array2[np.float64]
        Concentrations of stock solutions (m x n for m samples and n stocks).

    targets_v: Array2[np.float64]
        Intended target solution volumes (m x 1 for m samples).

    bounds: list(tuple(float))
        Bounds for each stock concentration (n x 2 for n stocks).

    Returns
    -------
    optimized_stock_c: Array1[np.float64]
    """
    # Optimize
    result = minimize(
        evaluate_stock_concentrations,
        stock_c,
        args=(targets_c, targets_v),
        bounds=bounds,
        method="SLSQP",
    )

    # Optimized stock_c
    optimized_stock_c = result.x

    optimised_stock_v, optimised_excess_volume = calculate_stock_volumes(
        optimized_stock_c, targets_c, targets_v
    )

    # Check if the optimiser found a feasible result
    if np.any(optimised_excess_volume < 0):
        problem_idx = np.where(optimised_excess_volume < 0.0)[0]

        print("Optimisation failed.")
        print("The negative volumes in this array are the reason:")
        print(optimised_excess_volume)
        print("These samples fail the test:")
        print(targets_c[problem_idx])
        print("Consider increasing bounds for optimisation.")
        print()

    return optimized_stock_c


def validate_or_optimize(
    stock_c: Array1[np.float64],
    targets_c: Array2[np.float64],
    targets_v: Array2[np.float64],
    bounds: list[tuple[float]],
) -> Array1[np.float64]:
    """
    Test if an experiment design will be feasible. If not, attempt to optimise
    it so that it is feasible.

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

    Returns
    -------
    optimized_stock_c: Array1[np.float64]
        Potentially feasible stock concentrations, stock solution required, and the
        difference between the intended sample volumes and the total calculated
        stock volumes per sample.
    """
    # First, test if the concentrations need to be optimised.
    stock_v, excess_volume = calculate_stock_volumes(stock_c, targets_c, targets_v)

    if np.any(excess_volume < 0):
        optimized_stock_c = optimize_stock_concentrations(
            stock_c, targets_c, targets_v, bounds
        )
    else:
        optimized_stock_c = stock_c

    return optimized_stock_c
