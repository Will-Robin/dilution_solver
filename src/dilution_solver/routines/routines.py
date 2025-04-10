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


def calculate_sample_concentrations(
    stock_c: Array1[np.float64],
    stock_v: Array2[np.float64],
    targets_v: Array1[np.float64],
):
    """
    Calculate the concentration of a dilution scheme.

    Parameters
    ----------
    stock_c: Array1[np.float64]
        Concentrations of stock solutions (n x 1 for n stocks).

    stock_v: Array2[np.float64]
        Stock solution addition volumes.

    targets_v: Array1[np.float64]
        Intended target solution volumes (m x 1 for m samples).

    Returns
    -------
    concentrations: Array2[np.float64]
        Resulting array of concentrations.
    """

    concentrations = stock_v * stock_c / targets_v[:, np.newaxis]

    return concentrations


def evaluate_stock_concentrations(
    stock_c: Array1[np.float64],
    targets_c: Array2[np.float64],
    targets_v: Array1[np.float64],
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
    targets_v: Array1[np.float64],
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

    targets_v: Array1[np.float64]
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
    targets_v: Array1[np.float64],
    bounds: list[tuple[float]],
) -> Array1[np.float64]:
    """
    Test if an experiment design will be feasible. If not, attempt to optimise
    it so that it is feasible.

    Parameters
    ----------
    stock_c: np.array
        Concentrations of stock solutions (n x 1 for n stocks).

    targets_c:
        Concentrations of stock solutions (m x n for m samples and n stocks).

    targets_v: Array1[np.float64]
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


def create_sequential_dilutions(
    stock_volumes: Array2[np.float64],
    solvent_volumes: Array2[np.float64],
    stock_concs: Array1[np.float64],
    targets_c: Array2[np.float64],
    targets_v: Array1[np.float64],
    min_volume: float,
):
    """
    Finds which volumes in stock_volumes is lower than the minimum volume which
    can be transferred (min_volume), and creates new stock solutions which
    allow for sample transfer to go ahead.

    Parameters
    ----------
    stock_volumes: Array2[np.float64]
    solvent_volumes: Array2[np.float64]
    stock_concs: Array1[np.float64]
    targets_c: Array2[np.float64]
    targets_v: Array1[np.float64]
    min_volume: float

    Returns
    -------
    (new_stock_volumes, new_stock_concs, diluted_stocks):
        tuple[Array2[np.float64], Array1[np.float64], list[int]]
        A new stock pipetting scheme, with additional columns for the new stock
        solutions required. The new stock concentrations, with diluted stocks
        appended. The indices of which stocks are required to be diluted (from
        the original stock array).
    """
    new_stock_volumes = stock_volumes.copy()
    new_stock_concs = stock_concs.copy()
    diluted_stocks = []

    for i in range(0, stock_volumes.shape[1]):
        # First, find volumes which are lower than min_volume
        too_low = stock_volumes[:, i] < min_volume

        # TODO: account for not having to update all stocks for all unfeasible
        # samples - target the specific volumes, rather than the whole sample.
        if np.any(too_low):
            # Record that the stock was diluted
            diluted_stocks.append(i)

            # Determine which stocks need diluting
            # idx = indices of volumes which need updating
            idx = np.where(too_low)

            # Get the lowest unfeasible stock volumes per stock
            min_vals = stock_volumes[idx, i].min(axis=0).min()

            # Calculate factor by which the stock_conc must be multiplied by
            # for an addition of min_volume to work.
            factor = min_vals / min_volume

            # Calculate the updated stock concentrations and add to
            # new stock concentrations
            updated_stock_conc = stock_concs[i] * factor
            new_stock_concs = np.hstack((new_stock_concs, updated_stock_conc))

            # Now, calculate the volumes required for the new stock applied to
            # the unfeasible samples
            updated_stock_volume = stock_volumes[idx, i] / factor

            # update the volume array
            # 1. Remove additions of the previous stock
            new_stock_volumes[idx, i] = 0.0

            # 2. Add the new stock volumes to the updated stock_volume array
            new_stock_volumes = np.hstack(
                (new_stock_volumes, np.zeros((new_stock_volumes.shape[0], 1)))
            )

            new_stock_volumes[idx, -1] = updated_stock_volume

    # Check if the samples will overflow
    new_tot_vol = np.sum(new_stock_volumes, axis=1)
    solv_vol = targets_v - new_tot_vol

    check_volumes = solv_vol < 0.0
    if np.any(check_volumes):
        print("error")
        problem_idx = np.where(check_volumes)[0]
        print(f"Sample(s) {problem_idx} too high in volume.")
        print("Sample volume update failed.")
        print("The negative volumes in this array are the reason:")
        print(solv_vol)
        print(
            f"Consider increasing the concentrations of stock solutions other than {i}."
        )
        print()

    return new_stock_volumes, new_stock_concs, diluted_stocks
