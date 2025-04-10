"""
Performing a sequential dilution when very small sample concentrations in
in comparison to the stock solution must be prepared.
"""

import pandas as pd
import numpy as np
from dilution_solver.routines import calculate_stock_volumes
from dilution_solver.routines import validate_or_optimize
from dilution_solver.routines import calculate_sample_concentrations


def check_volume_feasibility(
    stock_volumes, solvent_volumes, stock_concs, targets_c, targets_v, min_volume
):
    """ """
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


def sequential_dilution():
    """
    It is possible to solve for the dilution of any compound into any sample,
    given an arbitrary amount which can be transferred.
    However, in reality, the volume required to be transferred may be so small
    as to be impossible to measure, given the liquid transfer apparatus
    available. For instance, a 2 M solution must be diluted 1000 times to get a
    0.002 M solution. If the final solution volume is 100 μL, this requires
    pipetting a volume of 100 nL - too small for a 20 μL pipette.
    Therefore, an intermediate dilution is required so that realistic volumes
    can be prepared.

    1. Establish the minimum volume which can be transferred.
    2. Find which volumes transferred to samples are below this minimum
       transfer volume.
    3. Create a new, intermediate stock solution for these samples, created
       from an existing stock solution, which can be pipetted with a
       reasonable volume. Create a new stock solution to be prepared.
    4. Note that this new stock solution is 'special', and must be created as a
       sample from an existing stock, yet is not a sample which will be
       measured.
    """

    ## Stocks
    # Concentrations of four stock solutions, one for each component
    stock_df = pd.read_csv("data/stocks_high.csv")
    stock_c = stock_df.concentration.to_numpy()
    bounds = [(row.lower_bound, row.upper_bound) for _, row in stock_df.iterrows()]

    ## Targets
    target_df = pd.read_csv("data/targets.csv")
    # Target concentrations
    targets_c = target_df.drop(columns=["sample_name", "volume"]).to_numpy()
    # Target volumes
    targets_v = target_df.volume.to_numpy()

    # Lowest volume which can be transferred
    min_volume = 0.005

    stock_concs = validate_or_optimize(stock_c, targets_c, targets_v, bounds)

    stock_volumes, solvent_volumes = calculate_stock_volumes(
        stock_concs, targets_c, targets_v
    )

    new_stock_volumes, new_stock_concs, diluted_stocks = check_volume_feasibility(
        stock_volumes, solvent_volumes, stock_concs, targets_c, targets_v, min_volume
    )

def main():
    sequential_dilution()


if __name__ == "__main__":
    main()
