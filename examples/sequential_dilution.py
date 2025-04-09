"""
Performing a sequential dilution when very small sample concentrations in
in comparison to the stock solution must be prepared.
"""

import pandas as pd
import numpy as np
from dilution_solver.routines import calculate_stock_volumes
from dilution_solver.routines import validate_or_optimize


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
    min_volume = 0.001

    stock_concs = validate_or_optimize(stock_c, targets_c, targets_v, bounds)

    stock_volumes, solvent_volumes = calculate_stock_volumes(
        stock_concs, targets_c, targets_v
    )

    # First, find volumes which are lower than min_volume
    stock_volumes[1, 2] = 0.00001
    too_low = stock_volumes < min_volume
    if np.any(too_low):
        # Determine which stocks need diluting
        idx = np.where(too_low)
        sample_indices = idx[0]  # samples which need to be updated
        stock_indices = idx[1]  # stocks to be diluted

        # Set initial stock transfer volumes to zero
        stock_volumes[idx] = 0.0
        # Add a new column for a volume of a new stock with the minimum volume
        print(stock_volumes.shape)
        stock_volumes = np.hstack((stock_volumes, np.zeros((stock_volumes.shape[0], 1))))
        stock_volumes[sample_indices, -1] = min_volume

        # Now, calculate the stock concentration based on the minimum volume
        concentrations = targets_c[idx]
        # c = m/v
        moles = concentrations * min_volume
        print(concentrations)

        # Now, we need to back-calculate the number of moles required to be in
        # the updated volumes, based on the sample concentration
        # the volumes to be pipetted for the stocks are now fixed.
        # Therefore, calculate the moles from them


def main():
    sequential_dilution()


if __name__ == "__main__":
    main()
