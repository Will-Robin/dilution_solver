"""
Performing a sequential dilution when very small sample concentrations in
in comparison to the stock solution must be prepared.
"""

import pandas as pd
import numpy as np
from dilution_solver.routines import calculate_stock_volumes
from dilution_solver.routines import validate_or_optimize
from dilution_solver.routines import calculate_sample_concentrations
from dilution_solver.routines import create_sequential_dilutions


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

    new_stock_volumes, new_stock_concs, diluted_stocks = create_sequential_dilutions(
        stock_volumes,
        solvent_volumes,
        stock_concs,
        targets_c,
        targets_v,
        min_volume,
    )

    calculate_sample_concentrations(new_stock_concs, new_stock_volumes, targets_v)


def main():
    sequential_dilution()


if __name__ == "__main__":
    main()
