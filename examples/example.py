import numpy as np
import pandas as pd
from dilution_solver import Design


def create_design_output(design, col_names):
    optimized_source_c = design.get_source_concentrations()
    target_concentrations = design.get_target_concentrations()
    conc_design = np.vstack((optimized_source_c, target_concentrations))
    output_design = pd.DataFrame(conc_design, columns=col_names)

    # Create label columns
    sample_labels = design.get_source_labels()
    sample_labels.extend(design.get_target_labels())

    # Create assignment columns
    tags = ["stock" for x in design.get_root_labels()]
    tags.extend(
        [
            "intermediate"
            for c, _ in enumerate(design.get_source_labels())
            if c + 1 > len(tags)
        ]
    )
    tags.extend(["target" for _ in design.get_target_labels()])

    output_design["label"] = tags
    output_design["sample_name"] = sample_labels
    return output_design


def create_volume_output(design):
    """ """
    source_volumes, solvent_volumes = design.calculate_source_transfer_volumes()
    intermediate_source_volumes, intermediate_solvent_volumes = (
        design.calculate_intermediate_transfer_volumes()
    )

    root_vol_out = np.zeros(
        (
            design.root_concentrations.shape[0],
            design.source_concentrations.shape[0],
        )
    )

    source_vol_out = np.vstack(
        (root_vol_out, intermediate_source_volumes, source_volumes)
    )
    solvent_vol_out = np.hstack(
        (design.root_volumes, intermediate_solvent_volumes, solvent_volumes)
    )
    total_vols = source_vol_out.sum(axis=1) + solvent_vol_out

    # Create label columns
    sample_labels = design.get_source_labels()
    sample_labels.extend(design.get_target_labels())

    # Create assignment columns
    tags = ["stock" for x in design.get_root_labels()]
    tags.extend(
        [
            "intermediate"
            for c, _ in enumerate(design.get_source_labels())
            if c + 1 > len(tags)
        ]
    )
    tags.extend(["target" for _ in design.get_target_labels()])

    source_vol_df = pd.DataFrame(source_vol_out, columns=design.get_source_labels())
    source_vol_df["solvent"] = solvent_vol_out

    source_vol_df["volume"] = total_vols

    source_vol_df["sample_name"] = sample_labels
    source_vol_df["label"] = tags
    return source_vol_df


def main():
    # Specify input information
    exp_code = "EXP001"
    df = pd.read_csv("data/experiment_design.csv")

    conc_df = df.drop(columns=["sample_name", "label", "volume"])

    # Figure out how much each of the samples assigned source must be added to
    # the target samples to create the desired concentrations
    stock_df = conc_df[df.label == "stock"]
    stock_concentrations = stock_df.to_numpy()
    stock_voumes = stock_df.volume.to_numpy()
    stock_labels = stock_df.sample_name.to_list()

    target_df = conc_df[df.label == "sample"]
    target_concentrations = target_df.to_numpy()
    target_volumes = target_df.volume.to_numpy()
    target_labels = target_df.sample_name.to_list()

    # Create lower and upper bounds for the source concentrations
    ## (Could also be stored in a separate file)
    source_bounds = [(0.001, 4.0) for _ in range(stock_concentrations.shape[1])]

    # Create a list of possible transfer volumes
    ## This should be a list of all available volumes which can be transferred
    ## by the apparatus available for the experiment.
    possible_transfer_volumes = [0.005]

    design = Design(
        stock_concentrations,
        stock_voumes,
        stock_labels,
        source_bounds,
        target_concentrations,
        target_volumes,
        target_labels,
    )

    # Optimise the concentration of stock solutions so that the solution
    # transfer is optimised
    design.validate_or_optimize()

    # Check if the liquid transfers are feasible (no volumes to be transferred
    # are lower than the lowest amount possible to transfer)
    transfer_volumes, solvent_volumes = design.calculate_source_transfer_volumes()
    min_volume = np.amin(possible_transfer_volumes)
    idx_transfer = np.where(transfer_volumes < min_volume)[0]
    if len(idx_transfer) > 0:
        design.serial_dilution(possible_transfer_volumes)

    # Create DataFrame outputs
    ## Concentrations to be prepared (including stocks)
    output_design = create_design_output(design, conc_df.columns)
    output_design.to_csv(f"data/{exp_code}_conc_design_updated.csv", index=False)

    ## Volumes to be transferred (including stocks)
    source_vol_df = create_volume_output(design)
    source_vol_df.to_csv(f"data/{exp_code}_transfer_volumes.csv", index=False)

    print(output_design)
    print()
    print(source_vol_df)


if __name__ == "__main__":
    main()
