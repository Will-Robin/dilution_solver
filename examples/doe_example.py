"""
Loading experiment concentration ranges from a file and creating a Box-Behnken
design from the ranges.
"""
import pandas as pd
from dilution_solver import doe


def main():
    # Specify input information
    exp_code = "VPR001"
    ranges = pd.read_csv("data/sample_ranges.csv")

    # Generate the design
    n_factors = ranges.compound_name.shape[0]
    low = ranges.lower_bound.to_numpy()
    high = ranges.upper_bound.to_numpy()
    scaled_design = doe.box_behnken_design(n_factors, low, high)

    # Create output
    df = pd.DataFrame(scaled_design, columns=[x for x in ranges.compound_name])
    sample_names = [f"{exp_code}_{i:03}" for i in range(scaled_design.shape[0])]
    df["sample_name"] = sample_names

    # Reorder columns
    cols = ["sample_name"] + [x for x in ranges.compound_name]
    df = df[cols]

    print(df.head())


if __name__ == "__main__":
    main()
