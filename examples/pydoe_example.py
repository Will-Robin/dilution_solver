import pandas as pd
import numpy as np
import pyDOE3
import matplotlib.pyplot as plt
from typing import Annotated, Literal, TypeVar
import numpy.typing as npt

DType = TypeVar("DType", bound=np.generic)
Array1 = Annotated[npt.NDArray[DType], Literal[1]]


def bbdesign(n_factors: int, low: Array1[np.float64], high: Array1[np.float64]):
    # Generate the design
    design = pyDOE3.bbdesign(n_factors)
    # center design around 0.5
    design = (design - design.min()) / (design - design.min()).max()

    # Scale the design to the bounds
    mean_pos = high - low
    scaled_design = (design * mean_pos) + low

    return scaled_design


def ccdesign(n_factors, center):
    ccdesign(n, center, alpha, face)


def main():
    # Specify input information
    exp_code = "VPR001"
    ranges = pd.read_csv("data/sample_ranges.csv")

    # Generate the design
    n_factors = ranges.compound_name.shape[0]
    low = ranges.lower_bound.to_numpy()
    high = ranges.upper_bound.to_numpy()
    scaled_design = bbdesign(n_factors, low, high)

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
