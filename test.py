import numpy as np
import pandas as pd
from dilution_solver import doe

# Specify input information
exp_code = "VPR001"

# Generate the design
n_factors = 3
factor_names = [f"factor_{n}" for n in range(n_factors)]
low = np.full(n_factors, 0.1)
high = np.full(n_factors, 1.0)
scaled_design = doe.box_behnken_design(n_factors, low, high)

# Create output
df = pd.DataFrame(scaled_design, columns=[x for x in factor_names])
sample_names = [f"{exp_code}_{i:03}" for i in range(scaled_design.shape[0])]
df["sample_name"] = sample_names

# Reorder columns
cols = ["sample_name"] + [x for x in factor_names]
df = df[cols]

print(df.head())
