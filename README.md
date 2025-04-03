# Dilution Solver

Aims to figure out stock solution concentrations and dilution schemes for a
specified set of concentrations.

## Installation

Create a virtual environment in your preferred way, the install the following
packages:

- `pip install scipy`
- `pip install numpy`
- `pip install pandas`
- `pip install pyDOE3`

In the root directory of the repository, run the following command to make
dilution_solver available for import:

```
pip install -e .
```

## Usage

[Code documentation](https://will-robin.github.io/dilution_solver/dilution_solver.html).

### Beginning from a set of sample and stock concentrations

1. Create a csv file of target concentrations and volumes. Each line should be
   a sample, the concentration of each component is a column, and one column
   gives the sample volume, another gives the sample name. It makes things
   simpler if these quantities are given in unscaled SI units (M and L).
   Give the file an informative name.

   Example:

   ```
   sample_name,stock_1_concentration/ M,stock_2_concentration/ M,stock_3_concentration/ M,stock_4_concentration/ M,volume/ L
   test_01,0.1, 0.4, 0.6, 0.3,0.1
   test_02,0.5, 0.3, 0.7, 1.2,0.1
   test_03,0.6, 0.8, 0.3, 0.1,0.1
   ```

2. Create a separate csv file containing details of the stock solutions you wish
   to use. One column should contain the sample name. Provide initial guesses
   for the concentrations you will need in another column, and provide lower and
     upper bounds for the concentrations in two more columns. Again, to keep
     things simple, use base SI units. Give this file an informative name, and
     bear in mind that the aim is to calculate updates the concentrations of
     these stock solutions and output them in a separate file.


    Example:

    ```
    stock_name,concentration/ M,lower_bound/ M,upper_bound/ M
    stock_01,0.2,0.001,4.0
    stock_02,0.4,0.001,4.0
    stock_03,0.5,0.001,4.0
    stock_04,0.4,0.001,4.0
    ```

3. Below is an example script which takes in experimental designs and outputs
   files containing a suggested, feasible design. If the design does not work,
   warnings will be printed. See also: `example.py`.

  ```python
  import pandas as pd
  from dilution_solver.routines import calculate_stock_volumes
  from dilution_solver.routines import validate_or_optimize

  # Load concentrations of four stock solutions, one for each component
  stock_df = pd.read_csv("data/stocks.csv")
  stock_c = stock_df.concentration.to_numpy()
  bounds = [(row.lower_bound, row.upper_bound) for _, row in stock_df.iterrows()]

  ## Targets
  target_df = pd.read_csv("data/targets.csv")
  targets_c = target_df.drop(columns=["sample_name", "volume"]).to_numpy()
  targets_v = target_df.volume.to_numpy()

  stock_concs = validate_or_optimize(stock_c, targets_c, targets_v, bounds)

  stock_volumes, solvent_volumes = calculate_stock_volumes(
      stock_concs, targets_c, targets_v
  )

  # Create result output
  design = pd.DataFrame()
  design["stock_name"] = stock_df.stock_name
  design["concentration"] = stock_concs
  design["volume"] = stock_volumes

  # Create result output
  stock_design = pd.DataFrame()
  stock_design["stock_name"] = stock_df.stock_name
  stock_design["concentration"] = stock_concs
  stock_design["minimum_volume"] = stock_volumes.sum(axis=0)

  target_design = pd.DataFrame(stock_volumes,
      columns=[f"{stock}_volume" for stock in stock_df.stock_name]
  )
  target_design["sample_name"] = target_df.sample_name
  target_design["solvent"] = solvent_volumes

  # Reorder columns for output
  cols = target_design.columns.to_list()
  cols.remove("sample_name")
  cols  = ["sample_name"] + cols
  target_design = target_design[cols]
  stock_design.to_csv("data/stocks_design.csv", index=False)
  target_design.to_csv("data/sample_volumes.csv", index=False)
  ```

4. You can either go ahead with the design suggested by the software, or round
   up the suggested values it gives to make preparation more convenient and run
   the script again to get the volumes required for each sample. If the
   process fails to suggest a feasible design, consider increasing the upper
   bound of each stock concentration.

### Starting from desired concentration ranges

In this case, we can start with design of experiments (DoE) approaches in the
`dilution_solver.doe` module (see also [PyDOE3](https://github.com/relf/pyDOE3),
from which some of the strategies are derived).

You will have to inspect the DoE functions to see what arguments they accept. In
general, the main thing to supply are high and low limits for the concentrations
of each component. You may also have to supply the number of samples to make,
or the number of levels to create as required by the selected algorithm.

The following script creates a Box-Behnken design from some sample limits. Other
experimental designs are available `dilution_solver.doe` module. Note that you
can also store the inputs (experiment code, concentration ranges) externally in files
and load them in.

The below code will give a spreadsheet of sample concentrations which can be
used as input for the section above in selecting stock concentrations and sample
preparation volumes.

```python
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
```
