# Dilution Solver

Aims to figure out stock solution concentrations and dilution schemes for a
specified set of concentrations.

## Contents

- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [More Information](#more-information)

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

This code solves two problems:

1. 'I need to decide on a set of sample concentrations to prepare for my
  experiment'
2. 'I know which sample concentrations I would like to try, but I need to know
  which concentrations of stock solutions to prepare, and how much should be
  added to each sample'


### 1. I need to decide on a set of sample concentrations to prepare

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
used as input for the section below in selecting stock concentrations and sample
preparation volumes.

```python
import numpy as np
import pandas as pd
from dilution_solver import doe

# Specify input information
exp_code = "EXP001"

# Generate the design
n_factors = 3
factor_names = [f"factor_{n}" for n in range(n_factors)]
low = np.full(n_factors, 0.1)
high = np.full(n_factors, 1.0)
scaled_design = doe.box_behnken_design(low, high)

# Create output
df = pd.DataFrame(scaled_design, columns=[x for x in factor_names])
sample_names = [f"{exp_code}_{i:03}" for i in range(scaled_design.shape[0])]
df["sample_name"] = sample_names

# Reorder columns
cols = ["sample_name"] + [x for x in factor_names]
df = df[cols]

print(df.head())
```

### 2. I know which sample concentrations I would like to try, but I need to know which concentrations of stock solutions to prepare, and how much should be added to each sample

1. Create a csv file containing some candidate stock solutions and target concentrations and volumes.
  The file needs columns for sample names, volumes and a label column indicating if the sample is intended as a stock solution (`stock`) or if the sample is intended for measurement (`sample`).
  In addition to these columns, include one column for each compound to be used in the experiment.
  It makes things simpler if all quantities are given in unscaled SI units (M and L).
  Each sample is a row.
  Add the concentrations required for each sample where appropriate.
  Stock solutions can contain multiple compounds, but it is easier if they only contain one each.
  Give the file an informative name.

   Example:

   ```
   sample_name,compound_1,compound_2,compound_3,compound_4,volume,label
   stock_01,0.2,0.0,0.0,0.0,10.0,stock
   stock_02,0.0,0.4,0.0,0.0,10.0,stock
   stock_03,0.0,0.0,0.5,0.0,10.0,stock
   stock_04,0.0,0.0,0.0,0.4,10.0,stock
   test_01,0.1,0.4,0.6,0.3,0.3,sample
   test_02,0.5,0.3,0.7,1.2,1.2,sample
   test_03,0.6,0.8,0.3,0.1,0.1,sample
   ```

2. Below is an example script which takes in experimental designs and outputs
   files containing a suggested, feasible design. If the design does not work,
   warnings will be printed. See also: `example.py`.

3. The volumes suggested could be unfeasible (for instance, they may require the
   transfer of nL volumes, but the equipment you have available cannot transfer
   such small volumes). In this case, you can attempt to calculate a new liquid
   transfer scheme and set of stock solutions by creating diluted versions of
   existing stocks. The function `create_sequential_dilutions` attempts to do
   this automatically:

  ```python
  from dilution_solver.routines import create_sequential_dilutions
  # Lowest volume which can be transferred
  min_volume = 0.005

  # variables loaded in the snippet above.
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
  ```

  This step is optional. You could also attempt to lower the upper bounds of the
  stock concentrations to solve this. Always inspect the output and check that
  everything makes sense.

5. You can either go ahead with the design suggested by the software, or round
   up the suggested values it gives to make preparation more convenient and run
   the script again to get the volumes required for each sample. If the
   process fails to suggest a feasible design, consider increasing the upper
   bound of each stock concentration.

### Examples

The examples are written to be run from the repository's root directory. To run
one in the command line, type:

```shell
python examples/example.py
```

### More information

[Browse the code here.](https://will-robin.github.io/dilution_solver/dilution_solver.html).

