import numpy as np
from pyDOE3 import fullfact
from pyDOE3 import bbdesign

# Define the levels for the full factorial design
levels = [3, 3]  # Factor A and Factor B each have 3 levels

# Generate the full factorial design
design = bbdesign(3)

# Define the concentration ranges for the factors
low = [10, 5]  # Minimum values for Factor A and Factor B
high = [50, 20]  # Maximum values for Factor A and Factor B

# Scale the design matrix to the concentration ranges
scaled_design = design * (np.array(high) - np.array(low)) + np.array(low)

# Print the scaled design matrix
print("Scaled Concentrations:")
print(scaled_design)
