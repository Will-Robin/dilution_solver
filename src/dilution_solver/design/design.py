import numpy as np
from copy import deepcopy
from sklearn import linear_model
import numpy.typing as npt
from typing import Annotated, Literal, TypeVar
from scipy.optimize import minimize

DType = TypeVar("DType", bound=np.generic)

Array1 = Annotated[npt.NDArray[DType], Literal[1]]
Array2 = Annotated[npt.NDArray[DType], Literal[2]]


class Design:
    """
    Model:
        - *roots* are samples provided by the user (stock solutions)
        - *sources* are samples which can be used to create other samples by
        transferring liquids from them. Roots are sources, but samples can also
        be derived from them which are sources themselves (e.g. in serial
        dilutions).
        - *targets* are samples which are created from sources.
    """

    def __init__(
        self,
        source_concentrations,
        source_volumes,
        root_labels,
        root_bounds,
        target_concentrations,
        target_volumes,
        target_labels,
    ):
        # (n_roots x n_compounds)
        self.root_concentrations = source_concentrations
        # (n_roots)
        self.root_volumes = source_volumes
        # (n_roots)
        self.root_labels = root_labels
        # (n_root, 2)
        self.root_bounds = root_bounds

        # (n_source, n_compound)
        self.source_concentrations = source_concentrations
        # (n_source)
        self.source_volumes = source_volumes
        # (n_sources)
        self.source_labels = root_labels

        # (n_targets, n_compounds)
        self.target_concentrations = target_concentrations
        # (n_targets)
        self.target_volumes = target_volumes
        # (n_targets)
        self.target_labels = target_labels

        # Regression model
        self.reg = linear_model.LinearRegression(fit_intercept=False)

    def volume_by_regression(self, sources, targets_conc, targets_vol):
        """
        Solving how much volume of a source solution is required to create a
        concentration of a target solution with a defined volume.

        Solution via linear regression allows for easier use of mixtures as
        sources for dilution.

        This method could fail if physically impossible values are given
        (i.e. the problem has no solution). Always validate the output.
        """
        # (n_compounds x n_sources) (n_sources x n_samples) = (n_compounds x n_samples)
        # Calculate moles required
        target_moles = targets_conc * targets_vol[:, np.newaxis]

        # Regression of concentration of sources onto moles gives volumes
        self.reg.fit(sources.T, target_moles.T)
        transfer_volumes = self.reg.coef_.copy()  # (n_sources x n_samples)

        return transfer_volumes

    def calculate_source_transfer_volumes(self):
        """
        Calculate volumes required to make target concentrations, given
        source concentrations, then calculate the difference between the
        source volumes required and the target volumes to give the solvent
        required to make the correct solution concentration.
        """
        transfer_volumes = self.volume_by_regression(
            self.source_concentrations, self.target_concentrations, self.target_volumes
        )
        solvent_volumes = self.target_volumes - np.sum(transfer_volumes, axis=1)
        return transfer_volumes, solvent_volumes

    def calculate_intermediate_transfer_volumes(self):
        """
        Calculate the volumes to be transferred from roots to all other sources.
        """

        # Regress roots onto sources to calculate volumes to transfer
        transfer_volumes = self.volume_by_regression(
            self.root_concentrations, self.source_concentrations, self.source_volumes
        )
        solvent_volumes = self.source_volumes - np.sum(transfer_volumes, axis=1)
        transfer_volumes = transfer_volumes[self.root_concentrations.shape[0] :]

        # Create output accounting roots being part of sources (they should be
        # removed by removing rows). Also, more sources have been added, so
        # new columns must be added to the output.
        root_dim = self.root_concentrations.shape[0]
        source_dim = self.source_concentrations.shape[0]
        add_dim_1 = transfer_volumes.shape[0]
        add_dim_2 = source_dim - root_dim

        transfer_volumes = np.hstack(
            (transfer_volumes, np.zeros((add_dim_1, add_dim_2)))
        )
        solvent_volumes = solvent_volumes[self.root_volumes.shape[0] :]

        return transfer_volumes, solvent_volumes

    def validate_or_optimize(self) -> None:
        """
        Test if an experiment design will be feasible. If not, attempt to optimise
        it so that it is feasible.
        """

        transfer_volumes, solvent_volumes = self.calculate_source_transfer_volumes()

        # Find if any of the solvent volumes are less than one - indicates
        # source solution needs to be updated.
        if np.any(solvent_volumes < 0):
            self.optimize_source_concentrations()

    def optimize_source_concentrations(self) -> None:
        """
        Choose optimal source concentrations based on target
        concentrations and volumes within bounds.
        """

        # create a copy to iterate over
        temp_design = deepcopy(self)

        # Optimize
        ## Initial guess for concentrations
        w0 = np.ones(temp_design.source_concentrations.shape[0])

        weight_bounds = []
        root_concs = self.get_root_concentrations()
        root_bounds = self.root_bounds
        for x in range(0, root_concs.shape[0]):
            lower = 0.0
            # max_conc = conc * max_w, max_w = max_conc / conc
            upper = root_bounds[x][1] / np.amax(root_concs[x])
            weight_bounds.append((lower, upper))

        result = minimize(
            evaluate_source_concentrations,
            w0,
            args=(temp_design),
            bounds=weight_bounds,
            method="SLSQP",
        )

        # Set optimized concentrations
        new_root_concs = weighting_update(self.get_root_concentrations(), result.x)
        self.set_root_concentrations(new_root_concs)

        transfer_volumes, solvent_volumes = self.calculate_source_transfer_volumes()

        # Check if the optimiser found a feasible result
        if np.any(solvent_volumes < 0.0) or np.any(transfer_volumes < 0.0):
            transfer_output = str(transfer_volumes).replace("\n", "\n\t")
            raise ValueError(
                f"Optimisation failed: negative volumes detected.\n"
                f"Solvent addition volumes:\n\t{solvent_volumes}\n"
                f"Target transfer volumes:\n\t{transfer_output}\n"
                f"Consider increasing the bounds for optimisation of root solutions."
            )

    def serial_dilution(
        self, possible_transfer_volumes, intermediate_volume=1.0
    ) -> None:
        """
        Finds which volumes in stock_volumes is lower than the minimum volume which
        can be transferred (min_volume), and creates new stock solutions which
        allow for sample transfer to go ahead.
        """

        # Now check for volumes which are too small to transfer
        min_volume = np.amin(possible_transfer_volumes)

        # TODO: Account for solvent transfer volumes which are too low
        transfer_volumes, solvent_volumes = self.calculate_source_transfer_volumes()

        idx_transfer = np.where(transfer_volumes < min_volume)

        num_intermediates = len([x for x in self.source_labels if "intermediate" in x])
        for sample_idx, stock_idx in zip(idx_transfer[0], idx_transfer[1]):
            # Get the lowest unfeasible stock volumes per stock
            min_val = transfer_volumes[sample_idx, stock_idx]

            # Calculate factor by which the stock_conc must be multiplied by
            # for an addition of min_volume to work.
            factor = min_val / min_volume

            # Calculate the updated stock concentrations and add to
            # new stock concentrations
            stock_concs = self.get_source_concentrations()
            intermediate_stock_conc = stock_concs[stock_idx] * factor
            self.add_source(
                intermediate_stock_conc,
                intermediate_volume,
                f"intermediate_{num_intermediates}",
            )
            num_intermediates += 1

        root_volumes, intermediate_solvent_volumes = (
            self.calculate_intermediate_transfer_volumes()
        )

        check_volumes = intermediate_solvent_volumes < 0.0
        if np.any(check_volumes):
            print("error")
            problem_idx = np.where(check_volumes)[0]
            print(f"Sample(s) {problem_idx} too high in volume.")
            print("Sample volume update failed.")
            print("The negative volumes in this array are the reason:")
            print(intermediate_solvent_volumes)
            print(
                f"Consider increasing the concentrations of stock solutions other than {stock_idx}."
            )
            print()

    """Setters"""

    def set_root_concentrations(self, new: Array1[np.float64]):
        self.root_concentrations = new
        self.source_concentrations[: new.shape[0], : new.shape[1]] = new

    def set_root_volumes(self, new):
        self.root_volumes = new
        self.source_volumes[: new.shape[0], : new.shape[1]] = new

    def set_root_labels(self, new):
        self.root_labels = new

    def set_root_bounds(self, new):
        self.root_bounds = new

    def set_source_concentrations(self, new):
        self.source_concentrations = new

    def set_source_volumes(self, new):
        self.source_volumes = new

    def set_source_labels(self, new):
        self.source_labels = new

    def set_target_concentrations(self, new):
        self.target_concentrations = new

    def set_target_volumes(self, new):
        self.target_volumes = new

    def set_solvent_volumes(self, new):
        self.solvent_volumes = new

    """Getters"""

    def get_root_concentrations(self):
        return self.root_concentrations.copy()

    def get_root_volumes(self):
        return self.root_volumes.copy()

    def get_root_labels(self):
        return self.root_labels[:]

    def get_root_bounds(self):
        return self.root_bounds.copy()

    def get_source_concentrations(self):
        return self.source_concentrations.copy()

    def get_source_volumes(self):
        return self.source_volumes.copy()

    def get_source_labels(self):
        return self.source_labels[:]

    def get_target_concentrations(self):
        return self.target_concentrations.copy()

    def get_target_volumes(self):
        return self.target_volumes.copy()

    def get_target_labels(self):
        return self.target_labels.copy()


    """Updaters"""

    def add_source(
        self, concentrations: Array2[np.float64], volume: np.float64, name: str
    ) -> None:
        """
        Add a new source to the design.
        """
        update_conc = np.vstack((self.get_source_concentrations(), concentrations))
        update_vol = np.hstack((self.get_source_volumes(), [volume]))
        update_name = self.get_source_labels()
        update_name.append(name)

        self.set_source_concentrations(update_conc)
        self.set_source_volumes(update_vol)
        self.set_source_labels(update_name)


def weighting_update(
    source_c: Array2[np.float64],
    weights: Array1[np.float64],
) -> Array2[np.float64]:
    ans = source_c * weights[:, np.newaxis]
    return ans


def evaluate_source_concentrations(
    weights: Array1[np.float64],
    design: Design,
) -> float:
    """
    Calculate the error of a proposed source concentration for an
    experimental design, costed according to the difference betweem the target
    volumes and the sum of calculated source volumes (per sample).

    Parameters
    ----------
    weights: Array1[np.float64]
    design: Design

    Returns
    -------
    objective: float
        Positive sum of the negative parts of the difference between targets_v and
        the sample-wise sum of source volumes.
    """

    # Calculate the excess_volume
    initial_source_conc = design.get_root_concentrations()
    source_concs = weighting_update(initial_source_conc, weights)
    design.set_root_concentrations(source_concs)
    _, solvent_volumes = design.calculate_source_transfer_volumes()

    # Reset to original source concentrations for next round
    # (not ideal...)
    design.set_root_concentrations(initial_source_conc)

    # Objective: minimize the amount of solvent required to make a dilution
    objective = -np.sum(np.minimum(solvent_volumes, 0))

    return objective
