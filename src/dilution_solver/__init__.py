"""
dilution_solver

Code which helps with the preparation of solution samples *en masse*.

1. If you don't have any sample concentrations prepared, but you know the
   concentration range for each compound that you would like to explore, then
   the Design of Experiments in the `doe` module can help with the creation of
   a set of sample concentrations.

2. If you have a collection of sample concentrations to prepare, and you need
   to decide on which stock solutions to prepare, and how much to use for each
   sample, use the code in the `routines` module.
"""

from .design import Design

__all__ = ["Design"]
