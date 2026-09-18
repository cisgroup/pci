"""Solve a problem defined entirely in a YAML file."""

import os

import matplotlib.pyplot as plt

import ci

here = os.path.dirname(os.path.abspath(__file__))
results = ci.solve(os.path.join(here, "01_four_dof_jacobi_ukf.yaml"), progress=True)
print(results.summary(start=1.0))
results.plot_parameters()
results.plot_states(["S1.x2", "S2.x3"])
plt.show()
