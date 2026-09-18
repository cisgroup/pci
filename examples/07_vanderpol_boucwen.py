# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 7. Non-standard subsystems: Van der Pol meets Bouc-Wen
#
# Nothing in the framework assumes masses and springs. `07_vanderpol_boucwen.yaml`
# couples a **Van der Pol** oscillator with an unknown nonlinearity `mu` (UKF, RK4) to
# a **Bouc-Wen** hysteretic element with unknown `beta` and `gamma` (CKF) through a
# spring-damper interface, with Gauss-Seidel messages. Both subsystems are written as
# equation strings (`model: custom`, needs `sympy`); the ground truth is the same
# coupled model simulated with the true parameters.

# %%
import pci

print(open("07_vanderpol_boucwen.yaml").read())

# %% [markdown]
# ## Solve

# %%
results = pci.solve("07_vanderpol_boucwen.yaml")
print(results.summary(start=15.0))

# %%
results.plot_parameters();

# %%
results.plot_states(["VdP.q", "BW.q", "BW.z"]);

# %%
results.plot_messages();

# %% [markdown]
# The hysteretic variable `z` of the Bouc-Wen element is not measured; it is
# reconstructed from the displacement and acceleration sensors together with the
# parameters. `Abs` in the equations is the sympy absolute value, so the CKF sigma
# points pass through the non-smooth term without a Jacobian.
