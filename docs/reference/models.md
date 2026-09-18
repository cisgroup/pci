# Models

Ready-made physical models. `MassSpringChain` is a chain of masses connected by
springs (serial, or any explicit topology given as `Spring` objects) that can simulate
the monolithic truth, read sensors from it and decompose itself into a `System` from a
partition of its DOFs. `pci.models.paper` holds the paper's testbeds.

## Mass-spring chain

::: pci.models.mass_spring
    options:
      members:
        - MassSpringChain
        - Spring
        - Truth

## The paper's systems

::: pci.models.paper
