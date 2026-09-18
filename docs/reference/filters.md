# Filters

Local estimators available to subsystems. Use the string names `"kf"`, `"ekf"`, `"ukf"`
and `"ckf"` in configurations. New filters subclass `Filter` and register with
`@register_filter("name")`; they are then available under that name everywhere.

::: pci.filters
    options:
      members: false
      show_root_heading: false

## Registry

::: pci.filters.base
    options:
      members:
        - Filter
        - FilterStep
        - register_filter
        - get_filter
        - available_filters

## Linear and extended Kalman filters

::: pci.filters.kalman

## Sigma-point filters

::: pci.filters.sigma_point
