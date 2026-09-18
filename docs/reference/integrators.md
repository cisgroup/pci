# Integrators

Time integrators for the continuous-time subsystem dynamics. Use `"euler"`, `"heun"`
(default) or `"rk4"` in configurations, or pass your own step function
`step(rhs, x, t, dt) -> x_next`.

::: pci.integrators
