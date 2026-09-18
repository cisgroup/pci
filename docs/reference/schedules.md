# Schedules

How and when subsystems exchange messages. Use the string names `"jacobi"`,
`"gauss_seidel"` and `"ab2"` in configurations, or pass a dict such as
`{"type": "jacobi", "iterations": 2}`.

::: pci.schedules
    options:
      members:
        - Schedule
        - Jacobi
        - GaussSeidel
        - AdamsBashforth2
        - get_schedule
