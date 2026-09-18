# Interfaces

Interfaces are the directed edges of the graph. An `Interface` carries a message from
`sender` to `receiver`, built by an interface law from the interface variables of both
ends. `spring_damper` returns the two interfaces (one per direction) of a linear
spring-damper between two subsystems; `custom_interface` wraps any callable or
expression string.

::: pci.interface
    options:
      members:
        - Interface
        - spring_damper
        - spring_damper_law
        - custom_interface
