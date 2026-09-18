# Loads

External forces per input port. Loads can be combined with `+` and `*`; a plain array
of length `n_steps` is accepted wherever a load is expected. In configurations a load is
a dict with a `type` key (see [Configuration](../configuration.md#loads)).

::: pci.loads
