"""Subsystem models written as equation strings (requires ``sympy``).

Lets a subsystem be described in a configuration file instead of Python::

    states: [q, v]
    inputs: [f]
    parameters: {m: 1.0, c: 0.3}
    unknowns: {k: {initial: 5.0, std: 5.0}}
    equations:                     # d/dt of every state, in the order of ``states``
      q: v
      v: (f - k*q - c*v) / m
    measurements:                  # sensor name -> expression
      a: (f - k*q - c*v) / m

State, input and parameter names are plain symbols; ``t`` is time. The usual
functions (``sin``, ``cos``, ``exp``, ``sqrt``, ``abs``, ``tanh``, ``sign``,
``log``, ``pi``) are available. Interface laws use the prefixes ``s_`` and
``r_`` for the sender's and receiver's interface variables, e.g.
``"k*(s_q - r_q) + c*(s_v - r_v)"``.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

_FUNCS = ("sin", "cos", "tan", "exp", "sqrt", "tanh", "sinh", "cosh", "log", "sign", "Abs", "pi", "Max", "Min")


def _sympy():
    try:
        import sympy
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Equation strings require sympy: pip install pci-inference[symbolic]") from exc
    return sympy


def _namespace(sympy, names: Iterable[str]) -> Dict[str, object]:
    ns = {f: getattr(sympy, f) for f in _FUNCS}
    ns["abs"] = sympy.Abs
    ns["max"] = sympy.Max
    ns["min"] = sympy.Min
    for n in names:
        ns[n] = sympy.Symbol(n)
    return ns


def compile_model(
    expressions: Sequence[str],
    states: Sequence[str],
    inputs: Sequence[str],
    parameters: Sequence[str],
) -> Callable[[np.ndarray, np.ndarray, Mapping[str, float], float], np.ndarray]:
    """Compile expression strings into ``fn(x, u, p, t) -> array`` with the library's calling convention."""
    sympy = _sympy()
    states, inputs, parameters = list(states), list(inputs), list(parameters)
    clash = set(states) & set(inputs) | set(states) & set(parameters) | set(inputs) & set(parameters)
    if clash:
        raise ValueError(f"Names used in more than one role (state/input/parameter): {sorted(clash)}")
    names = states + inputs + parameters + ["t"]
    ns = _namespace(sympy, names)
    exprs = []
    for e in expressions:
        expr = sympy.sympify(str(e), locals=ns)
        unknown = {str(s) for s in expr.free_symbols} - set(names)
        if unknown:
            raise NameError(f"Unknown symbol(s) {sorted(unknown)} in expression '{e}'. Known: {names}")
        exprs.append(expr)
    args = [ns[n] for n in names]
    fn = sympy.lambdify(args, sympy.Matrix(exprs) if exprs else [], modules=["numpy"])
    n_out = len(exprs)
    np_idx = list(range(len(parameters)))

    def model(x, u, p, t):
        vals = [float(v) for v in np.asarray(x, dtype=float).ravel()] + \
               [float(v) for v in np.asarray(u, dtype=float).ravel()] + \
               [float(p[parameters[i]]) for i in np_idx] + [float(t)]
        out = fn(*vals)
        return np.asarray(out, dtype=float).reshape(n_out)

    model.expressions = [str(e) for e in expressions]  # type: ignore[attr-defined]
    return model


def compile_interface_law(
    expression: str,
    sender_vars: Sequence[str],
    receiver_vars: Sequence[str],
    constants: Optional[Mapping[str, float]] = None,
) -> Callable[[np.ndarray, np.ndarray, float], float]:
    """Compile an interface-law string using ``s_<name>``/``r_<name>`` for sender/receiver variables."""
    sympy = _sympy()
    constants = dict(constants or {})
    s_names = [f"s_{v}" for v in sender_vars]
    r_names = [f"r_{v}" for v in receiver_vars]
    names = s_names + r_names + list(constants) + ["t"]
    ns = _namespace(sympy, names)
    expr = sympy.sympify(str(expression), locals=ns)
    unknown = {str(s) for s in expr.free_symbols} - set(names)
    if unknown:
        raise NameError(f"Unknown symbol(s) {sorted(unknown)} in interface law '{expression}'. Known: {names}")
    fn = sympy.lambdify([ns[n] for n in names], expr, modules=["numpy"])
    const_vals = [float(constants[c]) for c in constants]

    def law(s, r, t):
        return float(fn(*[float(v) for v in s], *[float(v) for v in r], *const_vals, float(t)))

    return law
