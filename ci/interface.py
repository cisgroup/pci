"""Interfaces: the directed edges of the interaction graph.

An :class:`Interface` carries a message from ``sender`` to ``receiver``. The
message is constructed by the *interface law*

    m = law(s_sender, s_receiver, t)

where ``s_sender`` are the sender's interface variables (e.g. boundary
displacement and velocity) and ``s_receiver`` the receiver's. The message is
added, multiplied by ``sign``, to the receiver's input port ``target``.

For now messages carry means only. Uncertainty-carrying messages will extend
the law with a variance term (see ``System.message_type``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

InterfaceLaw = Callable[[np.ndarray, np.ndarray, float], float]


@dataclass
class Interface:
    sender: str
    receiver: str
    sender_vars: Sequence[str]
    receiver_vars: Sequence[str]
    target: str
    law: InterfaceLaw
    sign: float = 1.0
    name: str = ""
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.sender_vars = list(self.sender_vars)
        self.receiver_vars = list(self.receiver_vars)
        if not self.name:
            self.name = f"{self.sender}->{self.receiver}"

    def value(self, s_sender: np.ndarray, s_receiver: np.ndarray, t: float) -> float:
        """Physical value of the message (before the sign convention is applied)."""
        return float(self.law(np.asarray(s_sender, float), np.asarray(s_receiver, float), t))


def spring_damper_law(k: float, c: float) -> InterfaceLaw:
    """Interface law ``F = k (x_a - x_b) + c (v_a - v_b)`` with ``s = [x, v]`` on each side."""

    def law(s_a, s_b, t):
        return k * (s_a[0] - s_b[0]) + c * (s_a[1] - s_b[1])

    return law


def spring_damper(
    a: str,
    a_vars: Sequence[str],
    a_port: str,
    b: str,
    b_vars: Sequence[str],
    b_port: str,
    k: float,
    c: float = 0.0,
    name: Optional[str] = None,
) -> List[Interface]:
    """Two directed interfaces for a spring-damper connecting subsystem ``a`` to ``b``.

    ``a_vars``/``b_vars`` are ``[displacement, velocity]`` names on each side.
    The interface force ``F = k (x_a - x_b) + c (v_a - v_b)`` acts as ``-F`` on
    ``a_port`` and ``+F`` on ``b_port``.
    """
    law = spring_damper_law(k, c)
    name = name or f"F[{a}.{a_vars[0]}-{b}.{b_vars[0]}]"
    params = {"k": k, "c": c}
    return [
        Interface(sender=b, receiver=a, sender_vars=b_vars, receiver_vars=a_vars, target=a_port,
                  law=lambda s_b, s_a, t: law(s_a, s_b, t), sign=-1.0, name=name, params=params),
        Interface(sender=a, receiver=b, sender_vars=a_vars, receiver_vars=b_vars, target=b_port,
                  law=law, sign=+1.0, name=name, params=params),
    ]


def custom_interface(
    sender: str,
    receiver: str,
    sender_vars: Sequence[str],
    receiver_vars: Sequence[str],
    target: str,
    law: InterfaceLaw,
    name: Optional[str] = None,
) -> Interface:
    """Convenience wrapper for a single directed interface with an arbitrary law."""
    return Interface(sender, receiver, sender_vars, receiver_vars, target, law, 1.0, name or "")
