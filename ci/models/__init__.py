"""Ready-made physical models."""

from . import paper
from .mass_spring import GROUND, MassSpringChain, Spring, Truth

__all__ = ["MassSpringChain", "Spring", "Truth", "GROUND", "paper"]
