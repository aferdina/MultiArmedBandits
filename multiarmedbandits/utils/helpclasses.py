""" module containing helper classes for mab
"""
from dataclasses import dataclass


@dataclass
class ArmAttributes:
    """class to store all attributes for select arm method"""

    step_in_game: int | None = None


__all__ = [ArmAttributes.__name__]
