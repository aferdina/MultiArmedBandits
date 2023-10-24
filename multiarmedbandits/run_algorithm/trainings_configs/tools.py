import inspect
import sys
from enum import Enum

from multiarmedbandits.algorithms import *


class InputTypes(Enum):
    """ "
    Enum specficying all Input-Types for Run-Configs
    """

    INT = "int"
    FLOAT = "float"
    INT_LIST = "INT_list"
    FLOAT_LIST = "float_list"

# TODO: bugfix for boltzmann-rv
def get_args(cls, algo: bool = True):
    """
    function to get arguments of algos and environments
    """
    if algo:
        args = inspect.getfullargspec(getattr(sys.modules[__name__], cls)).annotations
    else:
        args = inspect.getfullargspec(cls).annotations
    if "return" in args:
        args.pop("return")
    if "bandit_env" in args:
        args.pop("bandit_env")

    return_args = dict()

    for key, value in args.items():
        if value == int:
            return_args[key] = InputTypes.INT

        elif value == float:
            return_args[key] = InputTypes.FLOAT

        elif type(value) == type(Enum):
            return_args[key] = [i.value for i in value]

        elif value == list[float]:
            return_args[key] = InputTypes.FLOAT_LIST

        elif value == list[int]:
            return_args[key] = InputTypes.INT_LIST

        else:
            return_args[key] = get_args(cls=value, algo=False)

    return return_args


def get_empty_assignments(args: dict, n_arms):
    """
    funtion to get empty assignments dict
    """
    return_assignments = dict()
    for key, value in args.items():
        
        if value == InputTypes.INT:
            return_assignments[key] = None
        
        elif value == InputTypes.FLOAT:
            return_assignments[key] = None

        elif value == InputTypes.INT_LIST:
            return_assignments[key] = n_arms*[None]

        elif value == InputTypes.FLOAT_LIST:
            return_assignments[key] = n_arms*[None]

        elif type(value) == dict:
            return_assignments[key] = get_empty_assignments(value, n_arms)

        else: #enum-case
            return_assignments[key] = None
    return return_assignments
        