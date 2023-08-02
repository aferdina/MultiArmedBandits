""" contains type checking function for repository
"""
from typing import Any


def is_list_of_floats(lst: list[Any]) -> bool:
    """check if list contains only float values

    Args:
        lst (list[Any]): list to check

    Returns:
        bool: true if list contains only float values
    """
    if not isinstance(lst, list):
        return False
    for item in lst:
        if not isinstance(item, float):
            return False
    return True


def is_float_between_0_and_1(value: Any) -> bool:
    """check if float is between 0 and 1

    Args:
        value (Any): float to check

    Returns:
        bool: true if value is between 0 and 1
    """
    if isinstance(value, float) and (1 >= value >= 0):
        return True
    return False


def is_positive_integer(value: Any) -> bool:
    """check if input is a positive integer

    Args:
        value (Any): variable to check

    Returns:
        bool: true if input is a positive integer
    """
    if isinstance(value, int) and value > 0:
        return True
    return False


def is_positive_float(value: Any) -> bool:
    """check if input is a positive float

    Args:
        value (Any): value to check

    Returns:
        bool: true if input is a positive float
    """
    if isinstance(value, float) and value > 0:
        return True
    return False


def check_floats_between_zero_and_one(lst: Any) -> bool:
    """check if input is a list of floats between zero and one

    Args:
        lst (Any): input to check

    Returns:
        bool: true, if input is a list of floats between zero and one
    """
    if isinstance(lst, list):
        for item in lst:
            if isinstance(item, float) and 0 < item < 1:
                return True
        return False
    return False


def is_list_of_positive_floats(input_list: Any) -> bool:
    """check if input is a list of postive floats

    Args:
        input_list (Any): input to check

    Returns:
        bool: true, if input is a list of postive floats
    """
    # Check if the input is a list
    if not isinstance(input_list, list):
        return False

    # Check if all elements in the list are floats
    if not all(isinstance(item, float) for item in input_list):
        return False

    # Check if all elements in the list are positive
    if not all(item > 0.0 for item in input_list):
        return False

    return True


__all__ = [
    is_list_of_floats.__name__,
    is_float_between_0_and_1.__name__,
    is_positive_integer.__name__,
    check_floats_between_zero_and_one.__name__,
    is_list_of_positive_floats.__name__,
    is_positive_float.__name__,
]
