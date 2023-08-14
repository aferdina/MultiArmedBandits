import functools
from enum import Enum, EnumMeta
from typing import Any

import yaml

# CONTSTANT TAGS
YAML_TAG_APPLY_OBJECT = "tag:yaml.org,2002:python/object/apply"
YAML_TAG_OBJECT = "tag:yaml.org,2002:python/object"
SEQUENCETAG = "tag:yaml.org,2002:seq"


def sequence_constructor(loader: yaml.Loader, node: yaml.nodes.Node) -> Any:
    """yaml constructor for sequence

    Args:
        loader (yaml.Loader): Loader object for sequence
        node (yaml.nodes.Node): Node to read from yaml file

    Returns:
        Any: python obejct created from sequence method
    """
    return loader.construct_sequence(node)


def general_constructor(load: yaml.Loader, node: yaml.nodes.Node, constructed_class: Any) -> Any:
    """general constructor function for any class in `ConfigEnumClasses`

    Args:
        load (yaml.Loader): Loader module to load data in
        node (yaml.nodes.Node): Node module to get data from
        constructed_class (ConfigEnumClasses): class to construct

    Returns:
        Any: Instance of `ConfigEnumClasses`object
    """
    fields = load.construct_mapping(node)
    fields_without_underscore = {}
    for key, value in fields.items():
        new_key = key.lstrip("_")  # Remove leading underscores from the key
        fields_without_underscore[new_key] = value
    used_class = constructed_class.__new__(constructed_class)
    # print(f"fieldsunderscore are{fields_without_underscore}")
    used_class.__init__(**fields_without_underscore)
    if hasattr(used_class, "__post_init__"):
        used_class.__post_init__()
    return used_class


def enum_constructor(loader: yaml.Loader, node: yaml.nodes.Node, constructed_class: Enum) -> Any:
    """general enum constructor function for all Enum classes in Supply Chain Config

    Args:
        loader (yaml.Loader): Loader module to load data in
        node (yaml.nodes.Node): Node module to get data from
        constructed_class (Enum): class to construct

    Returns:
        Any: Instance of `ConfigEnumClasses`object
    """
    value = loader.construct_scalar(node)
    return constructed_class(value)


def add_constructors(adding_classes: list[object]) -> None:
    """add constructors for a list of classes

    Args:
        adding_classes (list[object]): List of classes to add as constructors
    """

    for adding_class in adding_classes:
        if isinstance(adding_class, EnumMeta):
            yaml.add_constructor(
                f"{YAML_TAG_APPLY_OBJECT}:{adding_class.__module__}.{adding_class.__name__}",
                functools.partial(enum_constructor, constructed_class=adding_class),
            )
        else:
            yaml.add_constructor(
                f"{YAML_TAG_OBJECT}:{adding_class.__module__}.{adding_class.__name__}",
                functools.partial(general_constructor, constructed_class=adding_class),
            )


__all__ = [add_constructors.__name__, sequence_constructor.__name__]
