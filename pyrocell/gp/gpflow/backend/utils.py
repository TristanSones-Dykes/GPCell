# Standard Library Imports
from typing import Mapping

# Third-Party Library Imports
from tensorflow import Module

# Direct Namespace Imports
from gpflow import Parameter

# Internal Project Imports
from pyrocell.gp.gpflow.backend.types import GPPrior


def _multiple_assign(module: Module, parameters: Mapping[str, GPPrior]) -> None:
    """
    Assigns parameters in a dictionary to the Module (tf.Module or gpflow.Module)

    Parameters
    ----------
    module : Module
        Model to assign parameters to
    parameters : Mapping[str, GPPrior]
        Dictionary of parameters to assign
    """
    for key, value in parameters.items():
        _set_parameter_by_key(module, key, value)


def _set_parameter_by_key(module: Module, key: str, value: GPPrior):
    """
    Sets a parameter in a module by key

    Parameters
    ----------
    module : Module
        Module to set the parameter in
    key : str
        Key to the parameter
    value : GPPrior
        Value to set the parameter to
    """
    parts = key.split(".")
    target = module

    for i in range(len(parts) - 1):
        part = parts[i]
        if "[" in part and "]" in part:
            # Handle indexed attributes like "kernels[0]"
            attr_name, index = part.split("[")
            index = int(index.rstrip("]"))
            target = getattr(target, attr_name)[index]
        else:
            # Handle normal attributes
            target = getattr(target, part)

    # Finally, set the parameter
    match value:
        case Parameter():
            setattr(target, parts[-1], value)
        case _:
            getattr(target, parts[-1]).assign(value)
