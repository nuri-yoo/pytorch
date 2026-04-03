"""
Dynamo implementations of CPython's PyObject_* default slot algorithms.

Analogous to CPython's Objects/object.c, this module holds the general
dispatch machinery that is independent of any specific type.
Per-type hook implementations (bool_impl, richcompare_impl, etc.)
live in their respective VT files.
"""

from typing import TYPE_CHECKING

from ..utils import istype
from .base import NO_SUCH_SUBOBJ, VariableTracker
from .constant import CONSTANT_VARIABLE_FALSE, CONSTANT_VARIABLE_TRUE


if TYPE_CHECKING:
    from ..symbolic_convert import InstructionTranslator


def vt_identity_compare(
    left: VariableTracker,
    right: VariableTracker,
) -> "VariableTracker | None":
    """Try to determine Python identity (left is right) at trace time.

    Returns ConstantVariable(True/False) if determinable, else None.
    Mirrors the logic in BuiltinVariable's handle_is handler.
    """
    if left is right:
        return CONSTANT_VARIABLE_TRUE

    left_val = left.get_real_python_backed_value()
    right_val = right.get_real_python_backed_value()
    left_known = left_val is not NO_SUCH_SUBOBJ
    right_known = right_val is not NO_SUCH_SUBOBJ

    if left_known and right_known:
        return (
            CONSTANT_VARIABLE_TRUE if left_val is right_val else CONSTANT_VARIABLE_FALSE
        )

    # One side has a concrete backing object, the other doesn't — they can't
    # be the same object.
    if left_known != right_known:
        return CONSTANT_VARIABLE_FALSE

    # Mutable containers created during tracing: VT identity = Python identity.
    from .dicts import ConstDictVariable
    from .lists import ListVariable

    if isinstance(left, (ConstDictVariable, ListVariable)):
        return CONSTANT_VARIABLE_FALSE

    # Different Python types can never be the same object.
    try:
        if left.python_type() is not right.python_type():
            return CONSTANT_VARIABLE_FALSE
    except NotImplementedError:
        pass

    # Different exception types are never identical.
    from .. import variables

    if (
        istype(left, variables.ExceptionVariable)
        and istype(right, variables.ExceptionVariable)
        and left.exc_type is not right.exc_type  # type: ignore[attr-defined]
    ):
        return CONSTANT_VARIABLE_FALSE

    return None


def generic_bool(tx: "InstructionTranslator", obj: VariableTracker) -> VariableTracker:
    """Mirrors PyObject_IsTrue.

    https://github.com/python/cpython/blob/c09ccd9c429/Objects/object.c#L2135-L2158

    Resolution order: constants → nb_bool → mp_length/sq_length → truthy.
    """
    from .constant import ConstantVariable

    if obj.is_python_constant():
        return ConstantVariable.create(bool(obj.as_python_constant()))

    result = obj.bool_impl(tx)
    if result is not None:
        return result

    result = _bool_from_length(tx, obj)
    if result is not None:
        return result

    return CONSTANT_VARIABLE_TRUE


def _bool_from_length(
    tx: "InstructionTranslator", obj: VariableTracker
) -> VariableTracker | None:
    """mp_length / sq_length fallback in PyObject_IsTrue."""
    from .constant import ConstantVariable
    from .dicts import ConstDictVariable
    from .lists import BaseListVariable

    if isinstance(obj, BaseListVariable):
        return ConstantVariable.create(len(obj.items) > 0)
    if isinstance(obj, ConstDictVariable):
        return ConstantVariable.create(len(obj.items) > 0)

    if obj.has_unpack_var_sequence(tx):
        return ConstantVariable.create(len(obj.unpack_var_sequence(tx)) > 0)

    return None
