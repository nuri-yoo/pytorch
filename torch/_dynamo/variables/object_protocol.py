"""
Dynamo implementations of CPython's PyObject_* default slot algorithms.

Analogous to CPython's Objects/object.c, this module holds the general
comparison dispatch machinery that is independent of any specific type.
Per-type richcompare_impl hooks live in their respective VT files.
"""

from functools import lru_cache
from typing import TYPE_CHECKING

from torch._C._dynamo import (
    get_type_slots,
    has_slot,
    PyMappingSlots,
    PySequenceSlots,
    PyTypeSlots,
)

from .. import graph_break_hints, polyfills
from ..exc import raise_observed_exception, unimplemented
from ..utils import istype
from .base import NO_SUCH_SUBOBJ, raise_type_error_exc, VariableTracker
from .constant import CONSTANT_VARIABLE_FALSE, CONSTANT_VARIABLE_TRUE
from .functions import UserFunctionVariable


type_error = raise_type_error_exc


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


def debug_tp_slots(obj: VariableTracker) -> None:
    T = maybe_get_python_type(obj)
    seq_slots, map_slots, num_slots, type_slots = _get_cached_slots(T)
    print(f"Type {T} slots:")
    for slot, enum in (
        (seq_slots, PySequenceSlots),
        (map_slots, PyMappingSlots),
        (type_slots, PyTypeSlots),
    ):
        names: list[str] = []
        for slot_name, slot_bit in enum.__members__.items():  # type: ignore[missing-attributes]
            if has_slot(slot, slot_bit):
                names.append(slot_name)
        print(f"  {enum.__name__}: {', '.join(names)}")


@lru_cache(maxsize=256)
def _get_cached_slots(obj_type: type) -> tuple[int, int, int, int]:
    """Get all type slots for a type (cached)."""
    return get_type_slots(obj_type)


def type_implements_sq_length(obj_type: type) -> bool:
    """Check whether obj_type implements __len__ as sequence protocol"""
    seq_slots, _, _, _ = _get_cached_slots(obj_type)
    return has_slot(seq_slots, PySequenceSlots.SQ_LENGTH)


def type_implements_mp_length(obj_type: type) -> bool:
    """Check whether obj_type implements __len__ as mapping protocol"""
    _, map_slots, _, _ = _get_cached_slots(obj_type)
    return has_slot(map_slots, PyMappingSlots.MP_LENGTH)


def type_implements_tp_iter(obj_type: type) -> bool:
    _, _, _, type_slot = _get_cached_slots(obj_type)
    return has_slot(type_slot, PyTypeSlots.TP_ITER)


def type_sequence_check(obj_type: type) -> bool:
    """Implements PySequence_Check semantics for VariableTracker objects."""
    if issubclass(obj_type, dict):
        return False
    seq_slots, _, _, _ = _get_cached_slots(obj_type)
    return has_slot(seq_slots, PySequenceSlots.SQ_ITEM)


def maybe_get_python_type(obj: VariableTracker) -> type:
    try:
        return obj.python_type()
    except NotImplementedError:
        unimplemented(
            gb_type="Unsupported python_type() call",
            context=f"{obj} does not implement python_type()",
            explanation="This VariableTracker does not implement python_type(), "
            "which is required for object protocol operations.",
            hints=[
                *graph_break_hints.DYNAMO_BUG,
            ],
        )


def vt_mapping_size(
    tx: "InstructionTranslator", obj: "VariableTracker"
) -> "VariableTracker":
    # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/abstract.c#L2308-L2330
    T = maybe_get_python_type(obj)
    if type_implements_mp_length(T):
        return obj.mp_length(tx)

    if type_implements_sq_length(T):
        type_error(tx, f"{obj.python_type_name()} is not a mapping")

    type_error(tx, f"object of type {obj.python_type_name()} has no len()")


def generic_len(
    tx: "InstructionTranslator", obj: "VariableTracker"
) -> "VariableTracker":
    # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/abstract.c#L53-L69
    """
    Implements PyObject_Size/PyObject_Length semantics for VariableTracker objects.
    Dispatches to sq_length (sequences) or mp_length (mappings) depending on the VT type.
    """

    T = maybe_get_python_type(obj)
    if type_implements_sq_length(T):
        return obj.sq_length(tx)
    return vt_mapping_size(tx, obj)


def generic_getitem(
    tx: "InstructionTranslator", obj: "VariableTracker", item: "VariableTracker"
) -> "VariableTracker":
    """
    Implements PyObject_GetItem semantics for VariableTracker objects.
    Routes to obj.getitem_impl(tx, item)
    """
    return obj.call_method(tx, "__getitem__", [item], {})


def generic_getiter(
    tx: "InstructionTranslator", obj: "VariableTracker"
) -> "VariableTracker":
    """
    Implements PyObject_GetIter semantics for VariableTracker objects.
    Routes to obj.tp_iter(tx), the tp_iter slot on the object's type.
    """
    from .base import VariableTracker

    # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#2848
    # The algorithm for PyObject_GetIter is as follows: Steps:
    # 1. If the object has tp_iter slot, call it and return the result The
    #    return object must be an iterator (it must have a tp_iternext slot)
    # 2. If the object implements the sequence protocol - implements __getitem__
    #    and __len__, then create a sequence iterator for the object and return
    #    it.
    # 3. Otherwise, raise a TypeError

    T = maybe_get_python_type(obj)
    if type_implements_tp_iter(T):
        return obj.tp_iter(tx)
    elif type_sequence_check(T):
        return UserFunctionVariable(polyfills.builtins.sequence_iterator).call_function(
            tx, [obj], {}
        )
    else:
        msg = VariableTracker.build(
            tx, f"'{obj.python_type_name()}' object is not iterable"
        )
        raise_observed_exception(
            TypeError,
            tx,
            args=[msg],
        )
