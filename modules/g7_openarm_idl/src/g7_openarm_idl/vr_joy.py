from dataclasses import dataclass

import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types
from cyclonedds.idl import IdlStruct


@dataclass
@annotate.final
@annotate.autoid("sequential")
class VRJoy(IdlStruct, typename="VRJoy"):
    """Normalized VR joystick axes in the controller frame, each in [-1, 1]."""

    lx: types.float64
    ly: types.float64
    rx: types.float64
    ry: types.float64


def VRJoy_default():
    return VRJoy(0.0, 0.0, 0.0, 0.0)  # type: ignore
