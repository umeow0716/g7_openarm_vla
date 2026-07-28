from __future__ import annotations

from cyclonedds.idl import IdlStruct
from cyclonedds.internal import SampleInfo

class VRJoy(IdlStruct):
    lx: float
    ly: float
    rx: float
    ry: float

    sample_info: SampleInfo

    def __init__(self, lx: float, ly: float, rx: float, ry: float) -> None: ...

def VRJoy_default() -> VRJoy: ...
