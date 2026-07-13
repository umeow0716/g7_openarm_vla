from __future__ import annotations

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, float32
from cyclonedds.internal import SampleInfo

from typing import Annotated, Sequence

class AMRCmd(IdlStruct):
    data: list[float]
    
    sample_info: SampleInfo
    
    def __init__(self, data: Annotated[Sequence[Annotated[float, 'float32']], array[float32, 3]]) -> None:
        ...

def AMRCmd_default() -> AMRCmd:
    ...