from __future__ import annotations

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, float32
from cyclonedds.internal import SampleInfo

from typing import Annotated, Sequence

class Gr00tCmd(IdlStruct):
    data: list[float]
    
    sample_info: SampleInfo
    
    def __init__(self, data: Annotated[Sequence[Annotated[float, 'float32']], array[float32, 16]]) -> None:
        ...

def Gr00tCmd_default() -> Gr00tCmd:
    ...
