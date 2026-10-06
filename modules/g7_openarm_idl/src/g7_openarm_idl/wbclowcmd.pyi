from __future__ import annotations

from cyclonedds.idl import IdlStruct
from cyclonedds.internal import SampleInfo
from cyclonedds.idl.types import array, float32

from typing import Annotated, Sequence

class WBCLowCmd(IdlStruct):
    mobile: list[float]
    dq_des: list[float]
    left_gripper:  float
    right_gripper: float

    sample_info: SampleInfo

    def __init__(
        self,
        mobile: Annotated[Sequence[Annotated[float, "float32"]], array[float32, 3]],
        dq_des: Annotated[Sequence[Annotated[float, "float32"]], array[float32, 3]],
        left_gripper:  Annotated[float, "float32"],
        right_gripper: Annotated[float, "float32"],
    ) -> None: ...

def WBCLowCmd_default() -> WBCLowCmd: ...
