from dataclasses import dataclass

import cyclonedds.idl.annotations as annotate
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, float32


@dataclass
@annotate.final
@annotate.autoid("sequential")
class WBCLowCmd(IdlStruct, typename="WBCLowCmd"):
    mobile: array[float32, 3]
    dq_des: array[float32, 14]
    left_gripper: float32
    right_gripper: float32


def WBCLowCmd_default():
    return WBCLowCmd([0.0] * 3, [0.0] * 14, 1.0, 1.0)  # type: ignore
