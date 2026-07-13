from dataclasses import dataclass

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, float32

import cyclonedds.idl.annotations as annotate


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Gr00tCmd(IdlStruct, typename="RightPoseCmd"):
    """
    [
        vx, vy, wz,          (3,)
        left_hand_pose       (9,)
        right_hand_pose      (9,)
    ]
    """
    
    data: array[float32, 21]

def Gr00tCmd_default():
    return Gr00tCmd([0.0] * 21) # type: ignore