from dataclasses import dataclass

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, float32

import cyclonedds.idl.annotations as annotate


@dataclass
@annotate.final
@annotate.autoid("sequential")
class OpenArmCmd(IdlStruct, typename="OpenArmCmd"):
    """
    [
        L_1, L_2, L_3, L_4, L_5, L_6, L_7, L_gripper,
        R_1, R_2, R_3, R_4, R_5, R_6, R_7, R_gripper,
    ]
    """
    
    data: array[float32, 16]

def OpenArmCmd_default():
    return OpenArmCmd([0.0] * 16) # type: ignore
