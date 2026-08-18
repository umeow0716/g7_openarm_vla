from dataclasses import dataclass

import cyclonedds.idl.annotations as annotate
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, float32
from g7_openarm_config import general_config

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
    return OpenArmCmd(([0.0] * 7 + [general_config.initial_gripper]) * 2)  # type: ignore
