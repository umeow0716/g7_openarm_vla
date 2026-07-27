from dataclasses import dataclass

import cyclonedds.idl.annotations as annotate
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, float32


@dataclass
@annotate.final
@annotate.autoid("sequential")
class AMRCmd(IdlStruct, typename="AMRCmd"):
    """
    [ vx, vy, ω ]
    """

    data: array[float32, 3]


def AMRCmd_default():
    return AMRCmd([0.0] * 3)  # type: ignore
