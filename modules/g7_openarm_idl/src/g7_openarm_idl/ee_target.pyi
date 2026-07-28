from __future__ import annotations

from cyclonedds.idl import IdlStruct
from cyclonedds.internal import SampleInfo
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Pose_

class EETarget(IdlStruct):
    left_target: Pose_
    right_target: Pose_
    left_gripper: float
    right_gripper: float

    sample_info: SampleInfo

    def __init__(
        self,
        left_target: Pose_,
        right_target: Pose_,
        left_gripper: float,
        right_gripper: float,
    ) -> None: ...

def EETarget_default() -> EETarget: ...
