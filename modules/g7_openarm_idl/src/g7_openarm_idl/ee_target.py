from dataclasses import dataclass

from cyclonedds.idl import IdlStruct

import cyclonedds.idl.annotations as annotate

from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Pose_
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Pose_


@dataclass
@annotate.final
@annotate.autoid("sequential")
class EETarget(IdlStruct, typename="RightPoseCmd"):
    left_target: Pose_
    right_target: Pose_


def EETarget_default():
    return EETarget(
        geometry_msgs_msg_dds__Pose_(),
        geometry_msgs_msg_dds__Pose_()) # type: ignore