from dataclasses import dataclass

import cyclonedds.idl.annotations as annotate
from cyclonedds.idl import IdlStruct
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Pose_
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Pose_


@dataclass
@annotate.final
@annotate.autoid("sequential")
class EETarget(IdlStruct, typename="EETarget"):
    left_target: Pose_
    right_target: Pose_


def EETarget_default():
    return EETarget(geometry_msgs_msg_dds__Pose_(), geometry_msgs_msg_dds__Pose_())  # type: ignore
