from dataclasses import dataclass

import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types
from cyclonedds.idl import IdlStruct
from g7_openarm_config import general_config
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Pose_
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Pose_


@dataclass
@annotate.final
@annotate.autoid("sequential")
class EETarget(IdlStruct, typename="EETarget"):
    left_target: Pose_
    right_target: Pose_
    # Canonical normalized openness: 0.0=closed, 1.0=open.
    left_gripper: types.float64
    right_gripper: types.float64


def EETarget_default():
    return EETarget(geometry_msgs_msg_dds__Pose_(), geometry_msgs_msg_dds__Pose_(), general_config.initial_gripper, general_config.initial_gripper)  # type: ignore
