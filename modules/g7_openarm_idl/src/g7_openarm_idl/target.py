from dataclasses import dataclass

import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types
from cyclonedds.idl import IdlStruct
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Pose_, geometry_msgs_msg_dds__Point_
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Pose_, Point_

@dataclass
@annotate.final
@annotate.autoid("sequential")
class Target(IdlStruct, typename="Target"):
    left_target: Pose_
    right_target: Pose_
    # Canonical normalized openness: 0.0=closed, 1.0=open.
    left_gripper: types.float64
    right_gripper: types.float64
    look_at_point: Point_


def Target_default():
    return Target(geometry_msgs_msg_dds__Pose_(), geometry_msgs_msg_dds__Pose_(), 0.0, 0.0, geometry_msgs_msg_dds__Point_())  # type: ignore
