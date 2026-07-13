from dataclasses import dataclass

from cyclonedds.idl import IdlStruct

import cyclonedds.idl.annotations as annotate

from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point_, Quaternion_
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Point_, geometry_msgs_msg_dds__Quaternion_


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Odom(IdlStruct, typename="RightPoseCmd"):
    position: Point_
    quaternion: Quaternion_
    velocity: Point_
    angular_velocity: Point_
    vdot: Point_
    angular_vdot: Point_

def Odom_default():
    return Odom(
        geometry_msgs_msg_dds__Point_(),
        geometry_msgs_msg_dds__Quaternion_(),
        geometry_msgs_msg_dds__Point_(),
        geometry_msgs_msg_dds__Point_(),
        geometry_msgs_msg_dds__Point_(),
        geometry_msgs_msg_dds__Point_()) # type: ignore