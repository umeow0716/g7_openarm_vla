from __future__ import annotations

from cyclonedds.idl import IdlStruct
from cyclonedds.internal import SampleInfo

from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point_, Quaternion_


class Odom(IdlStruct):
    position: Point_
    quaternion: Quaternion_
    velocity: Point_
    angular_velocity: Point_
    vdot: Point_
    angular_vdot: Point_
    
    sample_info: SampleInfo

    def __init__(
        self,
        position: Point_,
        quaternion: Quaternion_,
        velocity: Point_,
        angular_velocity: Point_,
        vdot: Point_,
        angular_vdot: Point_,
    ) -> None:
        ...


def Odom_default() -> Odom:
    ...
