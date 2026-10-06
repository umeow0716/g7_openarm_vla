from enum import StrEnum
from typing import Any, Self

class ControlMode(StrEnum):
    WBC = "wbc"
    ARM_ONLY = "arm-only"
    BASE_ONLY = "base-only"
    LEFT_ARM = "left-arm"
    RIGHT_ARM = "right-arm"
    LEFT_ARM_ONLY = "left-arm-only"
    RIGHT_ARM_ONLY = "right-arm-only"

    @classmethod
    def parse(cls, value: Any):
        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            normalized = value.strip().casefold()
            try:
                return cls(normalized)
            except ValueError:
                pass

        allowed = ", ".join(mode.value for mode in cls)
        raise ValueError(f"general.control_mode must be one of [{allowed}], got {value!r}")

class Joint(StrEnum):
    AMR_FL = "AMR_FL"
    AMR_FLW = "AMR_FLW"
    AMR_FR = "AMR_FR"
    AMR_FRW = "AMR_FRW"
    AMR_RL = "AMR_RL"
    AMR_RLW = "AMR_RLW"
    AMR_RR = "AMR_RR"
    AMR_RRW = "AMR_RRW"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"
    L6 = "L6"
    L7 = "L7"
    L8 = "L8"
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"
    R5 = "R5"
    R6 = "R6"
    R7 = "R7"
    R8 = "R8"
    
    @property
    def left_arm(cls):
        return (Joint.L1,)

    @classmethod
    def is_base(cls, joint: Self):
        return joint in (
            Joint.AMR_FL,
            Joint.AMR_FLW,
            Joint.AMR_FR,
            Joint.AMR_FRW,
            Joint.AMR_RL,
            Joint.AMR_RLW,
            Joint.AMR_RR,
            Joint.AMR_RRW
        )

    @classmethod
    def is_wheel(cls, joint: Self):
        return joint in (
            Joint.AMR_FLW,
            Joint.AMR_FRW,
            Joint.AMR_RLW,
            Joint.AMR_RRW
        )

    @classmethod
    def is_gripper(cls, joint: Self):
        return joint in (
            Joint.L8,
            Joint.R8,
        )

    @classmethod
    def name_to_joint(cls, name: str):
        s = name.replace("_", "")
        for j in Joint:
            if Joint.is_wheel(j) and "W" not in s:
                continue
            if not Joint.is_wheel(j) and "W" in s:
                continue
            if s.startswith(j.replace("_", "")):
                return j
        raise RuntimeError(f"Could not found joint `{name}`")

class Dim(StrEnum):
    X = "x"
    Y = "y"
    Z = "z"
    QUAT_W = "q_w"
    QUAT_X = "q_x"
    QUAT_Y = "q_y"
    QUAT_Z = "q_z"
    LIN_VEL_X = "lin_vel_x"
    LIN_VEL_Y = "lin_vel_y"
    LIN_VEL_Z = "lin_vel_z"
    ANG_VEL_X = "ang_vel_x"
    ANG_VEL_Y = "ang_vel_y" 
    ANG_VEL_Z = "ang_vel_z"

class KinBody(StrEnum):
    L_TCP = "L_tcp"
    R_TCP = "R_tcp"

    @classmethod
    def name_to_kin_body(cls, name: str):
        for body in KinBody:
            if body == name:
                return body
        
        raise RuntimeError(f"Could not found KinBody `{name}`")