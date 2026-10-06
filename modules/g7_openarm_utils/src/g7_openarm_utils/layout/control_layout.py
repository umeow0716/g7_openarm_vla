from __future__ import annotations

from g7_openarm_config import general_config

from .enums import ControlMode, Joint


BASE_ENABLED = general_config.control_mode in (
    ControlMode.WBC,
    ControlMode.BASE_ONLY,
    ControlMode.LEFT_ARM,
    ControlMode.RIGHT_ARM,
)

LEFT_ARM_ENABLED = general_config.control_mode in (
    ControlMode.WBC,
    ControlMode.ARM_ONLY,
    ControlMode.LEFT_ARM,
    ControlMode.LEFT_ARM_ONLY,
)

RIGHT_ARM_ENABLED = general_config.control_mode in (
    ControlMode.WBC,
    ControlMode.ARM_ONLY,
    ControlMode.RIGHT_ARM,
    ControlMode.RIGHT_ARM_ONLY,
)

BASE_JOINT = (
    Joint.AMR_FL, Joint.AMR_FLW, Joint.AMR_FR, Joint.AMR_FRW,
    Joint.AMR_RL, Joint.AMR_RLW, Joint.AMR_RR, Joint.AMR_RRW,
)
ARM_JOINT = (
    Joint.L1, Joint.L2, Joint.L3, Joint.L4,
    Joint.L5, Joint.L6, Joint.L7,
    Joint.R1, Joint.R2, Joint.R3, Joint.R4,
    Joint.R5, Joint.R6, Joint.R7,
)
LEFT_ARM_JOINT = (
    Joint.L1, Joint.L2, Joint.L3, Joint.L4,
    Joint.L5, Joint.L6, Joint.L7,
)
RIGHT_ARM_JOINT = (
    Joint.R1, Joint.R2, Joint.R3, Joint.R4,
    Joint.R5, Joint.R6, Joint.R7
)


_joint_low_idx_dict: dict[Joint, int] = { k: v for v, k in enumerate(Joint) }
def low_idx(joint: Joint):
    return _joint_low_idx_dict[joint]