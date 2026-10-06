from g7_openarm_utils.layout import (
    Dim,
    Joint,
    IndexRegister,
    low_idx,
    openness_to_qpos,
    openness_rate_to_qvel,
)
from typing import TYPE_CHECKING

from .pinnzoo_binding import PinnZooModel
from .pinnzoo_func    import zero_state

if TYPE_CHECKING:
    from g7_openarm_idl import Odom
    from unitree_sdk2py.idl.default import HGLowState_

model = PinnZooModel()
q_index_register = IndexRegister()
v_index_register = IndexRegister()

for i, name in enumerate(model.config_names):
    if name in ["x", "y", "z", "q_w", "q_x", "q_y", "q_z"]:
        continue
    joint = Joint.name_to_joint(name)
    if q_index_register.is_registered(joint):
        continue
    q_index_register.register(joint, i)

for i, name in enumerate(model.vel_names):
    if name in  ["lin_v_x", "lin_v_y", "lin_v_z", "ang_v_x", "ang_v_y", "ang_v_z"]:
        continue
    joint = Joint.name_to_joint(name)
    if v_index_register.is_registered(joint):
        continue
    v_index_register.register(joint, i)

assert q_index_register.is_full()
assert v_index_register.is_full()

dim_idx_dict = {
    Dim.X: 0,
    Dim.Y: 1,
    Dim.Z: 2,
    Dim.QUAT_W: 3,
    Dim.QUAT_X: 4,
    Dim.QUAT_Y: 5,
    Dim.QUAT_Z: 6,
    Dim.LIN_VEL_X: model.nq,
    Dim.LIN_VEL_Y: model.nq + 1,
    Dim.LIN_VEL_Z: model.nq + 2,
    Dim.ANG_VEL_X: model.nq + 3,
    Dim.ANG_VEL_Y: model.nq + 4,
    Dim.ANG_VEL_Z: model.nq + 5,
}


def pinnzoo_q_idx(joint: Joint):
    return q_index_register.get(joint)

def pinnzoo_v_idx(joint: Joint):
    return model.nq + v_index_register.get(joint)

def pinnzoo_vdot_idx(joint: Joint):
    return v_index_register.get(joint)

def pinnzoo_dim_idx(dim: Dim):
    return dim_idx_dict[dim]

def build_lib_x(
    model: PinnZooModel,
    lowstate: "HGLowState_",
    odom: "Odom",
):
    x = zero_state(model)
    
    x[pinnzoo_dim_idx(Dim.X)] = odom.position.x
    x[pinnzoo_dim_idx(Dim.Y)] = odom.position.y
    x[pinnzoo_dim_idx(Dim.Z)] = odom.position.z
    x[pinnzoo_dim_idx(Dim.QUAT_W)] = odom.quaternion.w
    x[pinnzoo_dim_idx(Dim.QUAT_X)] = odom.quaternion.x
    x[pinnzoo_dim_idx(Dim.QUAT_Y)] = odom.quaternion.y
    x[pinnzoo_dim_idx(Dim.QUAT_Z)] = odom.quaternion.z
    x[pinnzoo_dim_idx(Dim.LIN_VEL_X)] = odom.velocity.x
    x[pinnzoo_dim_idx(Dim.LIN_VEL_Y)] = odom.velocity.y
    x[pinnzoo_dim_idx(Dim.LIN_VEL_Z)] = odom.velocity.z
    x[pinnzoo_dim_idx(Dim.ANG_VEL_X)] = odom.angular_velocity.x
    x[pinnzoo_dim_idx(Dim.ANG_VEL_Y)] = odom.angular_velocity.y
    x[pinnzoo_dim_idx(Dim.ANG_VEL_Z)] = odom.angular_velocity.z
    
    for joint in Joint:
        if Joint.is_gripper(joint):
            pqi = pinnzoo_q_idx(joint)
            pvi = pinnzoo_v_idx(joint)
            li  = low_idx(joint)
            qpos = openness_to_qpos(lowstate.motor_state[li].q)
            qvel = openness_rate_to_qvel(lowstate.motor_state[li].dq)
            x[pqi]   = qpos
            x[pqi+1] = qpos
            x[pvi]   = qvel
            x[pvi+1] = qvel
            continue
        x[pinnzoo_q_idx(joint)] = lowstate.motor_state[low_idx(joint)].q
        x[pinnzoo_v_idx(joint)] = lowstate.motor_state[low_idx(joint)].dq
    return x