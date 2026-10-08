from __future__ import annotations

import time
import mujoco
import mujoco.viewer
import numpy as np

from contextlib import contextmanager
from g7_openarm_config import general_config
from g7_openarm_idl import (
    EETarget,
    Odom,
    WBCLowCmd,
)
from g7_openarm_utils.layout.index_register import IndexRegister
from g7_openarm_utils.unitree import (
    ChannelFactoryInitialize,
    build_pub,
    build_sub,
    build_thread,
)
from g7_openarm_utils.layout import (
    Joint,
    BASE_ENABLED,
    RIGHT_ARM_ENABLED,
    LEFT_ARM_ENABLED,
    BASE_JOINT,
    LEFT_ARM_JOINT,
    RIGHT_ARM_JOINT,
    ARM_JOINT,
    low_idx,
)
from g7_openarm_utils.layout.gripper import (
    openness_to_qpos,
    qpos_to_openness,
    openness_rate_to_qvel,
    qvel_to_openness_rate,
)
from g7_openarm_idl import (
    pose_to_array,
    array_to_pose,
)
from importlib.resources import as_file, files
from typing import TYPE_CHECKING
from unitree_sdk2py.idl.default import (
    HGIMUState_,
    HGLowCmd_,
    HGLowState_,
    unitree_hg_msg_dds__IMUState_ as HGIMUState_default,
    unitree_hg_msg_dds__LowState_ as HGLowState_default,
)

from .config import config

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@contextmanager
def model_directory() -> "Generator[Path]":
    resource = files("g7_openarm_mujoco").joinpath("model")

    if not resource.is_dir():
        raise FileNotFoundError(f"Packaged MuJoCo model directory not found: {resource}")

    with as_file(resource) as path:
        yield path


class MujocoNode:
    def __init__(
        self,
        is_sim: bool,
        is_vr: bool,
        is_hommi: bool,
    ) -> None:
        self.is_sim   = is_sim
        self.is_vr    = is_vr
        self.is_hommi = is_hommi

        self.qpos_index_register = IndexRegister()
        self.dof_index_register  = IndexRegister()
        self.ctrl_index_register = IndexRegister()
        self.init_index_register()

        assert self.qpos_index_register.is_full()
        assert self.dof_index_register.is_full()
        assert self.ctrl_index_register.is_full()

        self.eetarget: EETarget | None = None
        self.imustate: HGIMUState_ | None = None
        self.lowcmd: HGLowCmd_ | None = None
        self.lowstate: HGLowState_ | None = None
        self.odom: Odom | None = None
        self.wbclowcmd: WBCLowCmd | None = None

        self.model  = self.build_model()
        self.data   = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.left_target_mocap_id  = self.model.body_mocapid[self.model.body("left_target").id]
        self.right_target_mocap_id = self.model.body_mocapid[self.model.body("right_target").id]

        self.acc_slice:  slice | None = None
        self.gyro_slice: slice | None = None
        self.quat_slice: slice | None = None
        self.init_imu()

        self.imustate = HGIMUState_default()
        self.lowstate = HGLowState_default()

        self.eetarget_pub  = build_pub("rt/eetarget", EETarget, self.is_sim and not self.is_vr)
        self.eetarget_sub  = build_sub("rt/eetarget", EETarget, self.eetarget_handler, self.is_vr or not self.is_sim)
        self.imustate_pub  = build_pub("rt/imustate", HGIMUState_, self.is_sim)
        self.lowstate_pub  = build_pub("rt/lowstate", HGLowState_, self.is_sim)
        self.lowstate_sub  = build_sub("rt/lowstate", HGLowState_, self.lowstate_handler, not self.is_sim)
        self.lowcmd_sub    = build_sub("rt/lowcmd", HGLowCmd_, self.lowcmd_handler, self.is_sim)
        self.odom_sub      = build_sub("rt/odom", Odom, self.odom_handler, not self.is_sim)
        self.wbclowcmd_sub = build_sub("rt/wbclowcmd", WBCLowCmd, self.wbclowcmd_handler)

        self.eetarget_thread = build_thread(config.hz, self.write_eetarget, self.is_sim and not self.is_vr)
        self.sim_thread      = build_thread(config.hz, self.sim_loop)
        self.lowstate_thread = build_thread(config.hz, self.write_lowstate, self.is_sim)
        self.imustate_thread = build_thread(config.imu_hz, self.write_imustate, self.is_sim)
        self.viewer_thread   = build_thread(config.fps, self.viewer_loop)

    def eetarget_handler(self, msg: EETarget):
        self.eetarget = msg

    def lowstate_handler(self, msg: HGLowState_):
        self.lowstate = msg

    def lowcmd_handler(self, msg: HGLowCmd_):
        self.lowcmd = msg

    def odom_handler(self, msg: Odom):
        self.odom = msg

    def wbclowcmd_handler(self, msg: WBCLowCmd):
        self.wbclowcmd = msg

    def write_eetarget(self):
        if self.eetarget_pub is None:
            return

        with self.viewer.lock():
            left_target = self.data.body("left_target")
            right_target = self.data.body("right_target")

        left_pose = np.concatenate((left_target.xpos, left_target.xquat), dtype=np.float64)
        right_pose = np.concatenate((right_target.xpos, right_target.xquat), dtype=np.float64)
        msg = EETarget(array_to_pose(left_pose), array_to_pose(right_pose), 0.0, 0.0)
        self.eetarget_pub.Write(msg)

    def write_lowstate(self):
        if self.lowstate_pub is None or self.lowstate is None:
            return

        with self.viewer.lock():
            qpos = self.data.qpos.copy()
            qvel = self.data.qvel.copy()
            qfrc_actuator = self.data.qfrc_actuator.copy()

        for joint in Joint:
            qposadr = self.qpos_index_register.get(joint)
            dofadr  = self.dof_index_register.get(joint)
            if Joint.is_gripper(joint):
                self.lowstate.motor_state[low_idx(joint)].q  = qpos_to_openness(qpos[qposadr])
                self.lowstate.motor_state[low_idx(joint)].dq = qvel_to_openness_rate(qvel[dofadr])
                self.lowstate.motor_state[low_idx(joint)].tau_est = 0.0
            else:
                self.lowstate.motor_state[low_idx(joint)].q  = qpos[qposadr]
                self.lowstate.motor_state[low_idx(joint)].dq = qvel[dofadr]
                self.lowstate.motor_state[low_idx(joint)].tau_est = qfrc_actuator[dofadr]
        self.lowstate_pub.Write(self.lowstate)

    def write_imustate(self):
        if self.imustate_pub is None or self.imustate is None:
            return

        with self.viewer.lock():
            sensordata = self.data.sensordata.copy()

        sensordata_list = sensordata.tolist()
        self.imustate.accelerometer = sensordata_list[self.acc_slice]
        self.imustate.gyroscope     = sensordata_list[self.gyro_slice]
        self.imustate.quaternion    = sensordata_list[self.quat_slice]
        self.imustate_pub.Write(self.imustate)

    def sim_loop(self):
        with self.viewer.lock():
            if self.eetarget is not None:
                left_pose  = pose_to_array(self.eetarget.left_target)
                right_pose = pose_to_array(self.eetarget.right_target)
                self.data.mocap_pos[self.left_target_mocap_id]   = left_pose[:3]
                self.data.mocap_quat[self.left_target_mocap_id]  = left_pose[3:]
                self.data.mocap_pos[self.right_target_mocap_id]  = right_pose[:3]
                self.data.mocap_quat[self.right_target_mocap_id] = right_pose[3:]

            if self.lowstate is not None:
                for joint in Joint:
                    if Joint.is_gripper(joint):
                        qpos_idx = self.qpos_index_register.get(joint)
                        qpos = openness_to_qpos(self.lowstate.motor_state[low_idx(joint)].q)
                        self.data.qpos[qpos_idx] = qpos
                        self.data.qpos[qpos_idx+1] = qpos
                    else:
                        self.data.qpos[self.qpos_index_register.get(joint)] = \
                            self.lowstate.motor_state[low_idx(joint)].q

            if self.odom is not None:
                self.data.qpos[0] = self.odom.position.x
                self.data.qpos[1] = self.odom.position.y
                self.data.qpos[2] = self.odom.position.z
                self.data.qpos[3] = self.odom.quaternion.w
                self.data.qpos[4] = self.odom.quaternion.x
                self.data.qpos[5] = self.odom.quaternion.y
                self.data.qpos[6] = self.odom.quaternion.z

            if self.lowcmd is not None:
                joint_list: list[Joint] = []
                if BASE_ENABLED:
                    joint_list += BASE_JOINT
                if LEFT_ARM_ENABLED:
                    joint_list += LEFT_ARM_JOINT
                    joint_list += [Joint.L8]
                if RIGHT_ARM_ENABLED:
                    joint_list += RIGHT_ARM_JOINT
                    joint_list += [Joint.R8]

                for joint in joint_list:
                    motor_cmd = self.lowcmd.motor_cmd[low_idx(joint)]
                    if Joint.is_gripper(joint):
                        qpos = openness_to_qpos(motor_cmd.q)
                        qvel = openness_rate_to_qvel(motor_cmd.dq)
                    else:
                        qpos = motor_cmd.q
                        qvel = motor_cmd.dq

                    q_err  = qpos - self.data.qpos[self.qpos_index_register.get(joint)]
                    dq_err = qvel - self.data.qvel[self.dof_index_register.get(joint)]
                    self.data.ctrl[self.ctrl_index_register.get(joint)] = \
                        q_err * motor_cmd.kp + \
                        dq_err * motor_cmd.kd + \
                        motor_cmd.tau
                    if Joint.is_gripper(joint):
                        self.data.ctrl[self.ctrl_index_register.get(joint)+1] = \
                            q_err  * motor_cmd.kp + \
                            dq_err * motor_cmd.kd + \
                            motor_cmd.tau

                print(
                    "[MUJOCO_ARM] "
                    "columns=joint,cmd_q,sim_q,cmd_dq,sim_dq,kp,kd,p_tau,d_tau,ff_tau,"
                    "ctrl,actuator_force,qfrc_actuator; rows="
                    + " | ".join(
                        f"{joint.name},"
                        f"{self.lowcmd.motor_cmd[low_idx(joint)].q:+.6f},"
                        f"{self.data.qpos[self.qpos_index_register.get(joint)]:+.6f},"
                        f"{self.lowcmd.motor_cmd[low_idx(joint)].dq:+.6f},"
                        f"{self.data.qvel[self.dof_index_register.get(joint)]:+.6f},"
                        f"{self.lowcmd.motor_cmd[low_idx(joint)].kp:+.6f},"
                        f"{self.lowcmd.motor_cmd[low_idx(joint)].kd:+.6f},"
                        f"{self.lowcmd.motor_cmd[low_idx(joint)].kp * (self.lowcmd.motor_cmd[low_idx(joint)].q - self.data.qpos[self.qpos_index_register.get(joint)]):+.6f},"
                        f"{self.lowcmd.motor_cmd[low_idx(joint)].kd * (self.lowcmd.motor_cmd[low_idx(joint)].dq - self.data.qvel[self.dof_index_register.get(joint)]):+.6f},"
                        f"{self.lowcmd.motor_cmd[low_idx(joint)].tau:+.6f},"
                        f"{self.data.ctrl[self.ctrl_index_register.get(joint)]:+.6f},"
                        f"{self.data.actuator_force[self.ctrl_index_register.get(joint)]:+.6f},"
                        f"{self.data.qfrc_actuator[self.dof_index_register.get(joint)]:+.6f}"
                        for joint in ARM_JOINT
                    ),
                    flush=True,
                )

            if True and self.wbclowcmd is not None:
                self.data.ctrl[:] = 0.0
                for i, joint in enumerate(ARM_JOINT):
                    self.data.qvel[self.dof_index_register.get(joint)] = self.wbclowcmd.dq_des[i]

            mujoco.mj_step(self.model, self.data)

    def viewer_loop(self):
        with self.viewer.lock():
            self.viewer.sync()

    def init_index_register(self):
        with model_directory() as model_dir:
            model = mujoco.MjModel.from_xml_path((model_dir / "scene.xml").as_posix())

        for i in range(model.njnt):
            model_joint = model.joint(i)
            name = model_joint.name
            if "floating_base" in name or "head" in name:
                continue
            joint = Joint.name_to_joint(name)
            qposadr = int(model.jnt_qposadr[model_joint.id])
            dofadr  = int(model.jnt_dofadr[model_joint.id])
            
            if self.qpos_index_register.is_registered(joint):
                continue
            self.qpos_index_register.register(joint, qposadr)
            self.dof_index_register.register(joint, dofadr)

        for i in range(model.nactuator):
            name  = model.actuator(i).name
            ctrl_adr = int(model.actuator_ctrladr[i])
            ctrl_num = int(model.actuator_ctrlnum[i])

            if ctrl_num != 1:
                raise RuntimeError(
                    f"Actuator {name} has {ctrl_num} controls; "
                    "current control layout only supports scalar actuators."
                )

            if not (0 <= ctrl_adr < model.nu):
                raise RuntimeError(
                    f"Invalid ctrl address {ctrl_adr} for actuator {name}"
                )

            if "head" in name:
                continue

            joint = Joint.name_to_joint(name)
            if self.ctrl_index_register.is_registered(joint):
                continue
            self.ctrl_index_register.register(joint, ctrl_adr)

    def init_imu(self):
        for i in range(self.model.nsensor):
            sensor = self.model.sensor(i)
            if sensor.type == mujoco.mjtSensor.mjSENS_ACCELEROMETER:
                if self.acc_slice is not None:
                    raise RuntimeError("multiple accelerometer found.")
                self.acc_slice = slice(sensor.adr[0], sensor.adr[0] + sensor.dim[0])
            elif sensor.type == mujoco.mjtSensor.mjSENS_GYRO:
                if self.gyro_slice is not None:
                    raise RuntimeError("multiple gyro found.")
                self.gyro_slice = slice(sensor.adr[0], sensor.adr[0] + sensor.dim[0])
            elif sensor.type == mujoco.mjtSensor.mjSENS_FRAMEQUAT:
                if self.quat_slice is not None:
                    raise RuntimeError("multiple quat found.")
                self.quat_slice = slice(sensor.adr[0], sensor.adr[0] + sensor.dim[0])

        if self.acc_slice is None:
            raise RuntimeError("accelerometer not found.")
        if self.gyro_slice is None:
            raise RuntimeError("gyro not found.")
        if self.quat_slice is None:
            raise RuntimeError("quat not found.")

    def load_hand_default_pose(self) -> EETarget:
        """Load the default left/right TCP poses from a MuJoCo XML model."""
        with model_directory() as model_dir:
            model_path = (model_dir / "scene.xml").as_posix()
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        for joint_config, qpos in general_config.initial_pos.items():
            if self.is_vr:
                continue
            joint = Joint.name_to_joint(joint_config)
            if not RIGHT_ARM_ENABLED and (joint in RIGHT_ARM_JOINT or joint == Joint.R8):
                continue
            if not LEFT_ARM_ENABLED and (joint in LEFT_ARM_JOINT or joint == Joint.L8):
                continue
            data.qpos[self.qpos_index_register.get(joint)] = qpos
                
        mujoco.mj_forward(model, data)

        left_hand  = data.body("L_tcp")
        right_hand = data.body("R_tcp")

        left_pose  = np.concatenate((left_hand.xpos, left_hand.xquat), dtype=np.float64)
        right_pose = np.concatenate((right_hand.xpos, right_hand.xquat), dtype=np.float64)
        return EETarget(array_to_pose(left_pose), array_to_pose(right_pose), 0.0, 0.0)

    def build_model(self) -> mujoco.MjModel:
        with model_directory() as model_dir:
            model_path = (model_dir / "scene.xml").as_posix()
        default_pose =  self.load_hand_default_pose()
        
        spec = mujoco.MjSpec.from_file(model_path)
        spec.option.timestep = 1.0 / config.hz

        left_arr = pose_to_array(default_pose.left_target)
        left_target = spec.worldbody.add_body(
            name="left_target",
            mocap=True,
            pos=left_arr[:3],
            quat=left_arr[3:],
        )
        left_target.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05],
            rgba=[1, 0, 0, 0.3],
            contype=0,
            conaffinity=0,
        )

        right_arr = pose_to_array(default_pose.right_target)
        right_target = spec.worldbody.add_body(
            name="right_target",
            mocap=True,
            pos=right_arr[:3],
            quat=right_arr[3:],
        )
        right_target.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05],
            rgba=[0, 0, 1, 0.3],
            contype=0,
            conaffinity=0,
        )

        return spec.compile()

def main(is_sim=True, is_vr=False, is_hommi=False):
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    node = MujocoNode(is_sim, is_vr, is_hommi)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main(is_sim=True, is_vr=False, is_hommi=False)
