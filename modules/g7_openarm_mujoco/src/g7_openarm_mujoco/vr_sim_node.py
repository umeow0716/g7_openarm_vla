from __future__ import annotations

import time

import mujoco
import mujoco.viewer
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__IMUState_,
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_
from unitree_sdk2py.utils.thread import RecurrentThread

from g7_openarm_idl import EETarget, EETarget_default
from g7_openarm_utils.idl import pose_to_array

from .config import config
from .resources import model_directory
from .sensors import scalar_sensor_address, sensor_slice


def _build_model() -> mujoco.MjModel:
    with model_directory() as model_dir:
        spec = mujoco.MjSpec.from_file((model_dir / "scene.xml").as_posix())
        spec.option.timestep = config.interval

        left_target = spec.worldbody.add_body(
            name="left_target",
            mocap=True,
            pos=[0.0, 0.0, 0.0],
            quat=[1.0, 0.0, 0.0, 0.0],
        )
        left_target.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05],
            rgba=[1, 0, 0, 0.3],
            contype=0,
            conaffinity=0,
        )

        right_target = spec.worldbody.add_body(
            name="right_target",
            mocap=True,
            pos=[0.0, 0.0, 0.0],
            quat=[1.0, 0.0, 0.0, 0.0],
        )
        right_target.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05],
            rgba=[0, 0, 1, 0.3],
            contype=0,
            conaffinity=0,
        )

        return spec.compile()


class VRSimulationNode:
    def __init__(self) -> None:
        self.model = _build_model()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        left_hand = self.data.body("L_gripper_tcp_link")
        right_hand = self.data.body("R_gripper_tcp_link")
        self.left_target_mocap_id = self.model.body_mocapid[self.model.body("left_target").id]
        self.right_target_mocap_id = self.model.body_mocapid[self.model.body("right_target").id]
        self.data.mocap_pos[self.left_target_mocap_id] = left_hand.xpos.copy()
        self.data.mocap_quat[self.left_target_mocap_id] = left_hand.xquat.copy()
        self.data.mocap_pos[self.right_target_mocap_id] = right_hand.xpos.copy()
        self.data.mocap_quat[self.right_target_mocap_id] = right_hand.xquat.copy()

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.motor_names = [
            "AMR_FL",
            "AMR_FLW",
            "AMR_FR",
            "AMR_FRW",
            "AMR_RL",
            "AMR_RLW",
            "AMR_RR",
            "AMR_RRW",
            "L_1",
            "L_2",
            "L_3",
            "L_4",
            "L_5",
            "L_6",
            "L_7",
            "gripper_L",
            "R_1",
            "R_2",
            "R_3",
            "R_4",
            "R_5",
            "R_6",
            "R_7",
            "gripper_R",
        ]
        self.pos_addresses = [
            scalar_sensor_address(self.model, f"{name}_pos") for name in self.motor_names
        ]
        self.vel_addresses = [
            scalar_sensor_address(self.model, f"{name}_vel") for name in self.motor_names
        ]
        self.torque_addresses = [
            scalar_sensor_address(self.model, f"{name}_torque") for name in self.motor_names
        ]
        self.secondary_gripper_pos_addresses = {
            15: scalar_sensor_address(self.model, "gripper_LR_pos"),
            23: scalar_sensor_address(self.model, "gripper_RR_pos"),
        }
        self.secondary_gripper_vel_addresses = {
            15: scalar_sensor_address(self.model, "gripper_LR_vel"),
            23: scalar_sensor_address(self.model, "gripper_RR_vel"),
        }
        self.quat_slice = sensor_slice(self.model, "imu_quat", expected_dim=4)
        self.gyro_slice = sensor_slice(self.model, "imu_gyro", expected_dim=3)
        self.acc_slice = sensor_slice(self.model, "imu_acc", expected_dim=3)

        self.lowstate = unitree_hg_msg_dds__LowState_()
        self.imustate = unitree_hg_msg_dds__IMUState_()
        self.eetarget = EETarget_default()
        self.lowcmd = unitree_hg_msg_dds__LowCmd_()

        self.lowstate_publisher = ChannelPublisher("rt/lowstate", LowState_)
        self.lowstate_publisher.Init()
        self.imustate_publisher = ChannelPublisher("rt/imustate", IMUState_)
        self.imustate_publisher.Init()
        self.lowcmd_subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.lowcmd_subscriber.Init(self.lowcmd_handler, 0)
        self.eetarget_subscriber = ChannelSubscriber("rt/eetarget", EETarget)
        self.eetarget_subscriber.Init(self.eetarget_handler, 0)

        self.simulation_thread = RecurrentThread(
            name="simulation_loop",
            interval=config.interval,
            target=self.simulation_loop,
        )
        self.lowstate_thread = RecurrentThread(
            name="write_lowstate",
            interval=config.interval,
            target=self.write_lowstate,
        )
        self.imustate_thread = RecurrentThread(
            name="write_imustate",
            interval=config.imu_interval,
            target=self.write_imustate,
        )
        self.update_target_thread = RecurrentThread(
            name="update_target",
            interval=config.eetarget_interval,
            target=self.update_target,
        )
        self.viewer_thread = RecurrentThread(
            name="viewer_loop",
            interval=config.fps_interval,
            target=self.viewer_loop,
        )

        self.simulation_thread.Start()
        self.lowstate_thread.Start()
        self.imustate_thread.Start()
        self.update_target_thread.Start()
        self.viewer_thread.Start()

    def lowcmd_handler(self, msg: LowCmd_) -> None:
        self.lowcmd = msg

    def write_lowstate(self) -> None:
        with self.viewer.lock():
            for index in range(len(self.motor_names)):
                self.lowstate.motor_state[index].q = self.data.sensordata[self.pos_addresses[index]]
                self.lowstate.motor_state[index].dq = self.data.sensordata[
                    self.vel_addresses[index]
                ]
                self.lowstate.motor_state[index].tau_est = self.data.sensordata[
                    self.torque_addresses[index]
                ]

            self.lowstate.imu_state = self.imustate

        self.lowstate_publisher.Write(self.lowstate)

    def write_imustate(self) -> None:
        with self.viewer.lock():
            quaternion = self.data.sensordata[self.quat_slice].copy()
            gyroscope = self.data.sensordata[self.gyro_slice].copy()
            accelerometer = self.data.sensordata[self.acc_slice].copy()

            for index in range(4):
                self.imustate.quaternion[index] = quaternion[index]
            for index in range(3):
                self.imustate.gyroscope[index] = gyroscope[index]
                self.imustate.accelerometer[index] = accelerometer[index]

        self.imustate_publisher.Write(self.imustate)

    def eetarget_handler(self, msg: EETarget) -> None:
        self.eetarget = msg

    def update_target(self) -> None:
        left_pose = pose_to_array(self.eetarget.left_target)
        right_pose = pose_to_array(self.eetarget.right_target)

        with self.viewer.lock():
            self.data.mocap_pos[self.left_target_mocap_id] = left_pose[:3]
            self.data.mocap_quat[self.left_target_mocap_id] = left_pose[3:]
            self.data.mocap_pos[self.right_target_mocap_id] = right_pose[:3]
            self.data.mocap_quat[self.right_target_mocap_id] = right_pose[3:]

    def simulation_loop(self) -> None:
        with self.viewer.lock():
            for index in range(len(self.motor_names)):
                position_address = self.pos_addresses[index]
                velocity_address = self.vel_addresses[index]
                motor_command = self.lowcmd.motor_cmd[index]

                q_error = motor_command.q - self.data.sensordata[position_address]
                dq_error = motor_command.dq - self.data.sensordata[velocity_address]

                actuator_index = index if index < 16 else index + 1
                self.data.ctrl[actuator_index] = (
                    q_error * motor_command.kp + dq_error * motor_command.kd + motor_command.tau
                )

                secondary_position = self.secondary_gripper_pos_addresses.get(index)
                if secondary_position is not None:
                    secondary_velocity = self.secondary_gripper_vel_addresses[index]
                    q_error = motor_command.q - self.data.sensordata[secondary_position]
                    dq_error = motor_command.dq - self.data.sensordata[secondary_velocity]
                    self.data.ctrl[actuator_index + 1] = (
                        q_error * motor_command.kp + dq_error * motor_command.kd + motor_command.tau
                    )

            mujoco.mj_step(self.model, self.data)

    def viewer_loop(self) -> None:
        with self.viewer.lock():
            self.viewer.sync(state_only=True)


def main() -> None:
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    _ = VRSimulationNode()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
