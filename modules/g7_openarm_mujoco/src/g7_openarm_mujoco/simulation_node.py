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

from g7_openarm_config import general_config
from g7_openarm_idl import EETarget, EETarget_default

from .config import config
from .model_layout import MuJoCoModelLayout
from .initial_pose import hand_poses_for_arm_position
from .resources import model_directory
from .state_mapping import apply_lowcmd_to_mujoco, write_lowstate_from_mujoco
from .sensors import sensor_slice


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


class SimulationNode:
    def __init__(self, push_eetarget=True) -> None:
        self.model = _build_model()
        self.data = mujoco.MjData(self.model)
        self.layout = MuJoCoModelLayout(self.model)
        self.push_eetarget = push_eetarget

        if self.push_eetarget:
            mujoco.mj_forward(self.model, self.data)

            self.left_target_mocap_id = self.model.body_mocapid[self.model.body("left_target").id]
            self.right_target_mocap_id = self.model.body_mocapid[self.model.body("right_target").id]
            if general_config.lowlevel_initial_allowed:
                left_pose, right_pose = hand_poses_for_arm_position(
                    self.model,
                    general_config.initial_pos,
                )
                self.data.mocap_pos[self.left_target_mocap_id] = left_pose[:3]
                self.data.mocap_quat[self.left_target_mocap_id] = left_pose[3:]
                self.data.mocap_pos[self.right_target_mocap_id] = right_pose[:3]
                self.data.mocap_quat[self.right_target_mocap_id] = right_pose[3:]
            else:
                left_hand = self.data.body("L_tcp")
                right_hand = self.data.body("R_tcp")
                self.data.mocap_pos[self.left_target_mocap_id] = left_hand.xpos.copy()
                self.data.mocap_quat[self.left_target_mocap_id] = left_hand.xquat.copy()
                self.data.mocap_pos[self.right_target_mocap_id] = right_hand.xpos.copy()
                self.data.mocap_quat[self.right_target_mocap_id] = right_hand.xquat.copy()

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.quat_slice = sensor_slice(self.model, "imu_quat", expected_dim=4)
        self.gyro_slice = sensor_slice(self.model, "imu_gyro", expected_dim=3)
        self.acc_slice = sensor_slice(self.model, "imu_acc", expected_dim=3)

        # All callback-visible state is created before any ChannelSubscriber.Init()
        # or RecurrentThread.Start() call.
        self.lowstate = unitree_hg_msg_dds__LowState_()
        self.imustate = unitree_hg_msg_dds__IMUState_()
        self.eetarget = EETarget_default()
        self.eetarget.left_gripper = general_config.initial_gripper
        self.eetarget.right_gripper = general_config.initial_gripper
        self.lowcmd = unitree_hg_msg_dds__LowCmd_()

        self.lowstate_publisher = ChannelPublisher("rt/lowstate", LowState_)
        self.lowstate_publisher.Init()
        self.imustate_publisher = ChannelPublisher("rt/imustate", IMUState_)
        self.imustate_publisher.Init()
        
        if self.push_eetarget:
            self.eetarget_publisher = ChannelPublisher("rt/eetarget", EETarget)
            self.eetarget_publisher.Init()

        self.lowcmd_subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.lowcmd_subscriber.Init(self.lowcmd_handler, 0)

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
        self.viewer_thread = RecurrentThread(
            name="viewer_loop",
            interval=config.fps_interval,
            target=self.viewer_loop,
        )
        if self.push_eetarget:
            self.write_eetarget_thread = RecurrentThread(
                name="write_eetarget",
                interval=config.eetarget_interval,
                target=self.write_eetarget,
            )

        # Start only after every state object, channel and thread object exists.
        self.simulation_thread.Start()
        self.lowstate_thread.Start()
        self.imustate_thread.Start()
        if self.push_eetarget:
            self.write_eetarget_thread.Start()
        self.viewer_thread.Start()

    def lowcmd_handler(self, msg: LowCmd_) -> None:
        self.lowcmd = msg

    def write_lowstate(self) -> None:
        with self.viewer.lock():
            write_lowstate_from_mujoco(
                self.layout, self.data, self.lowstate
            )
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

    def write_eetarget(self) -> None:
        with self.viewer.lock():
            left_position = self.data.mocap_pos[self.left_target_mocap_id]
            left_quaternion = self.data.mocap_quat[self.left_target_mocap_id]
            right_position = self.data.mocap_pos[self.right_target_mocap_id]
            right_quaternion = self.data.mocap_quat[self.right_target_mocap_id]

            self.eetarget.left_target.position.x = left_position[0]
            self.eetarget.left_target.position.y = left_position[1]
            self.eetarget.left_target.position.z = (
                left_position[2] - self.data.qpos[self.layout.qpos_index("z")]
            )
            self.eetarget.left_target.orientation.w = left_quaternion[0]
            self.eetarget.left_target.orientation.x = left_quaternion[1]
            self.eetarget.left_target.orientation.y = left_quaternion[2]
            self.eetarget.left_target.orientation.z = left_quaternion[3]

            self.eetarget.right_target.position.x = right_position[0]
            self.eetarget.right_target.position.y = right_position[1]
            self.eetarget.right_target.position.z = (
                right_position[2] - self.data.qpos[self.layout.qpos_index("z")]
            )
            self.eetarget.right_target.orientation.w = right_quaternion[0]
            self.eetarget.right_target.orientation.x = right_quaternion[1]
            self.eetarget.right_target.orientation.y = right_quaternion[2]
            self.eetarget.right_target.orientation.z = right_quaternion[3]

        self.eetarget_publisher.Write(self.eetarget)

    def simulation_loop(self) -> None:
        with self.viewer.lock():
            apply_lowcmd_to_mujoco(
                self.layout, self.data, self.lowcmd, general_config
            )
            mujoco.mj_step(self.model, self.data)

    def viewer_loop(self) -> None:
        with self.viewer.lock():
            self.viewer.sync(state_only=True)


def main(push_eetarget=True) -> None:
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    _ = SimulationNode(push_eetarget)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
