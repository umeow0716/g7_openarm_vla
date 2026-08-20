from __future__ import annotations

import time

import mujoco
import mujoco.viewer
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.thread import RecurrentThread

from g7_openarm_config import general_config
from g7_openarm_idl import EETarget, EETarget_default, Odom

from .config import config
from .initial_pose import hand_poses_for_arm_position
from .model_layout import MuJoCoModelLayout
from .resources import model_directory
from .state_mapping import write_floating_base_qpos, write_lowstate_qpos


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


class RealVisualizationNode:
    def __init__(self, push_eetarget=True) -> None:
        self.push_eetarget = push_eetarget

        self.model = _build_model()
        self.data = mujoco.MjData(self.model)
        self.layout = MuJoCoModelLayout(self.model)

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

        # Callback-visible state is initialized before subscriber Init calls.
        # None is intentional: a zero quaternion is not a valid initial odometry state.
        self.lowstate: LowState_ | None = None
        self.odom: Odom | None = None
        self.eetarget = EETarget_default()
        self.target_ready = (
            general_config.lowlevel_initial_allowed or not general_config.arm_actuation_enabled
        )

        if self.push_eetarget:
            self.eetarget_publisher = ChannelPublisher("rt/eetarget", EETarget)
            self.eetarget_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.odom_subscriber = ChannelSubscriber("rt/odom", Odom)
        self.lowstate_subscriber.Init(self.lowstate_handler, 0)
        self.odom_subscriber.Init(self.odom_handler, 0)

        self.simulation_thread = RecurrentThread(
            name="simulation_loop",
            interval=config.interval,
            target=self.simulation_loop,
        )

        if self.push_eetarget:
            self.write_eetarget_thread = RecurrentThread(
                name="write_eetarget",
                interval=config.eetarget_interval,
                target=self.write_eetarget,
            )

        self.viewer_thread = RecurrentThread(
            name="viewer_loop",
            interval=config.fps_interval,
            target=self.viewer_loop,
        )

        self.simulation_thread.Start()
        if self.push_eetarget:
            self.write_eetarget_thread.Start()
        self.viewer_thread.Start()

    def lowstate_handler(self, msg: LowState_) -> None:
        self.lowstate = msg

    def odom_handler(self, msg: Odom) -> None:
        self.odom = msg

    def write_eetarget(self) -> None:
        with self.viewer.lock():
            if not self.target_ready:
                return

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
        lowstate = self.lowstate
        odom = self.odom
        if lowstate is None or odom is None:
            return

        with self.viewer.lock():
            write_floating_base_qpos(self.layout, self.data, odom)
            write_lowstate_qpos(self.layout, self.data, lowstate)
            self.data.qvel[:] = 0.0
            self.data.qacc[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

            if not self.target_ready:
                left_hand = self.data.body("L_tcp")
                right_hand = self.data.body("R_tcp")
                self.data.mocap_pos[self.left_target_mocap_id] = left_hand.xpos.copy()
                self.data.mocap_quat[self.left_target_mocap_id] = left_hand.xquat.copy()
                self.data.mocap_pos[self.right_target_mocap_id] = right_hand.xpos.copy()
                self.data.mocap_quat[self.right_target_mocap_id] = right_hand.xquat.copy()
                self.target_ready = True

    def viewer_loop(self) -> None:
        with self.viewer.lock():
            self.viewer.sync(state_only=True)


def main(push_eetarget=True) -> None:
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    _ = RealVisualizationNode(push_eetarget)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
