import time
import mujoco
import mujoco.viewer

from pathlib import Path

from g7_openarm_idl import EETarget, EETarget_default
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default  import unitree_hg_msg_dds__LowState_, unitree_hg_msg_dds__LowCmd_

from .config import config


DEFAULT_MODEL_XML_PATH = Path(__file__).resolve().parent.parent.parent.parent.parent / "model" / "scene.xml"


class SimulationNode:
    def __init__(self):
        self.spec = mujoco.MjSpec.from_file(DEFAULT_MODEL_XML_PATH.as_posix())
        self.spec.option.timestep = config.interval
        print(config.interval)
        left_target = self.spec.worldbody.add_body(
            name='left_target',
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

        right_target = self.spec.worldbody.add_body(
            name='right_target',
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

        self.model = self.spec.compile()
        self.data  = mujoco.MjData(self.model)
        
        mujoco.mj_forward(self.model, self.data)

        left_hand_pos = self.data.body("L_gripper_tcp_link").xpos.copy()
        right_hand_pos = self.data.body("R_gripper_tcp_link").xpos.copy()
        left_hand_quat = self.data.body("L_gripper_tcp_link").xquat.copy()
        right_hand_quat = self.data.body("R_gripper_tcp_link").xquat.copy()

        self.left_target_mocap_id = self.model.body_mocapid[self.model.body('left_target').id]
        self.right_target_mocap_id = self.model.body_mocapid[self.model.body('right_target').id]

        self.data.mocap_pos[self.left_target_mocap_id] = left_hand_pos
        self.data.mocap_quat[self.left_target_mocap_id] = left_hand_quat
        self.data.mocap_pos[self.right_target_mocap_id] = right_hand_pos
        self.data.mocap_quat[self.right_target_mocap_id] = right_hand_quat

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
        self.pos_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name + "_pos")
            for sensor_name in self.motor_names
        ]
        self.vel_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name + "_vel")
            for sensor_name in self.motor_names
        ]
        self.torque_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name + "_torque")
            for sensor_name in self.motor_names
        ]
        self.quat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
        self.gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        self.acc_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acc")
        
        self.msg = unitree_hg_msg_dds__LowState_()
        self.lowcmd = unitree_hg_msg_dds__LowCmd_()
        
        self.simulation_thread = RecurrentThread(
            name="simulation_loop",
            interval=config.interval,
            target=self.simulation_loop,
        )
        self.simulation_thread.Start()
        
        self.viewer_thread = RecurrentThread(
            name="viewer_loop",
            interval=config.fps_interval,
            target=self.viewer_loop,
        )
        self.viewer_thread.Start()
        
        self.lowstate_publisher = ChannelPublisher("rt/lowstate", LowState_)
        self.lowstate_publisher.Init()
        self.write_lowstate_thread = RecurrentThread(
            name="write_lowstate",
            interval=config.interval,
            target=self.write_lowstate,
        )
        self.write_lowstate_thread.Start()
        
        self.eetarget = EETarget_default()
        self.eetarget_publisher = ChannelPublisher("rt/eetarget", EETarget)
        self.eetarget_publisher.Init()
        self.write_eetarget_thread = RecurrentThread(
            name="write_eetarget",
            interval=config.interval,
            target=self.write_eetarget,
        )
        self.write_eetarget_thread.Start()
        
        self.lowcmd_subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.lowcmd_subscriber.Init(self.lowcmd_handler, 10)
    
    def lowcmd_handler(self, msg: LowCmd_):
        self.lowcmd = msg

    def write_lowstate(self):
        with self.viewer.lock():
            for i, _ in enumerate(self.motor_names):
                pos_id = self.pos_ids[i]
                vel_id = self.vel_ids[i]
                torque_id = self.torque_ids[i]
                
                self.msg.motor_state[i].q  = self.data.sensordata[pos_id]
                self.msg.motor_state[i].dq = self.data.sensordata[vel_id]
                self.msg.motor_state[i].tau_est = self.data.sensordata[torque_id]
            self.msg.imu_state.quaternion[0]    = self.data.sensordata[self.quat_id]
            self.msg.imu_state.quaternion[1]    = self.data.sensordata[self.quat_id+1]
            self.msg.imu_state.quaternion[2]    = self.data.sensordata[self.quat_id+2]
            self.msg.imu_state.quaternion[3]    = self.data.sensordata[self.quat_id+3]
            self.msg.imu_state.gyroscope[0]     = self.data.sensordata[self.gyro_id]
            self.msg.imu_state.gyroscope[1]     = self.data.sensordata[self.gyro_id+1]
            self.msg.imu_state.gyroscope[2]     = self.data.sensordata[self.gyro_id+2]
            self.msg.imu_state.accelerometer[0] = self.data.sensordata[self.acc_id]
            self.msg.imu_state.accelerometer[1] = self.data.sensordata[self.acc_id+1]
            self.msg.imu_state.accelerometer[2] = self.data.sensordata[self.acc_id+2]
        self.lowstate_publisher.Write(self.msg)

    def write_eetarget(self):
        with self.viewer.lock():
            self.eetarget.left_target.position.x = self.data.mocap_pos[self.left_target_mocap_id][0]
            self.eetarget.left_target.position.y = self.data.mocap_pos[self.left_target_mocap_id][1]
            self.eetarget.left_target.position.z = self.data.mocap_pos[self.left_target_mocap_id][2] - self.data.qpos[2]
            self.eetarget.left_target.orientation.w = self.data.mocap_quat[self.left_target_mocap_id][0]
            self.eetarget.left_target.orientation.x = self.data.mocap_quat[self.left_target_mocap_id][1]
            self.eetarget.left_target.orientation.y = self.data.mocap_quat[self.left_target_mocap_id][2]
            self.eetarget.left_target.orientation.z = self.data.mocap_quat[self.left_target_mocap_id][3]
            self.eetarget.right_target.position.x = self.data.mocap_pos[self.right_target_mocap_id][0]
            self.eetarget.right_target.position.y = self.data.mocap_pos[self.right_target_mocap_id][1]
            self.eetarget.right_target.position.z = self.data.mocap_pos[self.right_target_mocap_id][2] - self.data.qpos[2]
            self.eetarget.right_target.orientation.w = self.data.mocap_quat[self.right_target_mocap_id][0]
            self.eetarget.right_target.orientation.x = self.data.mocap_quat[self.right_target_mocap_id][1]
            self.eetarget.right_target.orientation.y = self.data.mocap_quat[self.right_target_mocap_id][2]
            self.eetarget.right_target.orientation.z = self.data.mocap_quat[self.right_target_mocap_id][3]
        self.eetarget_publisher.Write(self.eetarget)

    def simulation_loop(self):
        with self.viewer.lock():
            for i, _ in enumerate(self.motor_names):
                pos_id = self.pos_ids[i]
                vel_id = self.vel_ids[i]

                q_err  = self.lowcmd.motor_cmd[i].q  - self.data.sensordata[pos_id]
                dq_err = self.lowcmd.motor_cmd[i].dq - self.data.sensordata[vel_id]
                
                idx = i if i < 16 else i + 1
                self.data.ctrl[idx] = q_err * self.lowcmd.motor_cmd[i].kp + dq_err * self.lowcmd.motor_cmd[i].kd + self.lowcmd.motor_cmd[i].tau
                
                if i == 15 or i == 23:
                    q_err  = self.lowcmd.motor_cmd[i].q  - self.data.sensordata[pos_id+1]
                    dq_err = self.lowcmd.motor_cmd[i].dq - self.data.sensordata[vel_id+1]
                    self.data.ctrl[idx+1] = q_err * self.lowcmd.motor_cmd[i].kp + dq_err * self.lowcmd.motor_cmd[i].kd + self.lowcmd.motor_cmd[i].tau
            
            # self.data.qvel[3:5] = 0.0
            mujoco.mj_step(self.model, self.data)

    def viewer_loop(self):
        with self.viewer.lock():
            self.viewer.sync(state_only=True)


def main():
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    _ = SimulationNode()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
