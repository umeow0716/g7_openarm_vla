import time
import openarm_can as oa

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_, IMUState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_, unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__IMUState_
from unitree_sdk2py.utils.hz_sample import RecurrentThread

from .config import config


BUS_CONFIGS = {
    config.base_can: {
        "motor_types": [oa.MotorType.DM8009, oa.MotorType.DM6006] * 4,
        "send_ids": [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
        "recv_ids": [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18],
        "control_modes": [oa.ControlMode.POS_VEL, oa.ControlMode.VEL] * 4,
    },
    config.left_arm_can: {
        "motor_types": [oa.MotorType.DM8009, oa.MotorType.DM8009, oa.MotorType.DM4340, oa.MotorType.DM4340,
                        oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310],
        "send_ids": [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
        "recv_ids": [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18],
        "control_modes": [oa.ControlMode.MIT] * 8,
    },
    config.right_arm_can: {
        "motor_types": [oa.MotorType.DM8009, oa.MotorType.DM8009, oa.MotorType.DM4340, oa.MotorType.DM4340,
                        oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310],
        "send_ids": [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
        "recv_ids": [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18],
        "control_modes": [oa.ControlMode.MIT] * 8,
    }
}


class HardwareNode:
    def __init__(self):
        can_interfaces = list(BUS_CONFIGS)
        self.group = oa.OpenArmGroup(
            can_interfaces=can_interfaces,
            enable_fd=config.can_fd
        )
        
        for can_interface, can_config in BUS_CONFIGS.items():
            arm = self.group.get_openarm(can_interface)

            arm.init_arm_motors(
                can_config["motor_types"],
                can_config["send_ids"],
                can_config["recv_ids"],
                can_config["control_modes"],
            )

            arm.set_callback_mode_all(oa.CallbackMode.STATE)

            print(
                f"{can_interface}: expected responses = "
                f"{arm.expected_response_count()}"
            )
            
            self.group.enable_all()

        self.lowstate = unitree_hg_msg_dds__LowState_()
        self.lowstate_publisher = ChannelPublisher("rt/lowstate", LowState_)
        self.lowstate_publisher.Init()

        self.lowcmd = unitree_hg_msg_dds__LowCmd_()
        self.lowcmd_subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.lowcmd_subscriber.Init(self.lowcmd_handler, 10)

        self.imustate = unitree_hg_msg_dds__IMUState_()
        self.imustate_subscriber = ChannelSubscriber("rt/imustate", IMUState_)
        self.imustate_subscriber.Init(self.imustate_handler, 10)
        
        self.control_thread = RecurrentThread(
            name="control_thread",
            target=self.control_loop,
            interval=config.interval,
        )
        self.control_thread.Start()

    def lowcmd_handler(self, msg: LowCmd_):
        self.lowcmd = msg

    def imustate_handler(self, msg: IMUState_):
        self.imustate = msg

    def control_loop(self):
        self.group.refresh_all_and_recv(3000)
        
        base = self.group.get_openarm(config.base_can)
        for i, motor in enumerate(base.get_arm().get_motors()):
            self.lowstate.motor_state[i].q  = motor.get_position()
            self.lowstate.motor_state[i].dq = motor.get_velocity()
            self.lowstate.motor_state[i].tau_est = motor.get_torque()

        left_arm = self.group.get_openarm(config.left_arm_can)
        for i, motor in enumerate(left_arm.get_arm().get_motors()):
            self.lowstate.motor_state[8+i].q  = motor.get_position()
            self.lowstate.motor_state[8+i].dq = motor.get_velocity()
            self.lowstate.motor_state[8+i].tau_est = motor.get_torque()

        right_arm = self.group.get_openarm(config.right_arm_can)
        for i, motor in enumerate(right_arm.get_arm().get_motors()):
            self.lowstate.motor_state[16+i].q  = motor.get_position()
            self.lowstate.motor_state[16+i].dq = motor.get_velocity()
            self.lowstate.motor_state[16+i].tau_est = motor.get_torque()
        
        self.lowstate.imu_state = self.imustate
        self.lowstate_publisher.Write(self.lowstate)
        
        for i in range(0, 8, 2):
            cmd = oa.PosVelParam(q=self.lowcmd.motor_cmd[i].q, dq=20.0)
            base.get_arm().posvel_control_one(i, cmd)
        for i in range(1, 8, 2):
            cmd = oa.VelParam(dq=self.lowcmd.motor_cmd[i].dq)
            base.get_arm().vel_control_one(i, cmd)

        left_cmds = [
            oa.MITParam(
                q=self.lowcmd.motor_cmd[8+i].q,
                dq=self.lowcmd.motor_cmd[8+i].dq,
                kp=self.lowcmd.motor_cmd[8+i].kp,
                kd=self.lowcmd.motor_cmd[8+i].kd,
                tau=self.lowcmd.motor_cmd[8+i].tau,
            )
            for i in range(8)
        ]
        right_cmds = [
            oa.MITParam(
                q=self.lowcmd.motor_cmd[16+i].q,
                dq=self.lowcmd.motor_cmd[16+i].dq,
                kp=self.lowcmd.motor_cmd[16+i].kp,
                kd=self.lowcmd.motor_cmd[16+i].kd,
                tau=self.lowcmd.motor_cmd[16+i].tau,
            )
            for i in range(8)
        ]
        left_arm.get_arm().mit_control_all(left_cmds)
        right_arm.get_arm().mit_control_all(right_cmds)


def main():
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    _ = HardwareNode()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
