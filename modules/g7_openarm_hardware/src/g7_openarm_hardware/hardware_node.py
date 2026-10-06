import damiao_can as dc
import time

from g7_openarm_utils.layout.control_layout import (
    Joint,
    BASE_ENABLED,
    LEFT_ARM_ENABLED,
    RIGHT_ARM_ENABLED,
    BASE_JOINT,
    LEFT_ARM_JOINT,
    RIGHT_ARM_JOINT,
    low_idx
)
from g7_openarm_utils.layout.gripper import (
    motor_to_openness,
    openness_to_motor,
)
from g7_openarm_utils.unitree import (
    ChannelFactoryInitialize,
    build_sub,
    build_pub,
    build_thread,
)
from unitree_sdk2py.idl.default import (
    HGLowCmd_,
    HGLowState_,
    unitree_hg_msg_dds__LowState_ as HGLowState_default,
)

from .config import config

bus_config = {
    config.base_can: {
       "motor_types": [
            dc.MotorType.DM8009, dc.MotorType.DM6006,
            dc.MotorType.DM8009, dc.MotorType.DM6006,
            dc.MotorType.DM8009, dc.MotorType.DM6006,
            dc.MotorType.DM8009, dc.MotorType.DM6006,
        ],
        "send_ids": config.base_ids,
        "recv_ids": [n + 16 for n in config.base_ids],
        "control_modes": [
            dc.ControlMode.POS_VEL, dc.ControlMode.VEL,
            dc.ControlMode.POS_VEL, dc.ControlMode.VEL,
            dc.ControlMode.POS_VEL, dc.ControlMode.VEL,
            dc.ControlMode.POS_VEL, dc.ControlMode.VEL
        ],
    },
    config.left_arm_can: {
        "motor_types": [
            dc.MotorType.DM8009, dc.MotorType.DM8009,
            dc.MotorType.DM4340, dc.MotorType.DM4340,
            dc.MotorType.DM4310, dc.MotorType.DM4310,
            dc.MotorType.DM4310, dc.MotorType.DM4310,
        ],
        "send_ids": [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
        "recv_ids": [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18],
        "control_modes": [
            dc.ControlMode.MIT, dc.ControlMode.MIT,
            dc.ControlMode.MIT, dc.ControlMode.MIT,
            dc.ControlMode.MIT, dc.ControlMode.MIT,
            dc.ControlMode.MIT, dc.ControlMode.POS_FORCE,
        ],
    },
    config.right_arm_can: {
        "motor_types": [
            dc.MotorType.DM8009, dc.MotorType.DM8009,
            dc.MotorType.DM4340, dc.MotorType.DM4340,
            dc.MotorType.DM4310, dc.MotorType.DM4310,
            dc.MotorType.DM4310, dc.MotorType.DM4310,
        ],
        "send_ids": [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
        "recv_ids": [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18],
        "control_modes": [
            dc.ControlMode.MIT, dc.ControlMode.MIT,
            dc.ControlMode.MIT, dc.ControlMode.MIT,
            dc.ControlMode.MIT, dc.ControlMode.MIT,
            dc.ControlMode.MIT, dc.ControlMode.POS_FORCE,
        ],
    },
}


def get_enable_interfaces() -> list[str]:
    result = []
    if BASE_ENABLED:
        result.append(config.base_can)
    if LEFT_ARM_ENABLED or RIGHT_ARM_ENABLED:
        result.append(config.left_arm_can)
        result.append(config.right_arm_can)
    return result


def _read_base(device: dc.DamiaoCAN, lowstate: HGLowState_):
    dev_list = device.get_motors()
    assert len(dev_list) == 8

    low_list = [
        lowstate.motor_state[low_idx(joint)] for joint in BASE_JOINT
    ]
    
    for i, (dev, low) in enumerate(zip(dev_list, low_list)):
        low.q  = dev.get_position() * config.base_direction[i]
        low.dq = dev.get_velocity() * config.base_direction[i]
        low.tau_est = dev.get_torque() * config.base_direction[i]


def _read_left_arm(device: dc.DamiaoCAN, lowstate: HGLowState_):
    dev_list = device.get_motors()
    assert len(dev_list) == 8

    low_list = [
        lowstate.motor_state[low_idx(joint)] for joint in LEFT_ARM_JOINT
    ]
    
    for i, (dev, low) in enumerate(zip(dev_list, low_list)):
        low.q  = dev.get_position() * config.left_arm_direction[i]
        low.dq = dev.get_velocity() * config.left_arm_direction[i]
        low.tau_est = dev.get_torque() * config.left_arm_direction[i]


    close_q = config.left_gripper_close
    open_q  = config.left_gripper_open

    gripper_low = lowstate.motor_state[low_idx(Joint.L8)]
    gripper_dev = dev_list[-1]

    gripper_low.q  = motor_to_openness(gripper_dev.get_position(), close_q, open_q)
    gripper_low.dq = gripper_dev.get_velocity() / (open_q - close_q)

def _read_right_arm(device: dc.DamiaoCAN, lowstate: HGLowState_):
    dev_list = device.get_motors()
    assert len(dev_list) == 8

    low_list = [
        lowstate.motor_state[low_idx(joint)] for joint in RIGHT_ARM_JOINT
    ]
    
    for i, (dev, low) in enumerate(zip(dev_list, low_list)):
        low.q  = dev.get_position() * config.right_arm_direction[i]
        low.dq = dev.get_velocity() * config.right_arm_direction[i]
        low.tau_est = dev.get_torque() * config.right_arm_direction[i]

    open_q  = config.right_gripper_open
    close_q = config.right_gripper_close

    gripper_low = lowstate.motor_state[low_idx(Joint.R8)]
    gripper_dev = dev_list[-1]

    gripper_low.q  = motor_to_openness(gripper_dev.get_position(), close_q, open_q)
    gripper_low.dq = gripper_dev.get_velocity() / (open_q - close_q)

def _write_base(device: dc.DamiaoCAN, lowcmd: HGLowCmd_):
    dev_list = device.get_motors()
    assert len(dev_list) == 8

    low_list = [
        lowcmd.motor_cmd[low_idx(joint)] for joint in BASE_JOINT
    ]

    for i, (dev, low) in enumerate(zip(dev_list, low_list)):
        if dev.get_motor_type() == dc.MotorType.DM8009:
            device.posvel_control_one(i, dc.PosVelParam(q=low.q * config.base_direction[i], dq=20.0))
        elif dev.get_motor_type() == dc.MotorType.DM6006:
            device.vel_control_one(i, dc.VelParam(dq=low.dq * config.base_direction[i]))


def _write_left_arm(device: dc.DamiaoCAN, lowcmd: HGLowCmd_):
    low_list = [
        lowcmd.motor_cmd[low_idx(joint)] for joint in LEFT_ARM_JOINT
    ]
    
    for i, low in enumerate(low_list):
        device.mit_control_one(i, dc.MITParam(
            kp=low.kp,
            kd=low.kd,
            q=low.q * config.left_arm_direction[i],
            dq=low.dq * config.left_arm_direction[i],
            tau=low.tau * config.left_arm_direction[i],
        ))

    open_q  = config.left_gripper_open
    close_q = config.left_gripper_close
    gripper_low = lowcmd.motor_cmd[low_idx(Joint.L8)]
    device.posforce_control_one(7, dc.PosForceParam(
        q=openness_to_motor(gripper_low.q, close_q, open_q),
        dq=20.0,
        i=0.5,
    ))


def _write_right_arm(device: dc.DamiaoCAN, lowcmd: HGLowCmd_):
    low_list = [
        lowcmd.motor_cmd[low_idx(joint)] for joint in RIGHT_ARM_JOINT
    ]
    
    open_q  = config.right_gripper_open
    close_q = config.right_gripper_close

    for i, low in enumerate(low_list):
        device.mit_control_one(i, dc.MITParam(
            kp=low.kp,
            kd=low.kd,
            q=low.q * config.right_arm_direction[i],
            dq=low.dq * config.right_arm_direction[i],
            tau=low.tau * config.right_arm_direction[i],
        ))

    gripper_low = lowcmd.motor_cmd[low_idx(Joint.R8)]
    device.posforce_control_one(7, dc.PosForceParam(
        q=openness_to_motor(gripper_low.q, close_q, open_q),
        dq=20.0,
        i=5000.0
    ))

class HardwareNode:
    def __init__(self):
        self.can_interfaces = get_enable_interfaces()
        for iface in self.can_interfaces:
            print(f"Initializing CAN interface {iface}...")
            helper = dc.CANHelper(iface)
            helper.set_down()
            helper.set_bitrate(1_000_000, 5_000_000, config.can_fd)
            helper.set_up()

        self.group = dc.DamiaoCANGroup(self.can_interfaces, config.can_fd)
        for iface in self.can_interfaces:
            device = self.group.get_device(iface)
            device.init_motors(
                bus_config[iface]["motor_types"],
                bus_config[iface]["send_ids"],
                bus_config[iface]["recv_ids"],
                bus_config[iface]["control_modes"],
            )
            device.set_callback_mode_all(dc.CallbackMode.STATE)
            print(f"{iface}: expected responses = {device.expected_response_count()}")

        self.group.enable_all()
        for _try in range(5):
            self.group.flush_rx()
            self.group.refresh_all()
            res = self.group.recv_all(1_000_000)
            if res.ok:
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("Initialize motor failed over 5 times...")

        self.lowcmd: HGLowCmd_ | None = None

        self.lowcmd_sub = build_sub("rt/lowcmd", HGLowCmd_, self.lowcmd_handle)
        self.lowstate_pub = build_pub("rt/lowstate", HGLowState_)

        self.control_thread = build_thread(config.hz, self.control_loop)

    def lowcmd_handle(self, msg: HGLowCmd_):
        self.lowcmd = msg

    def control_loop(self):
        self.group.flush_rx()
        self.group.refresh_all()
        self.group.recv_all(8_000)

        lowstate = HGLowState_default()
        if BASE_ENABLED:
            device = self.group.get_device(config.base_can)
            _read_base(device, lowstate)
        if LEFT_ARM_ENABLED or RIGHT_ARM_ENABLED:
            left_device = self.group.get_device(config.left_arm_can)
            right_device = self.group.get_device(config.right_arm_can)
            _read_left_arm(left_device, lowstate)
            _read_right_arm(right_device, lowstate)

        if self.lowstate_pub is not None:
            self.lowstate_pub.Write(lowstate)

        if self.lowcmd is None:
            return

        if BASE_ENABLED:
            device = self.group.get_device(config.base_can)
            _write_base(device, self.lowcmd)

        if LEFT_ARM_ENABLED:
            device = self.group.get_device(config.left_arm_can)
            _write_left_arm(device, self.lowcmd)

        if RIGHT_ARM_ENABLED:
            device = self.group.get_device(config.right_arm_can)
            _write_right_arm(device, self.lowcmd)

def main():
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    node = HardwareNode()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
