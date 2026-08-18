from __future__ import annotations

from pathlib import Path

from g7_openarm_utils import MOTOR_NAMES, UNITREE_HG_MOTOR_ARRAY_SIZE


SDK_ROOT = (
    Path(__file__).resolve().parents[1]
    / "third_party/unitree_sdk2_python/unitree_sdk2py/idl/unitree_hg/msg/dds_"
)


def test_unitree_hg_lowstate_lowcmd_motor_arrays_match_project_contract() -> None:
    lowstate = (SDK_ROOT / "_LowState_.py").read_text()
    lowcmd = (SDK_ROOT / "_LowCmd_.py").read_text()

    assert (
        "motor_state: types.array['unitree_sdk2py.idl.unitree_hg.msg.dds_.MotorState_', 35]"
        in lowstate
    )
    assert (
        "motor_cmd: types.array['unitree_sdk2py.idl.unitree_hg.msg.dds_.MotorCmd_', 35]"
        in lowcmd
    )
    assert UNITREE_HG_MOTOR_ARRAY_SIZE == 35
    assert len(MOTOR_NAMES) == 24
    assert len(MOTOR_NAMES) <= UNITREE_HG_MOTOR_ARRAY_SIZE
