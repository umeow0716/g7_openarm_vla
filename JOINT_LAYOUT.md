# G7 motor / joint layout contract

This project treats names as the robot API. Integer indices are transport or
library implementation details and are resolved once from names before a
control loop starts.

## Canonical logical motors

`g7_openarm_utils.joint_layout` is the single source of truth.

- Base: `AMR_FL`, `AMR_FLW`, `AMR_FR`, `AMR_FRW`, `AMR_RL`, `AMR_RLW`,
  `AMR_RR`, `AMR_RRW`.
- Left arm: `L_1` through `L_7`, then logical gripper `gripper_L`.
- Right arm: `R_1` through `R_7`, then logical gripper `gripper_R`.
- A logical gripper maps to two model joints (`LL/LR` or `RL/RR`). All other
  logical motors map to one model joint.

Unitree `LowState_` / `LowCmd_` are fixed-size, index-only DDS arrays. The G7
protocol uses the first 24 slots, but first-party code resolves those slots via
`motor_index(name)` / `motor_command(...)`; it must not hand-code slot numbers.
`OpenArmCmd` is similarly a fixed 16-value wire format and is converted only at
its transport boundary through the canonical arm-command name map.

## MuJoCo

`MuJoCoModelLayout` resolves each model joint by name at model initialization.
For every scalar joint it caches these independently:

- `jnt_qposadr` (q position index)
- `jnt_dofadr` (q velocity / DoF index)
- actuator id targeting that joint
- named primary torque-sensor address for each logical motor

The floating base has the explicit MJCF name `floating_base_joint`. Its seven
q components and six v components are separately named. Code must never infer
qvel indices from qpos indices; a free joint makes `nq != nv`.

## PinnZoo shared library

The Python binding reads `config_names`, `vel_names`, `torque_names`, and
`kinematics_bodies` from the loaded library and validates the exact G7 name
sets before control starts. It then caches name-to-index maps. q and v lookups
are always independent.

The packaged G7 binaries are compatible legacy API v1 libraries: they already
export the name arrays, so the refactored runtime can use them safely. The
updated PinnZoo generator produces API v2 libraries that additionally export:

- `get_config_index(name)`, `get_vel_index(name)`, `get_torque_index(name)`
- `get_joint_q_index(name)`, `get_joint_v_index(name)`
- `get_joint_nq(name)`, `get_joint_nv(name)`
- joint-name/count metadata and kinematics output width

The v2 binding cross-checks those lookup functions against the exported name
arrays and refuses to start if they disagree.

## Why motor names are not sent every control tick

No realtime DDS message carries strings. Names are used to establish and
validate identity at initialization; the realtime path uses cached integer
indices and numeric arrays. This avoids string lookup/allocation in the control
loop while keeping every index boundary auditable by name.

A startup-only layout-handshake IDL would be useful only if independently
versioned binaries need to negotiate layouts over DDS. It is intentionally not
added here because the current processes share this workspace's canonical
layout module, and changing the realtime wire protocol would add deployment
risk without improving the current mapping path.
