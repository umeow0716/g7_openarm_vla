"""CSV log of the end-effector target vs. the measured (forward-kinematics) pose.

One row per WBC control tick. Plot it with ``scripts/plot_ee_tracking.py``.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ik_solver import TaskState

LOG_DIR = Path("logs") / "ee-tracking"

_POSE_FIELDS = ("x", "y", "z", "qw", "qx", "qy", "qz")
COLUMNS = ["t"] + [
    f"{arm}_{kind}_{field}"
    for arm in ("left", "right")
    for kind in ("target", "ee")
    for field in _POSE_FIELDS
]


class TrackingLogger:
    def __init__(self, log_dir: Path = LOG_DIR) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.path = log_dir / f"{timestamp}.csv"
        # Line-buffered: the launcher stops nodes with SIGTERM/SIGKILL, so no
        # close() is guaranteed; every finished row is already on disk.
        self._file = self.path.open("w", buffering=1)
        self._file.write(",".join(COLUMNS) + "\n")
        self._t0 = time.monotonic()

    def write(self, target: TaskState, state: TaskState) -> None:
        values = [time.monotonic() - self._t0]
        for target_pose, ee_pose in (
            (target.left_pose, state.left_pose),
            (target.right_pose, state.right_pose),
        ):
            values.extend(target_pose.pos)
            values.extend(target_pose.quat)
            values.extend(ee_pose.pos)
            values.extend(ee_pose.quat)
        self._file.write(",".join(f"{v:.6f}" for v in values) + "\n")
