"""Replay a recorded iPhUMI demonstration through the HoMMI node.

Two modes, both showing the recording in the observation window:

* ``policy``: the recording replaces only the cameras (head pointmap / RGB
  and the two wrist images, in the training geometry, see ``dataset``). Robot
  state still comes from FK, so observation building, inference, action
  decoding and execution are exactly the deployment path.
* ``trajectory``: no inference; the robot follows the demonstrated end-effector
  trajectory (both TCPs, gripper openness, head look-at point). The recording
  is anchored to the robot by mapping the demonstration's left TCP at the
  start onto the robot's left TCP; the right TCP and the look-at point keep
  their demonstrated pose relative to it.

Playback is driven by the node's observation loop: one dataset step
(``down_sample_steps`` 60 Hz frames, i.e. the 20 Hz training rate) per
observation tick while running. Before the start and while paused the
current frame is held.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from .dataset import Episode, discover, load_episode, select_demonstration
from .geometry import matrix_to_pose7_wxyz
from .gripper import GripperWidthRange
from .trajectory import TargetPoint
from .vcam import HeadObservation

ReplayMode = Literal["policy", "trajectory"]


@dataclass(frozen=True, slots=True)
class ReplayOptions:
    """Picklable replay settings (passed from the launcher to the node process)."""

    root: str
    episode: str = "0"
    mode: ReplayMode = "policy"
    loop: bool = False


class DatasetReplay:
    def __init__(
        self, options: ReplayOptions, *, head_size: int, wrist_size: int, step: int
    ) -> None:
        recordings = discover(options.root)
        recording = select_demonstration(recordings, options.episode)
        print(f"HoMMI replay: loading {recording.title} ...", flush=True)
        start = time.monotonic()
        self.episode: Episode = load_episode(recording, recordings, head_size=head_size)
        self.options = options
        self.frames = list(range(0, self.episode.length, int(step)))
        sizes = {"head": head_size, "left": wrist_size, "right": wrist_size}
        self._rgb = self.episode.load_rgb(set(self.frames), sizes)
        self._head_size = head_size
        self._world_from_data = np.eye(4)
        self.cursor = 0
        print(
            f"HoMMI replay: {recording.title}: {len(self.frames)} steps "
            f"({self.episode.length / 60.0:.1f}s), loaded in {time.monotonic() - start:.1f}s",
            flush=True,
        )

    @property
    def finished(self) -> bool:
        return self.cursor >= len(self.frames) - 1

    @property
    def index(self) -> int:
        return self.frames[self.cursor]

    def advance(self) -> bool:
        """Step to the next frame; at the end hold it, or restart with ``loop``.

        Returns True when playback wrapped around to the first frame.
        """
        if not self.finished:
            self.cursor += 1
            return False
        if self.options.loop:
            self.cursor = 0
            return True
        return False

    def anchor(self, robot_left_tcp: npt.NDArray) -> None:
        """Map the demonstration's left TCP at the current frame onto ``robot_left_tcp``."""
        data_left = self.episode.tcp["left"][self.index]
        self._world_from_data = robot_left_tcp.astype(np.float64) @ np.linalg.inv(data_left)

    def target(
        self,
        offset: int,
        *,
        execute_at: float,
        gripper_ranges: dict[str, GripperWidthRange],
    ) -> TargetPoint:
        """Demonstrated target ``offset`` steps after the current frame (clamped)."""
        index = self.frames[min(self.cursor + offset, len(self.frames) - 1)]
        ep, world = self.episode, self._world_from_data

        def pose(side: str) -> tuple[float, ...]:
            return tuple(float(x) for x in matrix_to_pose7_wxyz(world @ ep.tcp[side][index]))

        look_at = world[:3, :3] @ ep.lookat_world(index) + world[:3, 3]
        return TargetPoint(
            execute_at=execute_at,
            left_pose7=pose("left"),
            right_pose7=pose("right"),
            left_gripper_openness=gripper_ranges["left"].to_openness(
                float(ep.width["left"][index])
            ),
            right_gripper_openness=gripper_ranges["right"].to_openness(
                float(ep.width["right"][index])
            ),
            look_at_point=(float(look_at[0]), float(look_at[1]), float(look_at[2])),
        )

    def frame(self) -> tuple[HeadObservation, dict[str, npt.NDArray[np.uint8]]]:
        index = self.index
        head = HeadObservation(
            rgb=self._rgb["head"][index], pointmap=self.episode.pointmap(index, self._head_size)
        )
        return head, {side: self._rgb[side][index] for side in ("left", "right")}

    def status(self) -> str:
        done = " (end)" if self.finished and not self.options.loop else ""
        return f"{self.options.mode} replay, step {self.cursor + 1}/{len(self.frames)}{done}"
