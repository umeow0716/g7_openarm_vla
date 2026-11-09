"""Use a recorded iPhUMI demonstration as the HoMMI node's cameras.

Only the camera inputs come from the dataset: the head pointmap / RGB and the
two wrist images, already in the training geometry (see ``dataset``). The
robot state (TCP poses, gripper widths) still comes from the robot's FK, so
the rest of the pipeline -- observation building, inference, action decoding
and execution -- is exactly the deployment path.

Playback is driven by the node's observation loop: one dataset step
(``down_sample_steps`` 60 Hz frames, i.e. the 20 Hz training rate) per
observation tick while the operator has started the policy. Before the start
and while paused the current frame is held.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .dataset import Episode, discover, load_episode, select_demonstration
from .vcam import HeadObservation


@dataclass(frozen=True, slots=True)
class ReplayOptions:
    """Picklable replay settings (passed from the launcher to the node process)."""

    root: str
    episode: str = "0"
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

    def advance(self) -> None:
        """Step to the next frame; at the end hold it, or restart with ``loop``."""
        if not self.finished:
            self.cursor += 1
        elif self.options.loop:
            self.cursor = 0

    def frame(self) -> tuple[HeadObservation, dict[str, npt.NDArray[np.uint8]]]:
        index = self.index
        head = HeadObservation(
            rgb=self._rgb["head"][index], pointmap=self.episode.pointmap(index, self._head_size)
        )
        return head, {side: self._rgb[side][index] for side in ("left", "right")}

    def status(self) -> str:
        done = " (end)" if self.finished and not self.options.loop else ""
        return f"dataset step {self.cursor + 1}/{len(self.frames)}{done}"
