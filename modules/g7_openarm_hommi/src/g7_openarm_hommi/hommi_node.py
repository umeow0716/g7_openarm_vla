from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable
from typing import Literal

from unitree_sdk2py.idl.default import HGLowState_

from g7_openarm_config import general_config
from g7_openarm_idl import Odom, Target, Target_default, array_to_point, array_to_pose
from g7_openarm_utils import ChannelFactoryInitialize, build_pub, build_sub
from g7_openarm_utils.layout import HEAD_ENABLED, LEFT_ARM_ENABLED, RIGHT_ARM_ENABLED

from .camera import BlackCamera, CameraInput, RealSenseCamera
from .config import config
from .observation import (
    Ego3dObservationBuilder,
    ObservationHistory,
    ObservationSample,
)
from .operator import OperatorControl, OperatorState
from .policy import HommiPolicyRunner
from .replay import DatasetReplay, ReplayOptions
from .state import RobotSnapshot, RobotStateProjector
from .trajectory import AtomicTrajectory, TargetPoint, decode_action_chunk, snapshot_target
from .vcam import process_head_frame, process_wrist_frame
from .viewer import ObservationWindow, compose, display_available

CameraKind = Literal["simulation", "realsense"]

# Trajectory replay: time to move from the current pose to the first replayed
# frame (at the start and when --loop wraps), so the arms never jump.
REPLAY_RAMP_S = 1.5


def _validate_control_mode() -> None:
    if not (LEFT_ARM_ENABLED and RIGHT_ARM_ENABLED):
        raise RuntimeError(
            "the ego3d HoMMI policy is bimanual; general.control_mode "
            f"{general_config.control_mode.value!r} must enable both arms"
        )
    if not HEAD_ENABLED:
        print("HoMMI: general.head_enabled=false, look-at actions will not move the head")


def _make_cameras(kind: CameraKind) -> dict[str, CameraInput]:
    if kind == "simulation":
        return {
            "head": BlackCamera(with_depth=True),
            "left": BlackCamera(),
            "right": BlackCamera(),
        }
    if kind != "realsense":
        raise ValueError(f"unknown camera kind {kind!r}")

    serials = config.camera_serials()
    cameras: dict[str, CameraInput] = {}
    try:
        for name, serial in serials.items():
            cameras[name] = RealSenseCamera(
                serial,
                name=name,
                with_depth=name == "head",
                fps=config.camera_fps,
                timeout_ms=config.camera_timeout_ms,
            )
    except BaseException:
        for camera in cameras.values():
            camera.close()
        raise
    return cameras


class HommiNode:
    def __init__(self, *, camera_kind: CameraKind, replay: ReplayOptions | None = None) -> None:
        _validate_control_mode()

        self._stop = threading.Event()
        self._error_lock = threading.Lock()
        self._fatal_error: BaseException | None = None
        self._state_lock = threading.Lock()
        self._lowstate: HGLowState_ | None = None
        self._odom: Odom | None = None
        self._sample_sequence = 0

        self._policy = HommiPolicyRunner(
            device=config.device, inference_precision=config.inference_precision
        )
        self._builder = Ego3dObservationBuilder(
            self._policy.shape_meta,
            crop=self._policy.crop,
            device=self._policy.device,
        )
        if self._builder.horizon != self._policy.n_obs_steps:
            raise ValueError(
                f"shape_meta horizon {self._builder.horizon} != policy "
                f"n_obs_steps {self._policy.n_obs_steps}"
            )
        warmup_start = time.monotonic()
        self._policy.warmup()
        print(f"HoMMI warm-up done in {time.monotonic() - warmup_start:.1f}s", flush=True)
        self._projector = RobotStateProjector(self._policy.gripper_ranges)
        self._history = ObservationHistory(self._builder.horizon)
        self._trajectory = AtomicTrajectory()
        self._latest_robot: RobotSnapshot | None = None
        # Serializes "check operator state + queue a chunk" against pause/start,
        # so a chunk finished just as the operator pauses cannot overwrite the hold.
        self._control_lock = threading.Lock()
        # Dataset replay replaces the cameras only; robot state still comes from FK.
        self._replay = None if replay is None else self._make_replay(replay)
        self._replay_anchored = False
        self._replay_ramp_until = 0.0
        self._cameras = {} if self._replay is not None else _make_cameras(camera_kind)

        self._publisher = build_pub("rt/target", Target)
        self._lowstate_sub = build_sub("rt/lowstate", HGLowState_, self._lowstate_handler)
        self._odom_sub = build_sub("rt/odom", Odom, self._odom_handler)

        # Nothing is published on rt/target until the operator presses Enter in
        # the monitor (the lowlevel/WBC keep the arms at initial_pos meanwhile).
        self._operator = OperatorControl(self._on_operator_change)

        self._threads = tuple(
            threading.Thread(
                target=self._thread_guard, args=(target,), name=f"hommi-{name}", daemon=True
            )
            for name, target in (
                ("observation", self._observation_loop),
                ("inference", self._inference_loop),
                ("publisher", self._publisher_loop),
            )
        )
        for thread in self._threads:
            thread.start()

        print(
            "HoMMI ego3d ready: "
            f"model={self._policy.model_path}, device={self._policy.device}, "
            f"precision={self._policy.precision}, obs_horizon={self._builder.horizon}, "
            f"action_steps={self._policy.n_action_steps}, hz={config.hz:g}, "
            f"publish_hz={config.publish_hz:g}, "
            f"cameras={f'dataset ({replay.mode} replay)' if replay else camera_kind}; "
            "waiting for Enter in the monitor",
            flush=True,
        )

    def _make_replay(self, options: ReplayOptions) -> DatasetReplay:
        meta = self._policy.shape_meta
        step = int(meta["obs"]["camera_head_pointmap"]["down_sample_steps"])
        if abs(config.hz * step - 60.0) > 1e-6:
            print(
                f"HoMMI replay: hommi.hz={config.hz:g} plays the 60 Hz recording at "
                f"{config.hz * step / 60.0:.2f}x speed",
                flush=True,
            )
        return DatasetReplay(
            options,
            head_size=self._builder.head_size,
            wrist_size=self._builder.wrist_size,
            step=step,
        )

    def _on_operator_change(self, state: OperatorState) -> None:
        with self._control_lock:
            self._apply_operator_change(state)

    def _apply_operator_change(self, state: OperatorState) -> None:
        now = time.monotonic()
        if state is OperatorState.PAUSED:
            self._trajectory.freeze(now=now)
        elif state is OperatorState.RUNNING and self._trajectory.sample(now=now) is None:
            # First start: seed from the current pose so the first chunk blends in.
            robot = self._latest_robot
            if robot is not None:
                self._trajectory.hold(
                    snapshot_target(robot, gripper_ranges=self._policy.gripper_ranges), now=now
                )

    def _lowstate_handler(self, msg: HGLowState_) -> None:
        with self._state_lock:
            self._lowstate = msg

    def _odom_handler(self, msg: Odom) -> None:
        with self._state_lock:
            self._odom = msg

    def _latest_state(self) -> tuple[HGLowState_, Odom] | None:
        with self._state_lock:
            if self._lowstate is None or self._odom is None:
                return None
            return self._lowstate, self._odom

    def _thread_guard(self, target: Callable[[], None]) -> None:
        try:
            target()
        except BaseException as exc:
            with self._error_lock:
                if self._fatal_error is None:
                    self._fatal_error = exc
            self._stop.set()
            self._history.wake()

    def _wait_until_next_tick(self, next_tick: float) -> float:
        delay = next_tick - time.monotonic()
        if delay > 0.0:
            self._stop.wait(delay)
            return next_tick
        return time.monotonic()

    def _observation_loop(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            if self._latest_state() is not None:
                # Grab the newest frame of every camera first, then pair them with
                # the newest robot state/FK available right after.
                frames = {name: camera.latest() for name, camera in self._cameras.items()}
                state = self._latest_state()
                assert state is not None
                timestamp = time.monotonic()
                robot = self._projector.snapshot(*state, timestamp=timestamp)
                if self._replay is not None:
                    self._replay_step(robot, timestamp)
                    head, wrist = self._replay.frame()
                else:
                    head = process_head_frame(frames["head"], output_size=self._builder.head_size)
                    wrist = {
                        side: process_wrist_frame(
                            frames[side], output_size=self._builder.wrist_size
                        )
                        for side in ("left", "right")
                    }
                self._sample_sequence += 1
                self._latest_robot = robot
                self._history.append(
                    ObservationSample(
                        sequence=self._sample_sequence,
                        timestamp=timestamp,
                        head=head,
                        wrist_rgb=wrist,
                        robot=robot,
                    )
                )

            next_tick += config.interval
            next_tick = self._wait_until_next_tick(next_tick)

    def _replay_step(self, robot: RobotSnapshot, now: float) -> None:
        """Advance the recording one step per tick while running.

        Before the start and while paused the current frame is held. In
        ``trajectory`` mode this also commands the demonstrated targets.
        """
        replay = self._replay
        assert replay is not None
        if replay.options.mode == "policy":
            if self._operator.state is OperatorState.RUNNING:
                replay.advance()
            return
        with self._control_lock:
            if self._operator.state is not OperatorState.RUNNING:
                return
            ranges = self._policy.gripper_ranges
            if not self._replay_anchored:
                replay.anchor(robot.left_tcp)
                self._replay_anchored = True
                self._ramp_to_current(now)
                return
            if now < self._replay_ramp_until:
                return  # still moving to the first frame
            if replay.advance():
                self._ramp_to_current(now)  # --loop wrapped: move back to the start
                return
            points = [
                replay.target(k, execute_at=now + (k + 1) * config.interval, gripper_ranges=ranges)
                for k in range(self._policy.n_action_steps)
            ]
            self._trajectory.replace(points, now=now, latency_compensation=True)

    def _ramp_to_current(self, now: float) -> None:
        assert self._replay is not None
        self._replay_ramp_until = now + REPLAY_RAMP_S
        first = self._replay.target(
            0, execute_at=self._replay_ramp_until, gripper_ranges=self._policy.gripper_ranges
        )
        self._trajectory.replace([first], now=now, latency_compensation=True)

    def _inference_loop(self) -> None:
        last_sequence = -1
        while not self._stop.is_set():
            samples = self._history.wait_for_new(last_sequence, self._stop)
            if samples is None:
                return

            last_sequence = samples[-1].sequence
            if self._operator.state is not OperatorState.RUNNING:
                continue
            if self._replay is not None and self._replay.options.mode == "trajectory":
                continue  # the observation loop commands the recorded trajectory
            generation = self._operator.generation
            obs = self._builder.build(samples)
            action_chunk = self._policy.predict(obs)
            points = decode_action_chunk(
                action_chunk,
                reference=samples[-1].robot,
                gripper_ranges=self._policy.gripper_ranges,
                interval=config.interval,
                obs_horizon=self._builder.horizon,
            )
            with self._control_lock:
                if (
                    self._operator.state is not OperatorState.RUNNING
                    or self._operator.generation != generation
                ):
                    continue  # paused (or resumed) while inferring: drop the stale chunk
                now = time.monotonic()
                expired = self._trajectory.replace(
                    points, now=now, latency_compensation=config.latency_compensation
                )
            if expired == len(points):
                # Still executed: the trajectory blends to the chunk's final step.
                print(
                    f"HoMMI result {now - samples[-1].timestamp:.3f}s after observation; "
                    f"all {len(points)} steps late, blending to the final step"
                )

    @staticmethod
    def _to_target(point: TargetPoint) -> Target:
        result = Target_default()
        result.left_target = array_to_pose(point.left_pose7)
        result.right_target = array_to_pose(point.right_pose7)
        result.left_gripper = point.left_gripper_openness
        result.right_gripper = point.right_gripper_openness
        result.look_at_point = array_to_point(point.look_at_point)
        return result

    def _publisher_loop(self) -> None:
        # Publish at the WBC rate, interpolating between the 20 Hz action steps.
        next_tick = time.monotonic()
        while not self._stop.is_set():
            target = None
            if self._operator.state is not OperatorState.WAITING:
                target = self._trajectory.sample(now=time.monotonic())
            if target is not None:
                self._publisher.Write(self._to_target(target))

            next_tick += config.publish_interval
            next_tick = self._wait_until_next_tick(next_tick)

    def observation_canvas(self):
        """Operator view of the newest observation (None before the first sample)."""
        sample = self._history.latest()
        if sample is None:
            return None
        robot = sample.robot
        mask = self._policy.crop.mask(
            sample.head.pointmap, robot.head, (robot.left_tcp, robot.right_tcp)
        )
        return compose(
            sample.head.pointmap,
            mask,
            sample.wrist_rgb["left"],
            sample.wrist_rgb["right"],
            crop_ratio=self._policy.policy.obs_encoder.config.eval_crop_ratio,
            status=self._operator.state.value
            + ("" if self._replay is None else f" | {self._replay.status()}"),
        )

    def raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._fatal_error
        if error is not None:
            raise RuntimeError("HoMMI worker failed") from error

    def close(self) -> None:
        self._stop.set()
        self._operator.close()
        self._history.wake()
        self._trajectory.clear()
        for thread in self._threads:
            thread.join(timeout=1.0)
        for camera in self._cameras.values():
            camera.close()


def run(camera_kind: CameraKind, replay: ReplayOptions | None = None) -> None:
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    node = HommiNode(camera_kind=camera_kind, replay=replay)

    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    signal.signal(signal.SIGTERM, request_stop)

    # The observation window is driven from this (main) thread: Qt requires it.
    window = ObservationWindow() if display_available() else None
    if window is None:
        print("HoMMI: no display, observation window disabled", flush=True)
    try:
        while not stop_requested.is_set():
            node.raise_if_failed()
            if window is not None:
                window.show(node.observation_canvas(), wait_ms=66)
            else:
                time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if window is not None:
            window.close()
        node.close()


def main(camera_kind: CameraKind | None = None, replay: ReplayOptions | None = None) -> None:
    # Standalone CLI defaults to the real cameras. Simulation launchers pass the
    # explicit black-camera mode, or ``replay`` to use a recorded dataset as cameras.
    run("realsense" if camera_kind is None else camera_kind, replay)


if __name__ == "__main__":
    main()
