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
from .observation import Ego3dObservationBuilder, ObservationHistory, ObservationSample
from .policy import HommiPolicyRunner
from .state import RobotStateProjector
from .trajectory import AtomicTrajectory, TargetPoint, decode_action_chunk
from .vcam import process_head_frame, process_wrist_frame

CameraKind = Literal["simulation", "realsense"]


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
    def __init__(self, *, camera_kind: CameraKind) -> None:
        _validate_control_mode()

        self._stop = threading.Event()
        self._error_lock = threading.Lock()
        self._fatal_error: BaseException | None = None
        self._state_lock = threading.Lock()
        self._lowstate: HGLowState_ | None = None
        self._odom: Odom | None = None
        self._sample_sequence = 0

        self._policy = HommiPolicyRunner(device=config.device)
        self._builder = Ego3dObservationBuilder(
            self._policy.shape_meta,
            depth_range=self._policy.depth_range,  # type: ignore[arg-type]
            device=self._policy.device,
        )
        if self._builder.horizon != self._policy.n_obs_steps:
            raise ValueError(
                f"shape_meta horizon {self._builder.horizon} != policy "
                f"n_obs_steps {self._policy.n_obs_steps}"
            )
        self._projector = RobotStateProjector(self._policy.gripper_ranges)
        self._history = ObservationHistory(self._builder.horizon)
        self._trajectory = AtomicTrajectory()
        self._cameras = _make_cameras(camera_kind)

        self._publisher = build_pub("rt/target", Target)
        self._lowstate_sub = build_sub("rt/lowstate", HGLowState_, self._lowstate_handler)
        self._odom_sub = build_sub("rt/odom", Odom, self._odom_handler)

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
            f"cameras={camera_kind}"
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
                head = process_head_frame(frames["head"], output_size=self._builder.head_size)
                wrist = {
                    side: process_wrist_frame(frames[side], output_size=self._builder.wrist_size)
                    for side in ("left", "right")
                }
                self._sample_sequence += 1
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

    def _inference_loop(self) -> None:
        last_sequence = -1
        while not self._stop.is_set():
            samples = self._history.wait_for_new(last_sequence, self._stop)
            if samples is None:
                return

            last_sequence = samples[-1].sequence
            obs = self._builder.build(samples)
            action_chunk = self._policy.predict(obs)
            points = decode_action_chunk(
                action_chunk,
                reference=samples[-1].robot,
                gripper_ranges=self._policy.gripper_ranges,
                interval=config.interval,
                obs_horizon=self._builder.horizon,
            )
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
            target = self._trajectory.sample(now=time.monotonic())
            if target is not None:
                self._publisher.Write(self._to_target(target))

            next_tick += config.publish_interval
            next_tick = self._wait_until_next_tick(next_tick)

    def raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._fatal_error
        if error is not None:
            raise RuntimeError("HoMMI worker failed") from error

    def close(self) -> None:
        self._stop.set()
        self._history.wake()
        self._trajectory.clear()
        for thread in self._threads:
            thread.join(timeout=1.0)
        for camera in self._cameras.values():
            camera.close()


def run(camera_kind: CameraKind) -> None:
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    node = HommiNode(camera_kind=camera_kind)

    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    signal.signal(signal.SIGTERM, request_stop)

    try:
        while not stop_requested.is_set():
            node.raise_if_failed()
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()


def main(camera_kind: CameraKind | None = None) -> None:
    # Standalone CLI defaults to the real cameras. Simulation launchers pass the
    # explicit black-camera mode.
    run("realsense" if camera_kind is None else camera_kind)


if __name__ == "__main__":
    main()
