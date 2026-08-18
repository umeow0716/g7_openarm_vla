from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable
from typing import Literal

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

from g7_openarm_config import ControlMode, general_config
from g7_openarm_idl import EETarget, Odom
from g7_openarm_utils.idl import array_to_pose

from .camera import BlackCamera, CameraInput, RealSenseCamera
from .config import config
from .observation import HommiObservationBuilder, ObservationHistory, ObservationSample
from .policy import HommiPolicyRunner
from .state import RobotStateProjector
from .trajectory import AtomicTrajectory, TargetPoint, decode_action_chunk

CameraKind = Literal["simulation", "realsense"]


def _validate_control_mode() -> None:
    mode = general_config.control_mode
    if mode is ControlMode.BASE_ONLY:
        raise RuntimeError("HoMMI requires an arm-enabled control mode")

    if config.arm == "left" and mode in (
        ControlMode.RIGHT_ARM,
        ControlMode.RIGHT_ARM_ONLY,
    ):
        raise RuntimeError(
            f"hommi.arm='left' is incompatible with general.control_mode={mode.value!r}"
        )
    if config.arm == "right" and mode in (
        ControlMode.LEFT_ARM,
        ControlMode.LEFT_ARM_ONLY,
    ):
        raise RuntimeError(
            f"hommi.arm='right' is incompatible with general.control_mode={mode.value!r}"
        )


def _make_camera(kind: CameraKind) -> CameraInput:
    if kind == "simulation":
        return BlackCamera()
    if kind == "realsense":
        return RealSenseCamera(
            fps=config.camera_fps,
            timeout_ms=config.camera_timeout_ms,
        )
    raise ValueError(f"unknown camera kind {kind!r}")


class HommiNode:
    def __init__(self, *, camera_kind: CameraKind) -> None:
        _validate_control_mode()

        self._stop = threading.Event()
        self._error_lock = threading.Lock()
        self._fatal_error: BaseException | None = None
        self._state_lock = threading.Lock()
        self._lowstate: LowState_ | None = None
        self._odom: Odom | None = None
        self._sample_sequence = 0
        self._last_target: TargetPoint | None = None

        self._policy = HommiPolicyRunner(device=config.device)
        self._builder = HommiObservationBuilder(
            self._policy.shape_meta,
            arm=config.arm,
            device=self._policy.device,
        )
        self._history = ObservationHistory(self._builder.horizon)
        self._trajectory = AtomicTrajectory()
        self._projector = RobotStateProjector()
        self._camera = _make_camera(camera_kind)

        self._publisher = ChannelPublisher("rt/eetarget", EETarget)
        self._publisher.Init()

        self._lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._odom_subscriber = ChannelSubscriber("rt/odom", Odom)
        self._lowstate_subscriber.Init(self._lowstate_handler, 0)
        self._odom_subscriber.Init(self._odom_handler, 0)

        self._threads = (
            threading.Thread(
                target=self._thread_guard,
                args=(self._observation_loop,),
                name="hommi-observation",
                daemon=True,
            ),
            threading.Thread(
                target=self._thread_guard,
                args=(self._inference_loop,),
                name="hommi-inference",
                daemon=True,
            ),
            threading.Thread(
                target=self._thread_guard,
                args=(self._publisher_loop,),
                name="hommi-publisher",
                daemon=True,
            ),
        )
        for thread in self._threads:
            thread.start()

        print(
            "HoMMI ready: "
            f"model={self._policy.model_path}, "
            f"backend={self._policy.backend}, "
            f"device={self._policy.device}, "
            f"precision={self._policy.precision}, "
            f"arm={config.arm}, "
            f"obs_horizon={self._builder.horizon}, "
            f"action_steps={self._policy.n_action_steps}, "
            f"hz={config.hz:g}"
        )

    def _lowstate_handler(self, msg: LowState_) -> None:
        with self._state_lock:
            self._lowstate = msg

    def _odom_handler(self, msg: Odom) -> None:
        with self._state_lock:
            self._odom = msg

    def _latest_state(self) -> tuple[LowState_, Odom] | None:
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
            state = self._latest_state()
            if state is not None:
                # Capture first, then pair it with the newest robot state/FK available
                # immediately after the frame arrives.
                rgb = self._camera.read()
                state = self._latest_state()
                if state is not None:
                    timestamp = time.monotonic()
                    lowstate, odom = state
                    robot = self._projector.snapshot(
                        lowstate,
                        odom,
                        timestamp=timestamp,
                    )
                    self._sample_sequence += 1
                    self._history.append(
                        ObservationSample(
                            sequence=self._sample_sequence,
                            timestamp=timestamp,
                            rgb=rgb,
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
                arm=config.arm,
                interval=config.interval,
                obs_horizon=self._builder.horizon,
            )
            now = time.monotonic()
            queued = self._trajectory.replace(
                points,
                now=now,
                drop_expired=config.latency_compensation,
            )
            if queued == 0 and config.latency_compensation:
                # Inference exceeded the full executable chunk. Keep the last
                # command rather than replaying an already-expired trajectory.
                print(
                    "HoMMI inference result expired before it could be queued; "
                    "holding the previous target"
                )

    @staticmethod
    def _to_eetarget(point: TargetPoint) -> EETarget:
        return EETarget(
            array_to_pose(point.left_pose7),
            array_to_pose(point.right_pose7),
            point.left_gripper_openness,
            point.right_gripper_openness,
        )

    def _publisher_loop(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            due = self._trajectory.pop_due(now=time.monotonic())
            if due is not None:
                self._last_target = due
            if self._last_target is not None:
                self._publisher.Write(self._to_eetarget(self._last_target))

            next_tick += config.interval
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
        self._camera.close()


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
    # Standalone CLI defaults to the real camera. Simulation launchers pass the
    # explicit black-camera mode.
    run("realsense" if camera_kind is None else camera_kind)


if __name__ == "__main__":
    main()
