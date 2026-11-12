"""Check that the robot's TCP frames follow HoMMI's TCP convention.

Run while the robot stack is up (it only listens to rt/lowstate and rt/odom):

    uv run python scripts/check_hommi_frames.py            # print TCP / head axes
    uv run python scripts/check_hommi_frames.py --cameras  # also show the wrist images

HoMMI (iPhUMI) TCP convention, which the ego3d policy was trained with:
  * origin at the fingertip centre,
  * +z along the approach direction (out of the gripper),
  * +x along the finger opening direction,
  * +y pointing AWAY from the wrist camera (the camera sits on the -y side),
    so the fingers appear at the BOTTOM of the wrist image.

For each arm, look at the robot and check that the printed +y direction points
from the wrist camera across the gripper. If an arm is flipped, its ``*_tcp``
frame in the MJCF/URDF must be rotated 180 deg about z (the PinnZoo model is
generated from that URDF).
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
from unitree_sdk2py.idl.default import HGLowState_

from g7_openarm_hommi.config import config
from g7_openarm_hommi.geometry import quaternion_wxyz_to_matrix
from g7_openarm_hommi.gripper import GripperWidthRange
from g7_openarm_hommi.state import RobotStateProjector, head_extrinsic
from g7_openarm_idl import Odom
from g7_openarm_utils import ChannelFactoryInitialize, build_sub

BODY_DIRECTIONS = {
    "forward": np.array([1.0, 0.0, 0.0]),
    "backward": np.array([-1.0, 0.0, 0.0]),
    "left": np.array([0.0, 1.0, 0.0]),
    "right": np.array([0.0, -1.0, 0.0]),
    "up": np.array([0.0, 0.0, 1.0]),
    "down": np.array([0.0, 0.0, -1.0]),
}


def describe(axis_in_body: np.ndarray) -> str:
    """Name the robot-body direction closest to ``axis_in_body``."""
    name, ref = max(BODY_DIRECTIONS.items(), key=lambda item: float(item[1] @ axis_in_body))
    angle = np.degrees(np.arccos(np.clip(float(ref @ axis_in_body), -1.0, 1.0)))
    return f"{name:<8s} ({angle:4.1f} deg off) {np.round(axis_in_body, 2)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check HoMMI TCP / head frame conventions.")
    parser.add_argument("--cameras", action="store_true", help="show wrist RealSense images")
    args = parser.parse_args()

    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)
    lock = threading.Lock()
    latest: dict[str, object] = {}

    def keep(name: str):
        def handler(msg: object) -> None:
            with lock:
                latest[name] = msg

        return handler

    _subs = (
        build_sub("rt/lowstate", HGLowState_, keep("lowstate")),
        build_sub("rt/odom", Odom, keep("odom")),
    )
    unit = GripperWidthRange(0.0, 1.0)
    projector = RobotStateProjector({"left": unit, "right": unit}, head_extrinsic(config))

    cameras = {}
    if args.cameras:
        import cv2

        from g7_openarm_hommi.camera import RealSenseCamera

        serials = config.camera_serials()
        for side in ("left", "right"):
            cameras[side] = RealSenseCamera(
                serials[side], name=side, with_depth=False, fps=config.camera_fps
            )

    print("waiting for rt/lowstate and rt/odom ...")
    try:
        while True:
            with lock:
                lowstate, odom = latest.get("lowstate"), latest.get("odom")
            if lowstate is None or odom is None:
                time.sleep(0.2)
                continue
            snap = projector.snapshot(lowstate, odom, timestamp=time.monotonic())  # type: ignore[arg-type]
            q = odom.quaternion  # type: ignore[attr-defined]
            body_rot = quaternion_wxyz_to_matrix([q.w, q.x, q.y, q.z]).astype(np.float64)

            lines = []
            for name, frame in (
                ("left_tcp", snap.left_tcp),
                ("right_tcp", snap.right_tcp),
                ("head_optical", snap.head),
            ):
                rot = body_rot.T @ frame[:3, :3].astype(np.float64)
                lines.append(f"[{name}]")
                for i, axis in enumerate("xyz"):
                    lines.append(f"  +{axis} -> {describe(rot[:, i])}")
            print("\033[2J\033[H" + "\n".join(lines))
            print(
                "\nExpected: tcp +z = approach direction, +y = away from the wrist camera; "
                "head_optical +z = viewing direction, +y = image down."
            )

            for side, camera in cameras.items():
                image = cv2.cvtColor(camera.latest().color, cv2.COLOR_RGB2BGR)
                image = cv2.resize(image, (960, 540))
                cv2.putText(
                    image,
                    f"{side} wrist: fingers should be at the BOTTOM",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(f"{side} wrist", image)
            if cameras and cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for camera in cameras.values():
            camera.close()


if __name__ == "__main__":
    main()
