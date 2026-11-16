"""Live preview of the ZED 2 head camera: rectified left image next to its depth.

    uv run python scripts/view_zed_depth.py

Runs the same ``ZedStereoCamera`` the HoMMI node uses, so it needs the Fast-FoundationStereo
engine from ``scripts/build_ffs_engine.py``. Over SSH, point it at the machine's own
screen with ``DISPLAY=:0``. Press q or Esc to quit.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import numpy.typing as npt

from g7_openarm_hommi.zed_camera import ZedStereoCamera

WINDOW = "ZED 2 depth (q / Esc to quit)"


def colorize(depth: npt.NDArray[np.float32], near: float, far: float) -> npt.NDArray[np.uint8]:
    """Turbo colour map, linear in inverse depth so near objects keep their contrast."""
    inv = (1.0 / np.maximum(depth, 1e-6) - 1.0 / far) / (1.0 / near - 1.0 / far)
    out = cv2.applyColorMap((np.clip(inv, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[depth <= 0] = 0
    return np.asarray(out, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--near", type=float, default=0.3, help="depth at the red end [m]")
    parser.add_argument("--far", type=float, default=5.0, help="depth at the blue end [m]")
    args = parser.parse_args()

    camera = ZedStereoCamera()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1800, 540)
    last_timestamp = None
    fps = 0.0
    try:
        while True:
            frame = camera.latest()
            # latest() repeats a frame until the stereo model finishes the next one.
            if frame.timestamp != last_timestamp:
                assert frame.depth is not None
                if last_timestamp is not None:
                    rate = 1.0 / (frame.timestamp - last_timestamp)
                    fps = 0.9 * fps + 0.1 * rate if fps else rate
                last_timestamp = frame.timestamp

                depth = frame.depth
                h, w = depth.shape
                patch = depth[h // 2 - 4 : h // 2 + 5, w // 2 - 4 : w // 2 + 5]
                centre = float(np.median(patch[patch > 0])) if (patch > 0).any() else 0.0
                view = np.hstack(
                    [
                        cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR),
                        colorize(depth, args.near, args.far),
                    ]
                )
                for x in (w // 2, w + w // 2):
                    cv2.drawMarker(view, (x, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
                text = f"{fps:4.1f} FPS   centre {centre:.2f} m   colour {args.near}-{args.far} m"
                cv2.putText(view, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
                cv2.putText(view, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.imshow(WINDOW, view)
            key = cv2.waitKey(5) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
