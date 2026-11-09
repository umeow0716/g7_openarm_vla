"""Operator window: what the ego3d policy sees.

Top half: the head point cloud fed to the model, seen from the head camera
(first-person), after cropping (depth range, points behind the grippers);
points are coloured by depth, cropped points are black. Bottom half: the
left / right wrist images exactly as the encoder gets them (virtual iPhone
camera, eval center crop).
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import numpy.typing as npt

WINDOW_NAME = "HoMMI observation"
PANEL = 512  # px, side of each square panel


def display_available() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _label(img: npt.NDArray[np.uint8], text: str, row: int = 0) -> None:
    org = (10, 26 + 24 * row)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def render_pointmap(
    pointmap: npt.NDArray[np.float32], mask: npt.NDArray[np.bool_]
) -> npt.NDArray[np.uint8]:
    """First-person depth-coloured image of the kept points (BGR, PANEL x PANEL)."""
    depth = pointmap[..., 2]
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if mask.any():
        valid = depth[mask]
        lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
        scaled = np.clip((depth - lo) / max(hi - lo, 1e-3), 0.0, 1.0)
        colored = cv2.applyColorMap((255 * (1.0 - scaled)).astype(np.uint8), cv2.COLORMAP_TURBO)
        out[mask] = colored[mask]
    return cv2.resize(out, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)


def render_wrist(rgb: npt.NDArray[np.uint8], crop_ratio: float) -> npt.NDArray[np.uint8]:
    """Wrist image after the encoder's eval center crop (BGR, PANEL x PANEL)."""
    size = rgb.shape[0]
    crop = int(size * crop_ratio)
    start = (size - crop) // 2
    cropped = rgb[start : start + crop, start : start + crop]
    return cv2.cvtColor(
        cv2.resize(cropped, (PANEL, PANEL), interpolation=cv2.INTER_LINEAR), cv2.COLOR_RGB2BGR
    )


def compose(
    pointmap: npt.NDArray[np.float32],
    mask: npt.NDArray[np.bool_],
    left_rgb: npt.NDArray[np.uint8],
    right_rgb: npt.NDArray[np.uint8],
    *,
    crop_ratio: float,
    status: str,
) -> npt.NDArray[np.uint8]:
    """2*PANEL x 2*PANEL canvas: point cloud on top, left | right wrist below."""
    canvas = np.zeros((2 * PANEL, 2 * PANEL, 3), dtype=np.uint8)
    top = render_pointmap(pointmap, mask)
    canvas[:PANEL, PANEL // 2 : PANEL // 2 + PANEL] = top
    canvas[PANEL:, :PANEL] = render_wrist(left_rgb, crop_ratio)
    canvas[PANEL:, PANEL:] = render_wrist(right_rgb, crop_ratio)
    cv2.line(canvas, (0, PANEL), (2 * PANEL, PANEL), (80, 80, 80), 1)
    cv2.line(canvas, (PANEL, PANEL), (PANEL, 2 * PANEL), (80, 80, 80), 1)

    head = canvas[:PANEL]
    _label(head, f"head point cloud (model input, cropped) - {100 * mask.mean():.0f}% points kept")
    _label(head, f"HoMMI: {status}", row=1)
    if not mask.any():
        _label(head, "no valid points", row=2)
    _label(canvas[PANEL:, :PANEL], "left wrist (model input)")
    _label(canvas[PANEL:, PANEL:], "right wrist (model input)")
    return canvas


class ObservationWindow:
    """OpenCV window driven from the main thread (Qt requires it)."""

    def __init__(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, PANEL * 2, PANEL * 2)

    def show(self, canvas: npt.NDArray[np.uint8] | None, wait_ms: int) -> None:
        if canvas is not None:
            cv2.imshow(WINDOW_NAME, canvas)
        cv2.waitKey(max(1, wait_ms))

    def close(self) -> None:
        cv2.destroyAllWindows()
