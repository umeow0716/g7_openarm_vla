"""Per-camera color correction that maps robot camera colors onto the iPhone look.

The policy was trained on iPhone 15 Pro footage (auto white balance + Apple tone
mapping); the D435i wrist cameras and the ZED 2 head camera are darker and their
auto white balance drifts with the scene. A camera with a calibration entry
therefore runs with auto white balance off at a fixed color temperature, and the
color image the policy sees goes through a saturation matrix (scale chroma about
Rec.601 luma) and a per-channel ``out = 255 * (gain * in / 255) ** gamma`` lookup
table, fitted by
``scripts/fit_camera_color.py`` against the training dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

CALIBRATION_PATH = Path(__file__).with_name("calibration") / "camera_color.json"
ZED_KEY = "zed2"  # calibration key of the ZED 2 head camera (RealSense uses its serial)


@dataclass(frozen=True, slots=True)
class ColorCalibration:
    white_balance: float  # Kelvin, auto white balance disabled
    gain: tuple[float, float, float]  # RGB
    gamma: tuple[float, float, float]  # RGB
    saturation: float = 1.0  # chroma scale about luma, applied before the LUT

    def lut(self) -> npt.NDArray[np.uint8]:
        """[256, 1, 3] uint8 table for ``cv2.LUT`` on RGB images (0 stays 0)."""
        x = np.arange(256, dtype=np.float64)[:, None] / 255.0
        y = np.clip(x * np.asarray(self.gain), 0.0, 1.0) ** np.asarray(self.gamma)
        return np.round(y * 255.0).astype(np.uint8).reshape(256, 1, 3)

    def saturation_matrix(self) -> npt.NDArray[np.float32]:
        """3x3 RGB matrix for ``cv2.transform``: ``luma + saturation * (rgb - luma)``."""
        luma = np.array([[0.299, 0.587, 0.114]], dtype=np.float64)
        s = self.saturation
        return (s * np.eye(3) + (1.0 - s) * np.ones((3, 1)) @ luma).astype(np.float32)

    def to_json(self) -> dict:
        return {
            "white_balance": self.white_balance,
            "saturation": round(self.saturation, 4),
            "gain": [round(g, 4) for g in self.gain],
            "gamma": [round(g, 4) for g in self.gamma],
        }

    @classmethod
    def from_json(cls, data: dict) -> ColorCalibration:
        gain = tuple(float(v) for v in data["gain"])
        gamma = tuple(float(v) for v in data["gamma"])
        if len(gain) != 3 or len(gamma) != 3:
            raise ValueError(f"color calibration needs 3 gains and gammas, got {data}")
        saturation = float(data.get("saturation", 1.0))
        return cls(float(data["white_balance"]), gain, gamma, saturation)  # type: ignore[arg-type]


def load_calibrations(path: Path = CALIBRATION_PATH) -> dict[str, ColorCalibration]:
    """Camera key -> calibration; an absent file means no camera is corrected."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {serial: ColorCalibration.from_json(entry) for serial, entry in data.items()}


class ColorCorrector:
    """Applies one calibration to RGB uint8 images (black stays black)."""

    def __init__(self, calibration: ColorCalibration) -> None:
        self.calibration = calibration
        self._lut = calibration.lut()
        self._matrix = None if calibration.saturation == 1.0 else calibration.saturation_matrix()

    def __call__(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        if self._matrix is not None:
            image = cv2.transform(image, self._matrix)  # saturates to [0, 255]
        return cv2.LUT(image, self._lut)
