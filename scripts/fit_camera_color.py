"""Fit the head + wrist camera color correction against the HoMMI training dataset.

Point the cameras at a scene like the demonstrations (white table, green towel,
red fingers in the wrist views), stop hommi so the cameras are free, then:

    uv run python scripts/fit_camera_color.py ~/Downloads/1014-demo-cropped
    uv run python scripts/fit_camera_color.py DATASET --cameras head   # just one

All cameras are first captured at every fixed white-balance temperature (about
10 s each, the only time the scene has to be held). Then, offline, a saturation
plus per-channel ``gain`` / ``gamma`` is fitted at each temperature on the
reference colors (white table, red fingers, green towel) seen in both, and the
temperature with the most reference colors and smallest residual is kept.
Results go to the hommi package's ``calibration/camera_color.json`` (keyed
by RealSense serial, or ``zed2``); ``--preview`` gets a raw | corrected | dataset
image per camera.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from g7_openarm_hommi.camera import COLOR_HEIGHT, COLOR_WIDTH, PinholeIntrinsics
from g7_openarm_hommi.color import CALIBRATION_PATH, ZED_KEY, ColorCalibration, ColorCorrector
from g7_openarm_hommi.config import config
from g7_openarm_hommi.vcam import resample_rgb_to_virtual_square
from g7_openarm_hommi.zed_camera import CAPTURE_SIZE, find_zed2

WB_CANDIDATES = tuple(range(2800, 6501, 300))
MIN_PIXELS = 500
SIZE = 224


def reference_masks(images: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Valid pixels and their white / red / green class masks (padding, clipping dropped)."""
    px = images.reshape(-1, 3)
    px = px[(px.astype(int).sum(1) > 0) & (px.max(1) < 250)]
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(int)
    h, s, v = hsv.T
    luma = px.mean(1)
    masks = {
        "white": (s < 40) & (luma > np.percentile(luma, 80)),
        "red": ((h < 8) | (h > 170)) & (s > 140) & (v > 80),
        "green": (h > 25) & (h < 90) & (s > 20) & (v > 60),
    }
    return px, {k: m for k, m in masks.items() if m.sum() >= MIN_PIXELS}


def reference_colors(images: np.ndarray) -> dict[str, np.ndarray]:
    """Mean RGB of each reference class present in ``images``."""
    px, masks = reference_masks(images)
    return {k: px[m].mean(0) for k, m in masks.items()}


def dataset_frames(root: Path, side: str, per_video: int = 12) -> np.ndarray:
    frames = []
    videos = sorted(root.rglob(f"*_{side}_rgb.mp4"))
    if not videos:
        raise FileNotFoundError(f"no *_{side}_rgb.mp4 under {root}")
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in np.linspace(0, n - 1, per_video).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                frame = cv2.resize(frame, (SIZE * 4 // 3, SIZE), interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    return np.stack(frames)


class RealSense:
    def __init__(self, serial: str) -> None:
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.rgb8, 30)
        self.pipeline = rs.pipeline()
        profile = self.pipeline.start(cfg)
        self.sensor = profile.get_device().first_color_sensor()
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = PinholeIntrinsics.from_rs(stream.get_intrinsics())

    def grab(self, white_balance: float, n: int = 8, settle: int = 45) -> np.ndarray:
        self.sensor.set_option(rs.option.enable_auto_white_balance, 0)
        self.sensor.set_option(rs.option.white_balance, white_balance)
        for _ in range(settle):
            self.pipeline.wait_for_frames()
        out = []
        for _ in range(n):
            color = self.pipeline.wait_for_frames().get_color_frame()
            img = np.asanyarray(color.get_data()).copy()
            out.append(resample_rgb_to_virtual_square(img, self.intrinsics, output_size=SIZE))
        return np.stack(out)

    def close(self) -> None:
        self.sensor.set_option(rs.option.enable_auto_white_balance, 1)
        self.pipeline.stop()


class Zed:
    """Left eye of the ZED 2 side-by-side UVC stream (color only, no rectification)."""

    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(find_zed2(), cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * CAPTURE_SIZE[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
        if not self.cap.isOpened():
            raise RuntimeError("cannot open the ZED 2")

    def grab(self, white_balance: float, n: int = 8, settle: int = 45) -> np.ndarray:
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, white_balance)
        for _ in range(settle):
            self.cap.read()
        out = []
        for _ in range(n):
            ok, sbs = self.cap.read()
            if not ok:
                raise RuntimeError("ZED 2 frame read failed")
            left = cv2.resize(
                sbs[:, : sbs.shape[1] // 2], (SIZE * 16 // 9, SIZE), interpolation=cv2.INTER_AREA
            )
            out.append(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
        return np.stack(out)

    def close(self) -> None:
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        self.cap.release()


def open_cameras(names: list[str]) -> dict[str, tuple[str, object]]:
    """name -> (calibration key, opener)."""
    cams = {}
    for name in names:
        if name == "head" and config.head_camera == "zed2":
            cams[name] = (ZED_KEY, Zed)
        else:
            serial = {
                "head": config.head_camera_serial,
                "left": config.left_wrist_camera_serial,
                "right": config.right_wrist_camera_serial,
            }[name]
            cams[name] = (serial, lambda s=serial: RealSense(s))
    return cams


def preview(image: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(image.reshape(image.shape[0], -1).max(1) > 0)  # drop padding
    return cv2.resize(image[rows[0] : rows[-1] + 1], (SIZE * 4 // 3, SIZE))


def fit_correction(frames: np.ndarray, target: dict, wb: float) -> tuple[ColorCalibration, list]:
    """Grid search saturation, then per channel gain/gamma, matching corrected class means.

    Class masks are fixed on the raw frames; for each saturation the transformed
    pixels of each class are reduced to 256-bin histograms per channel, so the
    corrected mean of every candidate curve is exact.
    """
    px, masks = reference_masks(frames)
    keys = [k for k in ("white", "red", "green") if k in masks and k in target]
    if len(keys) < 2:
        raise RuntimeError(f"need >=2 reference colors in view, found {keys}")
    gain_grid = np.linspace(0.7, 1.6, 181)
    gamma_grid = np.linspace(0.5, 1.6, 221)
    x = np.arange(256) / 255.0
    # curves[g, k, i]: corrected value of input i for gain g, gamma k
    curves = np.clip(gain_grid[:, None, None] * x, 0.0, 1.0) ** gamma_grid[None, :, None] * 255.0
    best = None
    for saturation in np.linspace(0.8, 2.0, 25):
        probe = ColorCalibration(wb, (1.0,) * 3, (1.0,) * 3, float(saturation))
        sat = cv2.transform(px.reshape(-1, 1, 3), probe.saturation_matrix()).reshape(-1, 3)
        total, gains, gammas = 0.0, [], []
        for c in range(3):
            err = np.zeros(curves.shape[:2])
            for k in keys:
                hist = np.bincount(sat[masks[k], c], minlength=256).astype(float)
                err += (curves @ (hist / hist.sum()) - target[k][c]) ** 2
            g, k = np.unravel_index(np.argmin(err), err.shape)
            total += float(err[g, k])
            gains.append(float(gain_grid[g]))
            gammas.append(float(gamma_grid[k]))
        if best is None or total < best[0]:
            best = (total, ColorCalibration(wb, tuple(gains), tuple(gammas), float(saturation)))
    return best[1], keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dataset", type=Path, help="unzipped HoMMI dataset root")
    parser.add_argument("--out", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--preview", type=Path, default=Path("camera_color_preview.jpg"))
    parser.add_argument(
        "--cameras", nargs="+", default=["head", "left", "right"], choices=["head", "left", "right"]
    )
    args = parser.parse_args()

    cameras = open_cameras(args.cameras)
    # Capture everything first (about 10 s per camera) so the scene only has to be
    # held for the shots; all fitting below runs offline on the stored frames.
    shots: dict[str, dict[int, np.ndarray]] = {}
    for side, (_, opener) in cameras.items():
        cam = opener()
        try:
            cam.grab(WB_CANDIDATES[0], n=1, settle=30)  # let auto exposure settle
            shots[side] = {wb: cam.grab(wb, settle=12) for wb in WB_CANDIDATES}
        finally:
            cam.close()
        print(f"[{side}] captured {len(WB_CANDIDATES)} white balance settings")
    print("capture done, the cameras can move now; fitting...")

    result = json.loads(args.out.read_text()) if args.out.exists() else {}
    rows = []
    for side, (key, _) in cameras.items():
        ds = dataset_frames(args.dataset, side)
        target = reference_colors(ds)
        print(f"[{side}] dataset refs: { {k: v.round(1).tolist() for k, v in target.items()} }")
        best = None
        for wb, frames in shots[side].items():
            try:
                calib, keys = fit_correction(frames, target, float(wb))
            except RuntimeError:
                continue
            correct = ColorCorrector(calib)
            px, masks = reference_masks(frames)  # same pixels as the fit
            fixed = correct(px.reshape(-1, 1, 3)).reshape(-1, 3)
            rms = float(
                np.sqrt(np.mean([(fixed[masks[k]].mean(0) - target[k]) ** 2 for k in keys]))
            )
            # Prefer more reference colors, then the smallest residual.
            score = (-len(keys), rms)
            print(f"[{side}]   WB {wb}K refs {keys} rms {rms:.2f}")
            if best is None or score < best[0]:
                best = (score, wb, calib, keys, frames, fixed, masks)
        if best is None:
            raise RuntimeError(f"{side}: fewer than 2 reference colors at every white balance")
        _, wb, calib, keys, frames, fixed, masks = best

        correct = ColorCorrector(calib)
        live = reference_colors(frames)
        after = {k: fixed[m].mean(0) for k, m in masks.items()}
        print(
            f"[{side}] WB {wb}K saturation {calib.saturation:.2f} "
            f"gain {np.round(calib.gain, 3)} gamma {np.round(calib.gamma, 3)}"
        )
        for k in keys:
            print(
                f"[{side}]   {k:5s} dataset {target[k].round(0)}  "
                f"raw {live[k].round(0)}  corrected {after[k].round(0)}"
            )
        result[key] = {"camera": side, **calib.to_json()}
        rows.append(
            np.concatenate(
                [preview(frames[-1]), preview(correct(frames[-1])), preview(ds[len(ds) // 2])], 1
            )
        )

    args.out.write_text(json.dumps(result, indent=2) + "\n")
    cv2.imwrite(str(args.preview), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    print(f"wrote {args.out} and {args.preview} (columns: raw | corrected | dataset)")


if __name__ == "__main__":
    main()
