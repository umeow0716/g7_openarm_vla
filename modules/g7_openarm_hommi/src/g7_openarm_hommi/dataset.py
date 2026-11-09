"""Rebuild HoMMI ego3d training samples from a raw iPhUMI export.

An iPhUMI export holds one folder per phone (head / left / right) with, per
recording, ``<title>_<side>.json`` (ARKit poses + frame timestamps),
``*_rgb.mp4`` (60 Hz main camera), ``*_ultrawidergb.mp4`` (10 Hz, used for the
finger ArUco tags) and, for the head, ``*_depth.raw`` (256x192 float16 LiDAR).

The processing mirrors HoMMI's ``demonstration_processing`` +
``UmiTaskDataset`` so the observations / actions match what the policy saw in
training:

* all phones' poses resampled at 60 Hz on their common time window
  (pos + rotvec linear interpolation, ``sample_poses_at_times``);
* TCP = ARKit camera pose @ diag(1, -1, -1) with the iPhUMI TCP offset
  (no offset for the head, whose "TCP" is its OpenCV optical frame);
* gripper width from the LEFT finger tag in the ultrawide video
  (``get_gripper_width_offset(only_side='left')``), calibrated with the
  session's gripper-calibration run (``width - min``, clamped);
* wrist RGB black-padded to a square and INTER_AREA-resized; head pointmap
  from the depth with the capture intrinsics, zero-padded and
  INTER_NEAREST-resized, stored as float16;
* look-at point = median depth of the 50 rays closest to the optical axis.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from .geometry import matrix_to_rotation_6d_umi
from .trajectory import ACTION_DIM, ARM_SLICES, LOOKAT
from .vcam import IPHONE_DEPTH, depth_to_xyz_map

SIDES = ("head", "left", "right")
FPS = 60.0
DEPTH_SHAPE = (192, 256)
_NAME_RE = re.compile(
    r"^(?P<title>.+_(?P<kind>demonstration|grippercalibration))_(?P<side>head|left|right)$"
)

# hommi/common/contants.py
TCP_OFFSET_FROM_MAIN_CAMERA = np.array([0.0, -0.083, -0.231])
_ULTRAWIDE_OFFSET_FROM_CENTER_X = -0.0192 - TCP_OFFSET_FROM_MAIN_CAMERA[0]
_ULTRAWIDE_TO_FINGER_TAG_Z = 0.11796
_TAG_Z_TOLERANCE = 0.08
# calibration/iphone15pro_calibration.yaml (ultrawide intrinsics for tag detection)
_ULTRAWIDE_K = np.array([[257.5137, 0.0, 318.7089], [0.0, 257.5137, 239.9295], [0.0, 0.0, 1.0]])
# calibration/aruco_config.yaml
_TAG_SIZES = {
    **{i: 0.016 for i in (0, 1, 6, 7)},
    **{i: 0.06 for i in (2, 3, 4, 5, 8, 9, 10, 11)},
}
_TAG_DEFAULT_SIZE = 0.16
_TAGS_PER_GRIPPER = 6
_CALIB_DETECTION_THRESHOLD = 0.2


def parse_time(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


# ------------------------------------------------------------------ recordings


@dataclass(frozen=True)
class Recording:
    title: str
    kind: str  # "demonstration" | "grippercalibration"
    files: dict[str, Path]  # side -> json path

    def meta(self, side: str) -> dict:
        return json.loads(self.files[side].read_text())

    def media(self, side: str, suffix: str) -> Path:
        path = self.files[side].with_name(f"{self.files[side].stem}_{suffix}")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path


def discover(root: str | Path) -> dict[str, Recording]:
    """Group every ``<title>_<side>.json`` under ``root`` by recording title."""
    recordings: dict[str, Recording] = {}
    for path in sorted(Path(root).expanduser().rglob("*.json")):
        match = _NAME_RE.match(path.stem)
        if match is None:
            continue
        rec = recordings.setdefault(match["title"], Recording(match["title"], match["kind"], {}))
        rec.files[match["side"]] = path
    return recordings


def demonstrations(recordings: dict[str, Recording]) -> list[Recording]:
    """Complete (head + left + right) demonstrations, sorted by title (= time)."""
    return sorted(
        (
            r
            for r in recordings.values()
            if r.kind == "demonstration" and set(SIDES) <= r.files.keys()
        ),
        key=lambda r: r.title,
    )


def select_demonstration(recordings: dict[str, Recording], episode: str) -> Recording:
    """Pick a demonstration by index (``"0"``) or by a unique title substring."""
    demos = demonstrations(recordings)
    if not demos:
        raise ValueError("no complete head/left/right demonstrations found")
    if episode.lstrip("-").isdigit():
        return demos[int(episode)]
    matches = [d for d in demos if episode in d.title]
    if len(matches) != 1:
        names = "\n  ".join(f"[{i}] {d.title}" for i, d in enumerate(demos))
        raise ValueError(f"episode {episode!r} matches {len(matches)} demonstrations:\n  {names}")
    return matches[0]


# ------------------------------------------------------------------ images


def pad_square_resize(image: npt.NDArray, size: int, interpolation: int) -> npt.NDArray:
    """HoMMI ``get_image_transform_with_border``."""
    h, w = image.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    padded = cv2.copyMakeBorder(
        image, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=0
    )
    return cv2.resize(padded, (size, size), interpolation=interpolation)


def head_pointmap(depth: npt.NDArray[np.float16], size: int) -> npt.NDArray[np.float32]:
    """Replay-buffer head pointmap ([S, S, 3], float16-rounded, head optical frame)."""
    xyz = depth_to_xyz_map(depth.astype(np.float32), IPHONE_DEPTH).astype(np.float16)
    resized = pad_square_resize(xyz.astype(np.float32), size, cv2.INTER_NEAREST)
    return resized.astype(np.float16).astype(np.float32)


def lookat_depth(pointmap: npt.NDArray, top_k: int = 50) -> float:
    """HoMMI ``compute_center_lookat_from_pointmap`` (the point is ``[0, 0, z]``)."""
    flat = pointmap.reshape(-1, 3)
    valid = (flat[:, 2] > 0) & np.isfinite(flat).all(axis=1)
    if not np.any(valid):
        return 0.0
    pts = flat[valid]
    dist = pts[:, 0] ** 2 + pts[:, 1] ** 2
    k = min(top_k, len(pts))
    idx = np.argpartition(dist, k - 1)[:k]
    return float(np.float16(np.median(pts[idx, 2])))


def decode_frames(path: Path, wanted: set[int]) -> dict[int, npt.NDArray[np.uint8]]:
    """Decode the given frame indices (RGB uint8) in one sequential pass."""
    capture = cv2.VideoCapture(str(path))
    frames: dict[int, npt.NDArray[np.uint8]] = {}
    last = max(wanted) if wanted else -1
    index = 0
    try:
        while index <= last:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        capture.release()
    missing = wanted - frames.keys()
    if missing:
        raise RuntimeError(f"{path.name}: could not decode frames {sorted(missing)[:5]}")
    return frames


# ------------------------------------------------------------------ gripper width


def _mirror_mask_polygons(shape: tuple[int, int]) -> list[npt.NDArray[np.int32]]:
    """UMI ``get_mirror_canonical_polygon`` mapped to an image of ``shape``."""
    left = np.array(
        [[540, 1700], [680, 1450], [590, 1070], [290, 1130], [290, 1770], [550, 1770]], float
    )
    canonical = (left - np.array([2704, 2028]) * 0.5) / 2028
    return [
        np.round(canonical * np.array([sign, 1.0]) * shape[0] + np.array(shape[::-1]) * 0.5).astype(
            np.int32
        )
        for sign in (1.0, -1.0)
    ]


def _ultrawide_time_offset(meta: dict) -> float:
    # The first ultrawide frame lands on main-camera frame i (60 Hz).
    first = next(i for i, t in enumerate(meta["ultrawideRGBTimes"]) if t)
    return first / 60.0


def detect_tags(rec: Recording, side: str) -> list[tuple[float, dict[int, npt.NDArray]]]:
    """HoMMI ``detect_aruco_iphone.py``: ``[(time, {tag_id: tvec})]`` per ultrawide frame."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params
    )
    offset = _ultrawide_time_offset(rec.meta(side))
    capture = cv2.VideoCapture(str(rec.media(side, "ultrawidergb.mp4")))
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    results: list[tuple[float, dict[int, npt.NDArray]]] = []
    polys = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if polys is None:
                polys = _mirror_mask_polygons(img.shape[:2])
            cv2.fillPoly(img, polys, color=(0, 0, 0))
            corners, ids, _ = detector.detectMarkers(img)
            tags: dict[int, npt.NDArray] = {}
            if ids is not None:
                for tag_id, tag_corners in zip(ids.flatten().tolist(), corners, strict=True):
                    s = _TAG_SIZES.get(tag_id, _TAG_DEFAULT_SIZE)
                    obj = np.array(
                        [
                            [-s / 2, s / 2, 0],
                            [s / 2, s / 2, 0],
                            [s / 2, -s / 2, 0],
                            [-s / 2, -s / 2, 0],
                        ]
                    )
                    ok_pnp, _rvec, tvec = cv2.solvePnP(
                        obj,
                        tag_corners.reshape(4, 2).astype(np.float64),
                        _ULTRAWIDE_K,
                        np.zeros(5),
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
                    if ok_pnp:
                        tags[tag_id] = tvec.reshape(3)
            results.append((len(results) / fps + offset, tags))
    finally:
        capture.release()
    return results


def _tag_width(tags: dict[int, npt.NDArray], left_id: int) -> float | None:
    tvec = tags.get(left_id)
    if tvec is None or not (
        _ULTRAWIDE_TO_FINGER_TAG_Z - _TAG_Z_TOLERANCE
        < tvec[2]
        < _ULTRAWIDE_TO_FINGER_TAG_Z + _TAG_Z_TOLERANCE
    ):
        return None
    return abs(tvec[0] + _ULTRAWIDE_OFFSET_FROM_CENTER_X) * 2.0


@dataclass(frozen=True)
class GripperCalibration:
    gripper_id: int
    detection_rate: float
    left_finger_tag_id: int
    min_width: float
    max_width: float


def calibrate_gripper(rec: Recording, side: str) -> GripperCalibration:
    detections = detect_tags(rec, side)
    counts: dict[int, int] = defaultdict(int)
    for _, tags in detections:
        for tag_id in tags:
            counts[tag_id] += 1
    if not counts:
        raise RuntimeError(f"{rec.title} {side}: no ArUco tags detected")
    stats = {k: v / len(detections) for k, v in counts.items()}
    best: tuple[int, float] | None = None
    for gripper_id in range(max(stats) // _TAGS_PER_GRIPPER + 1):
        left_id = gripper_id * _TAGS_PER_GRIPPER
        prob = min(stats.get(left_id, 0.0), stats.get(left_id + 1, 0.0))
        if prob > 0 and (best is None or prob > best[1]):
            best = (gripper_id, prob)
    if best is None or best[1] < _CALIB_DETECTION_THRESHOLD:
        raise RuntimeError(f"{rec.title} {side}: gripper tags not detected reliably ({best})")
    left_id = best[0] * _TAGS_PER_GRIPPER
    widths = np.array(
        [w if (w := _tag_width(tags, left_id)) is not None else np.nan for _, tags in detections]
    )
    return GripperCalibration(
        best[0], best[1], left_id, float(np.nanmin(widths)), float(np.nanmax(widths))
    )


def gripper_widths(
    rec: Recording, side: str, calib: GripperCalibration, aligned_times: npt.NDArray
) -> tuple[npt.NDArray[np.float32], float]:
    """Calibrated width per aligned frame, plus the tag detection rate."""
    detections = detect_tags(rec, side)
    span = calib.max_width - calib.min_width
    t, w = [], []
    for time, tags in detections:
        width = _tag_width(tags, calib.left_finger_tag_id)
        if width is not None:
            t.append(time)
            w.append(float(np.clip(width - calib.min_width, 0.0, span)))
    if len(t) < 2:
        raise RuntimeError(f"{rec.title} {side}: too few gripper tag detections")
    # HoMMI queries with the aligned times relative to the aligned start.
    widths = np.interp(aligned_times - aligned_times[0], t, w).astype(np.float32)
    return widths, len(t) / len(detections)


# ------------------------------------------------------------------ poses


def _rotvec(matrix: npt.NDArray) -> npt.NDArray:
    return cv2.Rodrigues(np.ascontiguousarray(matrix, dtype=np.float64))[0].reshape(3)


def resample_poses(poses: npt.NDArray, times: npt.NDArray, samples: npt.NDArray) -> npt.NDArray:
    """HoMMI ``sample_poses_at_times``: per-component linear interp of [pos, rotvec]."""
    six = np.concatenate([poses[:, :3, 3], np.stack([_rotvec(p[:3, :3]) for p in poses])], axis=1)
    out6 = np.stack([np.interp(samples, times, six[:, i]) for i in range(6)], axis=1)
    out = np.tile(np.eye(4), (len(samples), 1, 1))
    out[:, :3, 3] = out6[:, :3]
    out[:, :3, :3] = np.stack([cv2.Rodrigues(r)[0] for r in out6[:, 3:]])
    return out


def iphone_to_tcp(poses: npt.NDArray, *, with_offset: bool) -> npt.NDArray:
    """HoMMI ``iphone_to_tcp_poses`` (ARKit camera -> TCP / OpenCV optical frame)."""
    i_t_tcp = np.eye(4)
    i_t_tcp[:3, :3] = np.diag([1.0, -1.0, -1.0])
    if with_offset:
        i_t_tcp[:3, 3] = TCP_OFFSET_FROM_MAIN_CAMERA
    return poses @ i_t_tcp


def _frame_index(times: list[str], aligned: npt.NDArray) -> npt.NDArray[np.int64]:
    stamps = np.array([parse_time(t) if t else 0.0 for t in times])
    return np.searchsorted(stamps, aligned, side="right") - 1


# ------------------------------------------------------------------ episode


@dataclass
class Episode:
    """One demonstration resampled on the 60 Hz aligned timeline."""

    recording: Recording
    times: npt.NDArray[np.float64]  # [T] aligned timestamps (s)
    tcp: dict[str, npt.NDArray[np.float64]]  # side -> [T, 4, 4] world poses
    width: dict[str, npt.NDArray[np.float32]]  # left/right -> [T] meters
    detection_rate: dict[str, float]
    main_idx: dict[str, npt.NDArray[np.int64]]  # side -> [T] rgb video frame
    depth_idx: npt.NDArray[np.int64]  # [T] head depth frame
    depth: npt.NDArray[np.float16]  # [N, 192, 256]
    lookat_z: npt.NDArray[np.float32]  # [T] look-at depth in the head frame

    @property
    def name(self) -> str:
        return self.recording.title

    @property
    def length(self) -> int:
        return len(self.times)

    def pointmap(self, index: int, size: int) -> npt.NDArray[np.float32]:
        return head_pointmap(self.depth[self.depth_idx[index]], size)

    def lookat_world(self, index: int) -> npt.NDArray[np.float64]:
        return (self.tcp["head"][index] @ np.array([0.0, 0.0, self.lookat_z[index], 1.0]))[:3]

    def load_rgb(
        self, indices: set[int], sizes: dict[str, int]
    ) -> dict[str, dict[int, npt.NDArray[np.uint8]]]:
        """Training-resolution RGB for the given aligned frames: side -> {frame: image}."""
        out: dict[str, dict[int, npt.NDArray[np.uint8]]] = {}
        for side in SIDES:
            wanted = {int(self.main_idx[side][i]) for i in indices}
            frames = decode_frames(self.recording.media(side, "rgb.mp4"), wanted)
            resized = {
                k: pad_square_resize(v, sizes[side], cv2.INTER_AREA) for k, v in frames.items()
            }
            out[side] = {i: resized[int(self.main_idx[side][i])] for i in indices}
        return out

    def gt_action(self, cur: int, horizon: int, step: int) -> npt.NDArray[np.float32]:
        """Ground-truth ``[horizon, 23]`` chunk relative to the left TCP at ``cur``.

        Steps run ``cur, cur + step, ...`` and are clamped to the episode end.
        """
        base_inv = np.linalg.inv(self.tcp["left"][cur])
        out = np.zeros((horizon, ACTION_DIM), dtype=np.float32)
        for k in range(horizon):
            t = min(cur + k * step, self.length - 1)
            for side, (pos, rot, w) in ARM_SLICES.items():
                rel = base_inv @ self.tcp[side][t]
                out[k, pos] = rel[:3, 3]
                out[k, rot] = matrix_to_rotation_6d_umi(rel[:3, :3])
                out[k, w] = self.width[side][t]
            out[k, LOOKAT] = base_inv[:3, :3] @ self.lookat_world(t) + base_inv[:3, 3]
        return out


def load_episode(
    rec: Recording,
    recordings: dict[str, Recording],
    *,
    head_size: int,
    calibrations: dict[tuple[str, str], GripperCalibration] | None = None,
    log=print,
) -> Episode:
    calibrations = {} if calibrations is None else calibrations
    metas = {side: rec.meta(side) for side in SIDES}
    pose_times = {s: np.array([parse_time(t) for t in metas[s]["poseTimes"]]) for s in SIDES}
    start = max(t[0] for t in pose_times.values())
    end = min(t[-1] for t in pose_times.values())
    aligned = np.arange(start, end, 1.0 / FPS)

    tcp, width, rate, main_idx = {}, {}, {}, {}
    for side in SIDES:
        poses = resample_poses(np.array(metas[side]["poseTransforms"]), pose_times[side], aligned)
        tcp[side] = iphone_to_tcp(poses, with_offset=side != "head")
        main_idx[side] = _frame_index(metas[side]["rgbTimes"], aligned)
        if main_idx[side].min() < 0:
            raise RuntimeError(f"{rec.title}: {side} RGB starts after the aligned window")
        if side == "head":
            continue
        calib_title = metas[side]["gripperCalibrationRunName"].rsplit("_", 1)[0]
        if calib_title not in recordings:
            raise FileNotFoundError(f"gripper calibration {calib_title!r} is not in the export")
        key = (calib_title, side)
        if key not in calibrations:
            calibrations[key] = calibrate_gripper(recordings[calib_title], side)
            log(f"gripper calibration {side}: {calibrations[key]}")
        width[side], rate[side] = gripper_widths(rec, side, calibrations[key], aligned)

    depth = np.fromfile(rec.media("head", "depth.raw"), dtype=np.float16).reshape(-1, *DEPTH_SHAPE)
    depth_idx = _frame_index(metas["head"]["depthTimes"], aligned)
    if depth_idx.min() < 0 or depth_idx.max() >= len(depth):
        raise RuntimeError(f"{rec.title}: depth frames do not cover the aligned window")
    cache: dict[int, float] = {}
    lookat_z = np.empty(len(aligned), dtype=np.float32)
    for i, d in enumerate(depth_idx.tolist()):
        if d not in cache:
            cache[d] = lookat_depth(head_pointmap(depth[d], head_size))
        lookat_z[i] = cache[d]
    return Episode(rec, aligned, tcp, width, rate, main_idx, depth_idx, depth, lookat_z)


def obs_indices(cur: int, horizon: int, step: int) -> list[int]:
    """UMI sampler observation frames for ``cur`` (padded with the first frame)."""
    idx = [i for i in (cur - k * step for k in range(horizon - 1, -1, -1)) if i >= 0]
    return [idx[0]] * (horizon - len(idx)) + idx
