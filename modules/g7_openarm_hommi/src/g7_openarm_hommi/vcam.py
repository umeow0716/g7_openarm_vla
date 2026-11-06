"""Resample RealSense frames into the iPhone camera geometry used for training.

The ego3d HoMMI policy was trained on iPhUMI data (iPhone 15 Pro). Its replay
buffer stores:

* RGB from the iPhone main ("wide") camera, 1920x1440 (4:3), black-padded to a
  square and INTER_AREA-resized (224 for wrists, 512 for the head);
* the head pointmap from the 256x192 LiDAR depth (registered to the wide camera),
  unprojected with ``depth2xyzmap`` (OpenCV optical frame), zero-padded to a
  square and INTER_NEAREST-resized to 512.

Feeding raw 16:9 D435i images would change the field of view and the padding
layout the ViT saw during training. Instead every RealSense frame is mapped onto
a *virtual* iPhone camera with the training intrinsics: pixels outside the
RealSense field of view become black / invalid, everything else keeps the same
angular scale as in training. Intrinsics come from HoMMI's
``calibration/iphone15pro_capture_calibration.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from .camera import CameraFrame, PinholeIntrinsics

FloatArray = npt.NDArray[np.float32]

IPHONE_WIDE = PinholeIntrinsics(1920, 1440, 1303.1525, 1303.1525, 953.19354, 723.9154)
IPHONE_DEPTH = PinholeIntrinsics(256, 192, 173.8, 173.8, 127.1, 96.5)


def resample_rgb_to_virtual_square(
    image: npt.NDArray[np.uint8],
    source: PinholeIntrinsics,
    *,
    output_size: int,
    target: PinholeIntrinsics = IPHONE_WIDE,
) -> npt.NDArray[np.uint8]:
    """Map ``image`` onto the black-padded, resized virtual camera square.

    Equivalent to: remap ``image`` into ``target`` intrinsics, pad to a square
    with black borders, and INTER_AREA-resize to ``output_size``. The composite
    mapping is a per-axis scale + shift, so it is computed as one INTER_AREA
    resize (anti-aliasing at almost the final scale) followed by a near-identity
    ``warpAffine`` that applies the exact residual scale and sub-pixel shift.
    All coordinates use OpenCV's pixel-centre convention.
    """
    if image.shape[:2] != (source.height, source.width):
        raise ValueError(
            f"image shape {image.shape[:2]} does not match intrinsics "
            f"{source.height}x{source.width}"
        )
    side = max(target.width, target.height)
    pad_x = (side - target.width) // 2
    pad_y = (side - target.height) // 2
    out_per_side = output_size / side

    # Exact map from a source pixel index to an output pixel index:
    #   virtual  u_t = cx_t + fx_t * (u_s - cx_s) / fx_s
    #   output   U   = (u_t + pad + 0.5) * output_size / side - 0.5
    scale_x = out_per_side * target.fx / source.fx
    scale_y = out_per_side * target.fy / source.fy
    offset_x = out_per_side * (pad_x + target.cx - target.fx * source.cx / source.fx + 0.5) - 0.5
    offset_y = out_per_side * (pad_y + target.cy - target.fy * source.cy / source.fy + 0.5) - 0.5

    resized_w = max(1, round(source.width * scale_x))
    resized_h = max(1, round(source.height * scale_y))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    # resized pixel r <-> source pixel u_s = (r + 0.5) * W / resized_w - 0.5
    rx = source.width / resized_w
    ry = source.height / resized_h
    warp = np.array(
        [
            [scale_x * rx, 0.0, scale_x * (0.5 * rx - 0.5) + offset_x],
            [0.0, scale_y * ry, scale_y * (0.5 * ry - 0.5) + offset_y],
        ],
        dtype=np.float64,
    )
    canvas = cv2.warpAffine(
        resized,
        warp,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Anything outside the virtual camera's own frame is padding in training.
    valid_x0 = round(pad_x * out_per_side)
    valid_x1 = round((pad_x + target.width) * out_per_side)
    valid_y0 = round(pad_y * out_per_side)
    valid_y1 = round((pad_y + target.height) * out_per_side)
    canvas[:valid_y0] = 0
    canvas[valid_y1:] = 0
    canvas[:, :valid_x0] = 0
    canvas[:, valid_x1:] = 0
    return canvas


def splat_depth_to_virtual(
    depth: FloatArray,
    depth_intrinsics: PinholeIntrinsics,
    depth_to_color: npt.ArrayLike,
    *,
    target: PinholeIntrinsics = IPHONE_DEPTH,
    stride: int = 2,
) -> FloatArray:
    """Reproject metric depth into the color optical frame of a virtual camera.

    Returns a ``target.height x target.width`` depth image in meters (0 = no
    data); overlapping points keep the nearest depth (z-buffer).
    """
    if depth.shape != (depth_intrinsics.height, depth_intrinsics.width):
        raise ValueError(
            f"depth shape {depth.shape} does not match intrinsics "
            f"{depth_intrinsics.height}x{depth_intrinsics.width}"
        )
    sub = depth[::stride, ::stride]
    vs, us = np.nonzero(sub > 0.0)
    z = sub[vs, us].astype(np.float64)
    u = us.astype(np.float64) * stride
    v = vs.astype(np.float64) * stride
    x = (u - depth_intrinsics.cx) / depth_intrinsics.fx * z
    y = (v - depth_intrinsics.cy) / depth_intrinsics.fy * z

    transform = np.asarray(depth_to_color, dtype=np.float64)
    points = np.stack((x, y, z), axis=-1) @ transform[:3, :3].T + transform[:3, 3]
    in_front = points[:, 2] > 1e-6
    points = points[in_front]

    pu = np.rint(target.fx * points[:, 0] / points[:, 2] + target.cx).astype(np.int64)
    pv = np.rint(target.fy * points[:, 1] / points[:, 2] + target.cy).astype(np.int64)
    inside = (pu >= 0) & (pu < target.width) & (pv >= 0) & (pv < target.height)
    flat = pv[inside] * target.width + pu[inside]

    out = np.full(target.height * target.width, np.inf, dtype=np.float64)
    np.minimum.at(out, flat, points[inside, 2])
    out[~np.isfinite(out)] = 0.0
    return out.reshape(target.height, target.width).astype(np.float32)


def depth_to_xyz_map(depth: FloatArray, intrinsics: PinholeIntrinsics) -> FloatArray:
    """HoMMI ``depth2xyzmap``: OpenCV optical-frame xyz per pixel (0 depth -> 0)."""
    height, width = depth.shape
    vs, us = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    z = depth.astype(np.float32)
    x = (us - intrinsics.cx) * z / intrinsics.fx
    y = (vs - intrinsics.cy) * z / intrinsics.fy
    return np.stack((x, y, z), axis=-1).astype(np.float32)


def pad_square_resize_nearest(image: FloatArray, output_size: int) -> FloatArray:
    """HoMMI ``get_image_transform_with_border(mode='pointmap')``."""
    height, width = image.shape[:2]
    side = max(height, width)
    top = (side - height) // 2
    left = (side - width) // 2
    padded = cv2.copyMakeBorder(
        image.astype(np.float32),
        top,
        side - height - top,
        left,
        side - width - left,
        borderType=cv2.BORDER_CONSTANT,
        value=[0.0] * (image.shape[2] if image.ndim == 3 else 1),
    )
    return cv2.resize(padded, (output_size, output_size), interpolation=cv2.INTER_NEAREST)


@dataclass(frozen=True, slots=True)
class HeadObservation:
    rgb: npt.NDArray[np.uint8]  # [S, S, 3]
    pointmap: FloatArray  # [S, S, 3] in the head color optical frame, meters


def process_wrist_frame(frame: CameraFrame, *, output_size: int) -> npt.NDArray[np.uint8]:
    return resample_rgb_to_virtual_square(
        frame.color, frame.color_intrinsics, output_size=output_size
    )


def process_head_frame(frame: CameraFrame, *, output_size: int) -> HeadObservation:
    if frame.depth is None or frame.depth_intrinsics is None or frame.depth_to_color is None:
        raise ValueError("head camera frame must include depth")
    rgb = resample_rgb_to_virtual_square(
        frame.color, frame.color_intrinsics, output_size=output_size
    )
    depth = splat_depth_to_virtual(frame.depth, frame.depth_intrinsics, frame.depth_to_color)
    pointmap = pad_square_resize_nearest(depth_to_xyz_map(depth, IPHONE_DEPTH), output_size)
    return HeadObservation(rgb=rgb, pointmap=pointmap)
