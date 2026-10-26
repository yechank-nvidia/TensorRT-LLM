# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Move the per-channel rescale and normalize of patch rows from the host to the encoder device.

Image processors that rescale by ``rescale_factor`` and then normalize with a per-channel
mean and std fold the two steps into one affine map and apply it in FP32 before the
patches are cast to the model dtype. Resizing and patchifying do not depend on that map,
so the processor can instead emit the raw ``uint8`` patch rows and the encoder can apply
the identical FP32 map on its own device. The model-visible values are unchanged bit for
bit, the host skips the FP32 pass, and the host-to-device copy carries one byte per value.

The deferral is all-or-nothing per request: it applies only when every image and video
frame is 8-bit RGB, the processors use the fused rescale and normalize, and the request
does not override any of the normalization settings. Otherwise the processor runs as
before. Image items and video frames take the same path.

A modality payload whose normalization is deferred carries its fused constants under
``PIXEL_NORMALIZATION_KEY`` next to the ``uint8`` patch rows. The patch rows must be
channel-major: each row holds all values of channel 0, then of channel 1, and so on.

A model opts in with ``MultimodalModelMixin.supports_device_pixel_normalization`` once its
encoder batches pixels through ``concat_pixel_patches``; ``create_input_processor`` then
enables the deferral on that model's input processor. There is no user setting.
"""

from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

PIXEL_NORMALIZATION_KEY = "pixel_normalization"
"""Modality-payload key holding ``(mean, std)``: per-channel FP32 values as float tuples."""

PixelNormalization = Tuple[Tuple[float, ...], Tuple[float, ...]]

# Request kwargs that change the normalization itself. Any of them keeps the host path.
_NORMALIZATION_KWARGS = frozenset(
    {"do_rescale", "rescale_factor", "do_normalize", "image_mean", "image_std", "do_convert_rgb"}
)


def get_fused_pixel_normalization(processor: Any) -> Optional[PixelNormalization]:
    """Return the fused per-channel ``(mean, std)`` that ``processor`` applies, or None.

    The values come from the processor's own fusion helper, so they match what it would
    use bit for bit. Processors without that helper, or that do not both rescale and
    normalize, return None.
    """
    fuse = getattr(processor, "_fuse_mean_std_and_rescale_factor", None)
    if not callable(fuse):
        return None
    if not (getattr(processor, "do_rescale", False) and getattr(processor, "do_normalize", False)):
        return None
    mean = getattr(processor, "image_mean", None)
    std = getattr(processor, "image_std", None)
    rescale_factor = getattr(processor, "rescale_factor", None)
    if not isinstance(mean, (list, tuple)) or not isinstance(std, (list, tuple)):
        return None
    if len(mean) == 0 or len(mean) != len(std) or not rescale_factor:
        return None
    fused_mean, fused_std, still_rescale = fuse(
        do_normalize=True,
        image_mean=tuple(mean),
        image_std=tuple(std),
        do_rescale=True,
        rescale_factor=rescale_factor,
    )
    if still_rescale or not isinstance(fused_mean, torch.Tensor):
        return None
    if fused_mean.dtype != torch.float32 or fused_std.dtype != torch.float32:
        return None
    if bool((fused_std == 0).any()):
        return None
    return tuple(fused_mean.tolist()), tuple(fused_std.tolist())


def _is_rgb8_image(image: Any, convert_rgb: bool) -> bool:
    if isinstance(image, Image.Image):
        return image.mode == "RGB" or convert_rgb
    return (
        isinstance(image, np.ndarray)
        and image.dtype == np.uint8
        and image.ndim == 3
        and image.shape[-1] == 3
    )


def _is_rgb8_video(frames: Any, convert_rgb: bool) -> bool:
    if isinstance(frames, np.ndarray):
        return frames.dtype == np.uint8 and frames.ndim == 4 and frames.shape[-1] == 3
    if isinstance(frames, (list, tuple)) and frames:
        return all(_is_rgb8_image(frame, convert_rgb) for frame in frames)
    return False


def plan_deferred_pixel_normalization(
    hf_processor: Any,
    images: Optional[Sequence[Any]],
    videos: Optional[Sequence[Any]],
    processor_kwargs: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, PixelNormalization]]:
    """Decide whether ``hf_processor`` can leave this request's normalization to the encoder.

    Args:
        hf_processor: The multimodal HF processor, with ``image_processor`` and
            ``video_processor`` attributes.
        images: The request's images, or None.
        videos: The request's videos, each a frame sequence or a ``(T, H, W, 3)`` array.
        processor_kwargs: The request's processor kwargs.

    Returns:
        ``{modality: (mean, std)}`` for every modality present, or None to keep the
        host path. When a mapping is returned, call ``hf_processor`` with
        ``do_rescale=False, do_normalize=False``.
    """
    if not images and not videos:
        return None
    for key, value in (processor_kwargs or {}).items():
        if key in _NORMALIZATION_KWARGS or isinstance(value, Mapping):
            return None
    plan: Dict[str, PixelNormalization] = {}
    for modality, items, attr, is_rgb8 in (
        ("image", images, "image_processor", _is_rgb8_image),
        ("video", videos, "video_processor", _is_rgb8_video),
    ):
        if not items:
            continue
        processor = getattr(hf_processor, attr, None)
        normalization = get_fused_pixel_normalization(processor)
        if normalization is None:
            return None
        convert_rgb = bool(getattr(processor, "do_convert_rgb", False))
        if not all(is_rgb8(item, convert_rgb) for item in items):
            return None
        plan[modality] = normalization
    return plan


def normalize_pixel_patches(
    pixels: torch.Tensor, normalization: PixelNormalization, dtype: torch.dtype
) -> torch.Tensor:
    """Apply the fused rescale and normalize to channel-major patch rows.

    Computes ``(pixel - mean[c]) / std[c]`` in FP32 and casts the result to ``dtype``,
    which is the operation sequence of the host processor. Runs on ``pixels``' device.
    """
    mean, std = normalization
    channels = len(mean)
    if pixels.dim() != 2 or pixels.shape[1] % channels != 0:
        raise ValueError(
            f"Expected channel-major patch rows with a multiple of {channels} values per row, "
            f"got shape {tuple(pixels.shape)}."
        )
    mean_t, std_t = _normalization_tensors(normalization, pixels.device)
    values = pixels.reshape(pixels.shape[0], channels, -1).to(torch.float32, copy=True)
    values.sub_(mean_t).div_(std_t)
    return values.reshape(pixels.shape[0], -1).to(dtype)


@lru_cache(maxsize=16)
def _normalization_tensors(
    normalization: PixelNormalization, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean, std = normalization
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1).to(device)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, -1, 1).to(device)
    return mean_t, std_t


def pack_pixel_patches(
    pixel_key: str,
    pixels: torch.Tensor,
    normalization: Optional[PixelNormalization],
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """Build the modality-payload entries for processor output ``pixels``.

    With ``normalization``, ``uint8`` rows stay raw and carry the constants for the
    encoder. Any other output is finished on the host with the same map, so a processor
    that did not keep ``uint8`` still yields the values of the host path.
    """
    if normalization is None:
        return {pixel_key: pixels.to(dtype)}
    if pixels.dtype == torch.uint8:
        return {pixel_key: pixels, PIXEL_NORMALIZATION_KEY: normalization}
    return {pixel_key: normalize_pixel_patches(pixels, normalization, dtype)}


def concat_pixel_patches(
    items: Sequence[Tuple[torch.Tensor, Optional[PixelNormalization]]],
    dtype: Optional[torch.dtype],
) -> torch.Tensor:
    """Concatenate per-request patch rows, normalizing deferred ``uint8`` rows to ``dtype``.

    Adjacent raw items with the same constants are normalized together. Items that are
    already floating point pass through unchanged, as they did before deferral existed.
    """
    pieces: List[torch.Tensor] = []
    run: List[torch.Tensor] = []
    run_normalization: Optional[PixelNormalization] = None

    def flush() -> None:
        if run:
            raw = run[0] if len(run) == 1 else torch.cat(run, dim=0)
            pieces.append(normalize_pixel_patches(raw, run_normalization, dtype))
            run.clear()

    for pixels, normalization in items:
        if pixels.dtype == torch.uint8:
            if normalization is None or dtype is None:
                raise ValueError(
                    "Raw uint8 patch rows need their normalization and a target dtype."
                )
            if normalization != run_normalization:
                flush()
                run_normalization = normalization
            run.append(pixels)
        else:
            flush()
            pieces.append(pixels)
    flush()
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
