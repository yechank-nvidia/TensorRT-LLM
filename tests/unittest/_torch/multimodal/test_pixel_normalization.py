# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for deferring the processor's rescale and normalize to the encoder device."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

from tensorrt_llm.inputs import pixel_normalization as pn

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _processors(mean=None, std=None):
    kwargs = {} if mean is None else {"image_mean": mean, "image_std": std}
    image = Qwen2VLImageProcessor(min_pixels=32 * 32, max_pixels=96 * 96, **kwargs)
    video = Qwen2VLVideoProcessor(min_pixels=32 * 32, max_pixels=96 * 96, **kwargs)
    return SimpleNamespace(image_processor=image, video_processor=video)


def _random_images(count, seed=0):
    rng = np.random.default_rng(seed)
    sizes = [(37, 53), (64, 64), (91, 30), (120, 77)]
    return [
        rng.integers(0, 256, size=(*sizes[i % len(sizes)], 3), dtype=np.uint8) for i in range(count)
    ]


@pytest.mark.cpu_only
@pytest.mark.parametrize("mean,std", [(None, None), ([0.5] * 3, [0.5] * 3)])
def test_fused_constants_match_processor(mean, std):
    processor = _processors(mean, std).image_processor
    expected_mean, expected_std, still_rescale = processor._fuse_mean_std_and_rescale_factor(
        do_normalize=True,
        image_mean=tuple(processor.image_mean),
        image_std=tuple(processor.image_std),
        do_rescale=True,
        rescale_factor=processor.rescale_factor,
    )
    assert not still_rescale
    assert pn.get_fused_pixel_normalization(processor) == (
        tuple(expected_mean.tolist()),
        tuple(expected_std.tolist()),
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "change",
    [
        {"do_normalize": False},
        {"do_rescale": False},
        {"image_std": [0.5, 0.0, 0.5]},
        {"image_mean": 0.5},
        {"_fuse_mean_std_and_rescale_factor": None},
    ],
)
def test_fused_constants_absent(change):
    processor = _processors().image_processor
    for key, value in change.items():
        setattr(processor, key, value)
    assert pn.get_fused_pixel_normalization(processor) is None


@pytest.mark.cpu_only
def test_plan_accepts_rgb8_images_and_videos():
    processors = _processors()
    images = [Image.new("RGB", (16, 16)), np.zeros((16, 16, 3), dtype=np.uint8)]
    videos = [
        np.zeros((4, 16, 16, 3), dtype=np.uint8),
        [Image.new("RGB", (16, 16))] * 2,
        [np.zeros((16, 16, 3), dtype=np.uint8)] * 2,
    ]
    plan = pn.plan_deferred_pixel_normalization(
        processors, images, videos, {"num_frames": 4, "fps": None}
    )
    assert set(plan) == {"image", "video"}
    assert plan["image"] == pn.get_fused_pixel_normalization(processors.image_processor)
    assert plan["video"] == pn.get_fused_pixel_normalization(processors.video_processor)
    assert set(pn.plan_deferred_pixel_normalization(processors, images, None, None)) == {"image"}
    assert set(pn.plan_deferred_pixel_normalization(processors, None, videos, {})) == {"video"}


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "images,videos,kwargs",
    [
        (None, None, {}),
        ([np.zeros((16, 16, 3), dtype=np.float32)], None, {}),
        ([np.zeros((16, 16), dtype=np.uint8)], None, {}),
        ([np.zeros((16, 16, 4), dtype=np.uint8)], None, {}),
        ([torch.zeros(3, 16, 16)], None, {}),
        (None, [[torch.zeros(3, 16, 16)]], {}),
        (None, [np.zeros((2, 16, 16, 3), dtype=np.float32)], {}),
        (None, [[]], {}),
        ([Image.new("RGB", (16, 16))], None, {"do_normalize": True}),
        ([Image.new("RGB", (16, 16))], None, {"image_mean": [0.4] * 3}),
        ([Image.new("RGB", (16, 16))], None, {"rescale_factor": 1 / 127.5}),
        ([Image.new("RGB", (16, 16))], None, {"do_convert_rgb": False}),
        ([Image.new("RGB", (16, 16))], None, {"images_kwargs": {"do_normalize": True}}),
        # One ineligible item keeps the whole request on the host path.
        ([Image.new("RGB", (16, 16)), np.zeros((16, 16, 3), np.float32)], None, {}),
        ([Image.new("RGB", (16, 16))], [[torch.zeros(3, 16, 16)]], {}),
    ],
)
def test_plan_keeps_host_path(images, videos, kwargs):
    assert pn.plan_deferred_pixel_normalization(_processors(), images, videos, kwargs) is None


@pytest.mark.cpu_only
def test_plan_follows_rgb_conversion():
    processors = _processors()
    gray = [Image.new("L", (16, 16)), Image.new("RGBA", (16, 16))]
    processors.image_processor.do_convert_rgb = True
    assert pn.plan_deferred_pixel_normalization(processors, gray, None, {}) is not None
    processors.image_processor.do_convert_rgb = False
    assert pn.plan_deferred_pixel_normalization(processors, gray, None, {}) is None


@pytest.mark.cpu_only
def test_plan_requires_processor_support():
    processors = _processors()
    processors.video_processor = SimpleNamespace(do_rescale=True, do_normalize=True)
    images = [Image.new("RGB", (16, 16))]
    videos = [np.zeros((2, 16, 16, 3), dtype=np.uint8)]
    assert pn.plan_deferred_pixel_normalization(processors, images, None, {}) is not None
    assert pn.plan_deferred_pixel_normalization(processors, images, videos, {}) is None


def _host_and_raw(processor, items, video):
    key = "pixel_values_videos" if video else "pixel_values"
    arg = {"videos": items} if video else {"images": items}
    host = processor(**arg, return_tensors="pt")[key]
    raw = processor(**arg, do_rescale=False, do_normalize=False, return_tensors="pt")[key]
    return host, raw


@pytest.mark.cpu_only
@pytest.mark.parametrize("mean,std", [(None, None), ([0.5] * 3, [0.5] * 3)])
@pytest.mark.parametrize("video", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_host_normalization_matches_processor(mean, std, video, dtype):
    processors = _processors(mean, std)
    processor = processors.video_processor if video else processors.image_processor
    images = _random_images(4)
    items = [np.stack(images[:1] * 3)] if video else images
    host, raw = _host_and_raw(processor, items, video)
    assert raw.dtype == torch.uint8
    normalization = pn.get_fused_pixel_normalization(processor)
    assert torch.equal(pn.normalize_pixel_patches(raw, normalization, dtype), host.to(dtype))


@pytest.mark.cpu_only
def test_normalize_rejects_rows_without_whole_channels():
    with pytest.raises(ValueError, match="channel-major"):
        pn.normalize_pixel_patches(
            torch.zeros(2, 7, dtype=torch.uint8), ((0.0,) * 3, (1.0,) * 3), torch.float32
        )


@pytest.mark.cpu_only
def test_pack_pixel_patches():
    norm = ((127.5,) * 3, (127.5,) * 3)
    raw = torch.arange(12, dtype=torch.uint8).view(2, 6)
    packed = pn.pack_pixel_patches("pixel_values", raw, norm, torch.bfloat16)
    assert packed["pixel_values"] is raw
    assert packed[pn.PIXEL_NORMALIZATION_KEY] == norm
    host = pn.pack_pixel_patches("pixel_values", raw.float(), norm, torch.bfloat16)
    assert set(host) == {"pixel_values"}
    assert torch.equal(host["pixel_values"], pn.normalize_pixel_patches(raw, norm, torch.bfloat16))
    plain = pn.pack_pixel_patches("pixel_values", raw.float(), None, torch.bfloat16)
    assert set(plain) == {"pixel_values"} and plain["pixel_values"].dtype == torch.bfloat16


@pytest.mark.cpu_only
def test_concat_keeps_order_and_groups_runs(monkeypatch):
    half = ((127.5,) * 3, (127.5,) * 3)
    other = ((100.0,) * 3, (50.0,) * 3)
    raw = [torch.randint(0, 256, (n, 6), dtype=torch.uint8) for n in (2, 3, 1, 4)]
    ready = torch.randn(5, 6).to(torch.bfloat16)
    items = [(raw[0], half), (raw[1], half), (ready, None), (raw[2], other), (raw[3], half)]
    calls = []
    original = pn.normalize_pixel_patches
    monkeypatch.setattr(
        pn,
        "normalize_pixel_patches",
        lambda pixels, norm, dtype: calls.append(pixels.shape[0]) or original(pixels, norm, dtype),
    )
    out = pn.concat_pixel_patches(items, torch.bfloat16)
    expected = torch.cat(
        [
            original(torch.cat(raw[:2]), half, torch.bfloat16),
            ready,
            original(raw[2], other, torch.bfloat16),
            original(raw[3], half, torch.bfloat16),
        ]
    )
    assert torch.equal(out, expected)
    assert calls == [5, 1, 4]
    assert pn.concat_pixel_patches([(ready, None)], None) is ready
    with pytest.raises(ValueError):
        pn.concat_pixel_patches([(raw[0], None)], torch.bfloat16)
    with pytest.raises(ValueError):
        pn.concat_pixel_patches([(raw[0], half)], None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "mean,std", [(None, None), ([0.5] * 3, [0.5] * 3), ([0.2, 0.7, 0.1], [0.3, 0.9, 0.05])]
)
@pytest.mark.parametrize("video", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_device_normalization_is_bitwise_identical(mean, std, video, dtype):
    """The encoder-side map reproduces the host processor's values bit for bit."""
    processors = _processors(mean, std)
    processor = processors.video_processor if video else processors.image_processor
    images = _random_images(8, seed=1)
    items = [np.stack([image[:30, :30] for image in images[:4]])] if video else images
    host, raw = _host_and_raw(processor, items, video)
    normalization = pn.get_fused_pixel_normalization(processor)
    device = pn.concat_pixel_patches([(raw.cuda(), normalization)], dtype)
    assert device.is_cuda and device.dtype == dtype
    assert torch.equal(device.cpu(), host.to(dtype))
    # Every uint8 value in every channel, not only the ones random images hit.
    channels = len(normalization[0])
    values = torch.arange(256, dtype=torch.uint8).repeat(channels, 1).reshape(1, -1)
    expected = (
        values.reshape(channels, 256).float() - torch.tensor(normalization[0]).view(-1, 1)
    ) / torch.tensor(normalization[1]).view(-1, 1)
    got = pn.normalize_pixel_patches(values.cuda(), normalization, dtype)
    assert torch.equal(got.cpu(), expected.reshape(1, -1).to(dtype))
