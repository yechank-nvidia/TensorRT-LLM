# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import base64
from io import BytesIO
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import soundfile
import torch
from PIL import Image

from tensorrt_llm.inputs import MultimodalDataTracker
from tensorrt_llm.inputs.media_io import (
    AudioMediaIO,
    BaseMediaIO,
    ImageMediaIO,
    VideoMediaIO,
    convert_image_mode,
)
from tensorrt_llm.serve.chat_utils import parse_chat_message_content_part

pytestmark = pytest.mark.cpu_only


class CustomError(Exception):
    pass


@pytest.mark.parametrize(
    ("mode", "image_format"),
    [("RGB", "JPEG"), ("L", "JPEG"), ("CMYK", "JPEG"), ("RGBA", "PNG")],
)
def test_image_loading_preserves_rgb_pixels(mode, image_format, tmp_path):
    shape = (7, 8) if mode == "L" else (7, 8, len(mode))
    pixels = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    image = Image.fromarray(pixels, mode=mode)

    buffer = BytesIO()
    image.save(buffer, format=image_format)
    encoded = buffer.getvalue()
    image_path = tmp_path / f"input.{image_format.lower()}"
    image_path.write_bytes(encoded)
    expected = np.asarray(convert_image_mode(Image.open(BytesIO(encoded)), "RGB"))
    numpy_io = ImageMediaIO(format="np")
    tensor_io = ImageMediaIO(format="pt")

    numpy_from_bytes = numpy_io.load_bytes(encoded)
    numpy_from_base64 = numpy_io.load_base64(
        f"image/{image_format.lower()}", base64.b64encode(encoded).decode()
    )
    tensor_from_bytes = tensor_io.load_bytes(encoded)
    tensor_from_base64 = tensor_io.load_base64(
        f"image/{image_format.lower()}", base64.b64encode(encoded).decode()
    )
    numpy_from_file = numpy_io.load_file(str(image_path))
    tensor_from_file = tensor_io.load_file(str(image_path))
    expected_tensor = (
        torch.from_numpy(np.array(expected, copy=True))
        .permute(2, 0, 1)
        .to(dtype=torch.get_default_dtype())
        .div_(255)
    )

    np.testing.assert_array_equal(numpy_from_bytes, expected)
    np.testing.assert_array_equal(numpy_from_base64, expected)
    np.testing.assert_array_equal(numpy_from_file, expected)
    torch.testing.assert_close(tensor_from_bytes, expected_tensor, rtol=0, atol=0)
    torch.testing.assert_close(tensor_from_base64, expected_tensor, rtol=0, atol=0)
    torch.testing.assert_close(tensor_from_file, expected_tensor, rtol=0, atol=0)
    assert numpy_from_bytes.flags.c_contiguous
    assert numpy_from_base64.flags.c_contiguous
    assert numpy_from_file.flags.c_contiguous
    assert tensor_from_bytes.is_contiguous()
    assert tensor_from_base64.is_contiguous()
    assert tensor_from_file.is_contiguous()


def test_audio_loading_returns_float32(tmp_path):
    samples = np.linspace(-1, 1, 32, dtype=np.float32)
    buffer = BytesIO()
    soundfile.write(buffer, samples, 16_000, format="WAV", subtype="PCM_16")
    encoded = buffer.getvalue()
    path = tmp_path / "input.wav"
    path.write_bytes(encoded)
    loader = AudioMediaIO()

    from_bytes, bytes_rate = loader.load_bytes(encoded)
    from_base64, base64_rate = loader.load_base64("audio/wav", base64.b64encode(encoded).decode())
    from_file, file_rate = loader.load_file(str(path))

    assert from_bytes.dtype == np.float32
    np.testing.assert_array_equal(from_base64, from_bytes)
    np.testing.assert_array_equal(from_file, from_bytes)
    assert bytes_rate == base64_rate == file_rate == 16_000


class TestMultimodalLoadErrorPropagation:
    """Verify that errors from multimodal loading propagate."""

    @pytest.fixture
    def mm_tracker(self):
        return MultimodalDataTracker(model_type="dummy")

    @pytest.mark.parametrize(
        "part, patch_target",
        [
            (
                {"type": "image_url", "image_url": {"url": "http://bad-url/img.png"}},
                "tensorrt_llm.inputs.media_io.ImageMediaIO.async_load",
            ),
            (
                {"type": "video_url", "video_url": {"url": "http://bad-url/vid.mp4"}},
                "tensorrt_llm.inputs.media_io.VideoMediaIO.async_load",
            ),
            (
                {"type": "audio_url", "audio_url": {"url": "http://bad-url/aud.wav"}},
                "tensorrt_llm.inputs.media_io.AudioMediaIO.async_load",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_loader_exception_propagates(self, mm_tracker, part, patch_target):
        """Exceptions from async loaders must propagate, not be swallowed."""
        with patch(patch_target, new_callable=AsyncMock, side_effect=CustomError):
            result = parse_chat_message_content_part(part, mm_tracker)
            assert result is not None
            with pytest.raises(CustomError):
                await result["data"]

    @pytest.mark.asyncio
    async def test_image_embeds_exception_propagates(self, mm_tracker):
        """Exceptions from image embed decoding must propagate."""
        part = {"type": "image_embeds", "image_embeds": {"data": "notbase64"}}
        with patch(
            "tensorrt_llm.serve.chat_utils.load_base64_image_embeds",
            side_effect=CustomError,
        ):
            result = parse_chat_message_content_part(part, mm_tracker)
            assert result is not None
            with pytest.raises(CustomError):
                await result["data"]


class TestVideoMediaIOMergeInteraction:
    """`VideoMediaIO.merge_kwargs` couples `fps` and `num_frames`."""

    @pytest.mark.parametrize(
        "runtime, expected",
        [
            ({"num_frames": 32}, {"num_frames": 32}),
            ({"fps": 4}, {"fps": 4}),
            ({"num_frames": 32, "fps": 4}, {"num_frames": 32, "fps": 4}),
        ],
    )
    def test_overriding_one_drops_partner_unless_both_given(self, runtime, expected):
        server = {"num_frames": 8, "fps": 1}
        assert VideoMediaIO.merge_kwargs(server, runtime) == expected

    def test_unrelated_request_key_does_not_trigger_drop(self):
        merged = VideoMediaIO.merge_kwargs(
            {"num_frames": 8, "fps": 1},
            {"format": "pt"},
        )
        assert merged == {"num_frames": 8, "fps": 1, "format": "pt"}

    @pytest.mark.parametrize("media_io_cls", [BaseMediaIO, ImageMediaIO, AudioMediaIO])
    def test_non_video_classes_use_plain_shallow_merge(self, media_io_cls):
        merged = media_io_cls.merge_kwargs(
            {"num_frames": 8, "fps": 1},
            {"num_frames": 32},
        )
        assert merged == {"num_frames": 32, "fps": 1}
