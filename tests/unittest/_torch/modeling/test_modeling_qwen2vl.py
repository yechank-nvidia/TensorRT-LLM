# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionPatchEmbed

from tensorrt_llm._torch.models import modeling_qwen2vl
from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2_5_VisionModel


def test_qwen2_5_vision_patch_projection_matches_conv3d(monkeypatch) -> None:
    patch_embed = Qwen2_5_VisionPatchEmbed(
        patch_size=4,
        temporal_patch_size=2,
        in_channels=3,
        embed_dim=8,
    )
    pixel_values = torch.randn(6, 3 * 2 * 4 * 4)
    expected = patch_embed(pixel_values)

    vision = Qwen2_5_VisionModel.__new__(Qwen2_5_VisionModel)
    torch.nn.Module.__init__(vision)
    vision.patch_embed = patch_embed
    vision.spatial_merge_unit = 1
    vision._rope_position_ids_buffer = None
    vision.full_attn_metadata = object()
    vision.window_attn_metadata = object()
    vision._full_attn_max_seq_len = 6
    vision._window_attn_max_seq_len = 6
    vision.fullatt_block_indexes = []
    vision.blocks = torch.nn.ModuleList()
    vision.merger = torch.nn.Identity()
    vision.get_rotary_pos_emb_window_data = lambda grid: (
        [torch.empty(6, 0)],
        [torch.empty(6, 0)],
        [torch.arange(6)],
        [6],
    )
    vision.prepare_attn_metadata = lambda seq_lens, metadata, **kwargs: metadata
    monkeypatch.setattr(
        modeling_qwen2vl,
        "async_tensor_h2d",
        lambda tensor, *, dtype, device: tensor.to(dtype=dtype, device=device),
    )

    actual = vision(pixel_values, torch.tensor([[1, 2, 3]]))

    torch.testing.assert_close(actual, expected)
